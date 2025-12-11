import copy
import logging
from dataclasses import dataclass

from typing import Dict, Sequence

import torch
import transformers
from datasets import load_dataset
from torch import nn

from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import Trainer
import ipdb

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_name) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_name, "main")
    train_set = dataset['train']
    eval_set = dataset['test']
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    eval_dataset = SupervisedDataset(raw_data=eval_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


class CustomTrainer(Trainer):
    def __init__(self, *args_, ref_model=None, target_model=None, 
                 use_on_policy=True, kl_type="forward", max_new_tokens=512, 
                 temperature=1.0, top_k=0, top_p=0.0, **kwargs):
        """
        Args:
            use_on_policy: 是否使用 on-policy distillation。True 为 on-policy，False 为 off-policy
            kl_type: KL 散度类型，"forward" 或 "reverse"
                - "forward": KL(P||Q) = sum(P * log(P/Q))，使用 KLDivLoss(q_log, p)
                - "reverse": KL(Q||P) = sum(Q * log(Q/P))，使用 KLDivLoss(p_log, q)
            max_new_tokens: on-policy 采样时的最大生成长度
            temperature: 采样温度
            top_k: top-k 采样参数
            top_p: top-p 采样参数
        """
        super().__init__(*args_, **kwargs)
        self.ref_model = ref_model
        self.target_model = target_model
        self.use_on_policy = use_on_policy
        self.kl_type = kl_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        from trl.trainer.utils import prepare_deepspeed
        if self.ref_model is not None:
            self.ref_model = prepare_deepspeed(
                self.ref_model, self.args.per_device_train_batch_size, self.args.fp16,
                self.args.bf16
            )
            self.ref_model.eval()
        if self.target_model is not None:
            self.target_model = prepare_deepspeed(
                self.target_model, self.args.per_device_train_batch_size, self.args.fp16,
                self.args.bf16
            )
            self.target_model.eval()
        
        # 注册 hook 来监控梯度计算
        self._loss_hook_registered = False

    def _sample_from_model(self, model, input_ids, attention_mask, max_new_tokens):
        """使用模型进行自回归采样生成序列（on-policy 采样）
        
        使用 HuggingFace 的 model.generate() 方法，更稳定可靠
        """
        was_training = model.training
        model.eval()  # 采样时使用 eval 模式以确保稳定性
        
        batch_size = input_ids.shape[0]
        
        # 设置 pad_token_id（如果没有的话）
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id
        
        # 构建 bad_words_ids 列表，阻止生成 "Human" 等对话标记
        bad_words_ids = []
        for bad_word in ["Human", "Human:", "Assistant", "Assistant:"]:
            encoded = self.processing_class.encode(bad_word, add_special_tokens=False)
            if encoded:
                bad_words_ids.append(encoded)
        
        # ====== 将右 padding 转换为左 padding（生成时推荐）======
        # 计算每个序列的实际长度
        seq_lengths = attention_mask.sum(dim=1)  # [batch_size]
        max_seq_len = input_ids.shape[1]
        
        # 创建左 padding 的 input_ids 和 attention_mask
        left_padded_input_ids = torch.full_like(input_ids, pad_token_id)
        left_padded_attention_mask = torch.zeros_like(attention_mask)
        
        for i in range(batch_size):
            seq_len = seq_lengths[i].item()
            # 将有效 token 移到右边
            left_padded_input_ids[i, max_seq_len - seq_len:] = input_ids[i, :seq_len]
            left_padded_attention_mask[i, max_seq_len - seq_len:] = 1
        
        # 获取模型的最大序列长度限制
        max_length = getattr(self.processing_class, 'model_max_length', 2048)
        current_length = left_padded_input_ids.shape[1]
        # 限制 max_new_tokens 以确保不超过模型最大长度
        effective_max_new_tokens = min(max_new_tokens, max_length - current_length)
        
        if effective_max_new_tokens <= 0:
            # 如果 prompt 已经达到最大长度，直接返回
            if was_training:
                model.train()
            return input_ids, attention_mask
        
        with torch.no_grad():
            # 禁用混合精度，使用 fp32 进行采样以避免数值溢出（低 temperature 会放大 logits）
            with torch.cuda.amp.autocast(enabled=False):
                # 使用 model.generate() 进行采样（使用左 padding 的输入）
                generated_ids = model.generate(
                    input_ids=left_padded_input_ids,
                    attention_mask=left_padded_attention_mask,
                    max_new_tokens=effective_max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else 1.0,
                    top_k=self.top_k if self.top_k > 0 else 50,
                    top_p=self.top_p if self.top_p > 0 else 1.0,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    bad_words_ids=bad_words_ids if bad_words_ids else None,  # 阻止生成这些词
                    use_cache=True,  # 使用 KV cache 加速生成
                )
        
        # 生成对应的 attention mask
        generated_attention_mask = torch.ones_like(generated_ids, dtype=attention_mask.dtype)
        # 对于 padding 部分设为 0（左边的 padding）
        generated_attention_mask[generated_ids == pad_token_id] = 0
        
        # 恢复模型原来的训练状态
        if was_training:
            model.train()
        
        return generated_ids, generated_attention_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 在评估模式下，使用 off-policy 方式（不进行采样生成）
        # 因为评估时需要确定性的结果
        is_training = model.training
        
        if self.use_on_policy and is_training:
            # ========== On-Policy Distillation ==========
            # 1. 提取 prompt（source 部分）
            labels = inputs.get("labels", None)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            # 找到 prompt 的结束位置（labels 中 IGNORE_INDEX 的部分是 prompt）
            if labels is not None:
                # labels 中 IGNORE_INDEX 的位置是 prompt 部分
                # 找到每个样本中最后一个 IGNORE_INDEX 的位置（prompt 的结束位置）
                prompt_mask = labels == IGNORE_INDEX
                # 计算每个样本的 prompt 长度（最后一个 IGNORE_INDEX 的位置 + 1）
                prompt_lengths = []
                for i in range(labels.shape[0]):
                    # 找到最后一个 IGNORE_INDEX 的位置
                    ignore_positions = (labels[i] == IGNORE_INDEX).nonzero(as_tuple=True)[0]
                    if len(ignore_positions) > 0:
                        prompt_len = ignore_positions[-1].item() + 1
                    else:
                        # 如果没有 IGNORE_INDEX，使用整个序列
                        prompt_len = labels.shape[1]
                    prompt_lengths.append(prompt_len)
                prompt_lengths = torch.tensor(prompt_lengths, device=input_ids.device)
            else:
                # 如果没有 labels，假设整个输入都是 prompt
                if attention_mask is not None:
                    prompt_lengths = attention_mask.sum(dim=1)
                else:
                    prompt_lengths = torch.full((input_ids.shape[0],), input_ids.shape[1], device=input_ids.device)
            
            # 提取 prompt（每个样本的前 prompt_len 个 tokens）
            batch_size = input_ids.shape[0]
            max_prompt_len = prompt_lengths.max().item()
            prompt_ids = input_ids[:, :max_prompt_len].clone()
            if attention_mask is not None:
                prompt_attention_mask = attention_mask[:, :max_prompt_len].clone()
            else:
                prompt_attention_mask = torch.ones((batch_size, max_prompt_len), device=input_ids.device, dtype=torch.long)
            
            # 对于长度不足的样本，用 pad_token 填充
            for i in range(batch_size):
                prompt_len = prompt_lengths[i].item()
                if prompt_len < max_prompt_len:
                    prompt_ids[i, prompt_len:] = self.processing_class.pad_token_id
                    prompt_attention_mask[i, prompt_len:] = 0
            
            # 2. 使用 draft 模型（学生）进行自回归采样生成序列
            # 限制 max_new_tokens 以确保不超过模型最大长度
            prompt_len = prompt_ids.shape[1]
            max_length = getattr(self.processing_class, 'model_max_length', 2048)
            effective_max_new_tokens = min(self.max_new_tokens, max_length - prompt_len)
            
            if effective_max_new_tokens <= 0:
                # 如果 prompt 已经达到最大长度，使用 off-policy 方式
                labels = inputs.get("labels", None)
                if labels is not None:
                    inputs["labels"] = labels
                else:
                    inputs["labels"] = None
                # 回退到 off-policy
                outputs = model(**inputs)
                with torch.no_grad():
                    target_outputs = self.target_model(**inputs, use_cache=False)
                logits = outputs["logits"]
                target_logits = target_outputs["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_target_logits = target_logits[..., :-1, :].contiguous()
                shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
                shift_target_logits = shift_target_logits.view(-1, shift_target_logits.shape[-1])
                shift_logits = shift_logits.float()
                shift_target_logits = shift_target_logits.float()
                p = F.softmax(shift_target_logits, dim=-1)
                q_log = F.log_softmax(shift_logits, dim=-1)
                loss_fct = nn.KLDivLoss(reduction='batchmean')
                loss = loss_fct(q_log, p)
                return (loss, outputs) if return_outputs else loss
            
            generated_ids, generated_attention_mask = self._sample_from_model(
                model, prompt_ids, prompt_attention_mask, effective_max_new_tokens
            )
            # decode the generated ids
            generated_ids_decoded = self.processing_class.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 3. 使用 target 模型（教师）对生成的序列进行评分
            # 注意：generated_ids 是左 padding 的：[PAD PAD prompt new_tokens]
            # prompt_ids 是右 padding 的：[prompt PAD PAD]
            # 
            # 简化处理：直接对完整的 generated_ids 做 forward
            # 这样避免了左/右 padding 对齐的复杂性
            ids_for_logits = generated_ids
            mask_for_logits = generated_attention_mask
            
            # 计算 prompt 的实际长度（不含 padding）
            actual_prompt_len = prompt_attention_mask.sum(dim=1).max().item()
            # 新生成的 token 数量 = 总长度 - 原 padded 长度
            padded_prompt_len = prompt_ids.shape[1]
            new_tokens_len = generated_ids.shape[1] - padded_prompt_len
            
            
            # 在左 padding 序列中，有效内容的起始位置
            # generated_ids 结构: [PAD...PAD prompt new_tokens]
            # 有效内容从 (total_len - actual_prompt_len - new_tokens_len) 开始
            total_len = generated_ids.shape[1]
            content_start = total_len - actual_prompt_len - new_tokens_len
            
            # 为了计算 KL，我们只关心 prompt 最后部分 + 新生成的 tokens 对应的 logits
            context_len = min(32, actual_prompt_len)  # 只保留最后 32 个 tokens 作为上下文
            
            # 学生模型需要计算梯度，教师模型不需要
            student_outputs = model(input_ids=ids_for_logits, attention_mask=mask_for_logits, use_cache=False)
            student_logits = student_outputs.logits
            
            with torch.no_grad():
                teacher_outputs = self.target_model(input_ids=ids_for_logits, attention_mask=mask_for_logits, use_cache=False)
                teacher_logits = teacher_outputs.logits
            
            # 4. 计算 KL 散度
            # 序列结构（左 padding）: [PAD...PAD prompt new_tokens]
            # logits[i] 预测的是 token[i+1]
            # 我们只对 new_tokens 部分计算 KL loss
            # 
            # 预测 new_tokens 的 logits 位置:
            #   - 第一个 new_token 由位置 (content_start + actual_prompt_len - 1) 的 logits 预测
            #   - 最后一个 new_token 由位置 (total_len - 2) 的 logits 预测
            gen_logits_start = content_start + actual_prompt_len - 1
            gen_logits_end = total_len - 1  # 不包含最后一个位置（因为它预测的是 EOS 之后）
            
            
            # 确保有有效的 logits 来计算
            if gen_logits_end <= gen_logits_start:
                # 没有生成新 tokens，跳过这个 batch 或返回 0 loss
                print("[Debug] No new tokens generated, returning zero loss", flush=True)
                loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
                return (loss, student_outputs) if return_outputs else loss
            
            student_logits_gen = student_logits[:, gen_logits_start:gen_logits_end, :].contiguous()
            teacher_logits_gen = teacher_logits[:, gen_logits_start:gen_logits_end, :].contiguous()
            
            # 展平
            student_logits_flat = student_logits_gen.view(-1, student_logits_gen.shape[-1])
            teacher_logits_flat = teacher_logits_gen.view(-1, teacher_logits_gen.shape[-1])
            
            # 转换为 float
            student_logits_flat = student_logits_flat.float()
            teacher_logits_flat = teacher_logits_flat.float()
            
            # 计算 KL 散度
            if self.kl_type == "forward":
                # Forward KL: KL(P||Q) = sum(P * log(P/Q))
                # 使用 KLDivLoss(q_log, p)，其中 q_log = log(Q), p = P
                p = F.softmax(teacher_logits_flat, dim=-1)  # P (teacher)
                q_log = F.log_softmax(student_logits_flat, dim=-1)  # log(Q) (student)
                loss_fct = nn.KLDivLoss(reduction='batchmean')
                loss = loss_fct(q_log, p)
            elif self.kl_type == "reverse":
                # Reverse KL: KL(Q||P) = sum(Q * log(Q/P))
                # 使用 KLDivLoss(p_log, q)，其中 p_log = log(P), q = Q
                q = F.softmax(student_logits_flat, dim=-1)  # Q (student)
                p_log = F.log_softmax(teacher_logits_flat, dim=-1)  # log(P) (teacher)
                loss_fct = nn.KLDivLoss(reduction='batchmean')
                loss = loss_fct(p_log, q)
            else:
                raise ValueError(f"Unknown KL type: {self.kl_type}. Must be 'forward' or 'reverse'")
            
            # 使用 student_outputs 作为 outputs（它已经是正确的格式）
            outputs = student_outputs
            
            # 应用后处理（与 off-policy 分支保持一致）
            if (
                    self.args.average_tokens_across_devices
                    and (self.model_accepts_loss_kwargs or self.compute_loss_func)
                    and num_items_in_batch is not None
            ):
                loss *= self.accelerator.num_processes
            
            # 注册 hook 来监控 loss 的梯度计算
            if not self._loss_hook_registered:
                def loss_backward_hook(grad):
                    return grad
                
                if loss.requires_grad:
                    loss.register_hook(loss_backward_hook)
                    self._loss_hook_registered = True
            
            # 检查是否需要返回 outputs
            if return_outputs:
                return (loss, outputs)
            else:
                return loss
            
        else:
            # 评估模式或 off-policy 模式：使用标准的 forward pass
            # ========== Off-Policy Distillation (原始实现) ==========
            labels = inputs.pop("labels")[:, 1:]

            outputs = model(**inputs)
            with torch.no_grad():
                target_outputs = self.target_model(**inputs, use_cache=False)

            logits = outputs["logits"]
            target_logits = target_outputs["logits"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_target_logits = target_logits[..., :-1, :].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_target_logits = shift_target_logits.view(-1, shift_target_logits.shape[-1])
            mask = labels.ne(IGNORE_INDEX).flatten().unsqueeze(-1)

            shift_logits = torch.masked_select(shift_logits, mask=mask).view(-1, shift_logits.shape[-1])
            shift_target_logits = torch.masked_select(shift_target_logits, mask=mask).view(-1,
                                                                                           shift_target_logits.shape[-1])

            shift_logits = shift_logits.float()
            shift_target_logits = shift_target_logits.float()

            p = F.softmax(shift_target_logits, dim=-1)
            q_log = F.log_softmax(shift_logits, dim=-1)

            if num_items_in_batch is not None:
                loss_fct = nn.KLDivLoss(reduction="sum")
                loss = loss_fct(q_log, p)
                loss = loss / num_items_in_batch
            else:
                loss_fct = nn.KLDivLoss(reduction='batchmean')
                loss = loss_fct(q_log, p)

        if (
                self.args.average_tokens_across_devices
                and (self.model_accepts_loss_kwargs or self.compute_loss_func)
                and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        result = (loss, outputs) if return_outputs else loss
        return result
