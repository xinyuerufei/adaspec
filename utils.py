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
        
        返回生成的序列、attention mask、以及采样时的 log probabilities
        """
        was_training = model.training
        model.eval()  # 采样时使用 eval 模式以确保稳定性
        
        batch_size = input_ids.shape[0]
        
        # 设置 pad_token_id（如果没有的话）
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id
        
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
            # 返回空的 logprobs
            empty_logprobs = torch.zeros((batch_size, 0), device=input_ids.device)
            return input_ids, attention_mask, empty_logprobs, 0
        
        # ====== 自回归采样并记录 log probabilities ======
        generated_tokens = []
        sampled_logprobs = []
        
        current_ids = left_padded_input_ids.clone()
        current_mask = left_padded_attention_mask.clone()
        
        # 记录每个样本是否已经结束生成
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        
        with torch.no_grad():
            for step in range(effective_max_new_tokens):
                # Forward pass
                outputs = model(input_ids=current_ids, attention_mask=current_mask, use_cache=False)
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
                
                # 用原始 logits 计算 log probabilities（不带 temperature）
                # 这样和后面计算 teacher_logprobs、new_logprobs 保持一致
                log_probs = F.log_softmax(logits.float(), dim=-1)
                
                # 应用 temperature 进行采样（只影响采样，不影响 log prob 计算）
                if self.temperature > 0 and self.temperature != 1.0:
                    sampling_logits = logits / self.temperature
                else:
                    sampling_logits = logits
                
                # 采样
                probs = F.softmax(sampling_logits.float(), dim=-1)
                
                # Top-k filtering
                if self.top_k > 0:
                    top_k = min(self.top_k, probs.shape[-1])
                    top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
                    # 重新归一化
                    top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                    # 从 top-k 中采样
                    sampled_indices = torch.multinomial(top_k_probs, num_samples=1)  # [batch_size, 1]
                    next_tokens = torch.gather(top_k_indices, dim=-1, index=sampled_indices).squeeze(-1)  # [batch_size]
                else:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [batch_size]
                
                # 获取采样 token 的 log probability（用原始 logits 的 log prob）
                token_logprobs = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)  # [batch_size]
                
                # 对已经结束的样本，设置 logprob 为 0
                token_logprobs = token_logprobs.masked_fill(finished, 0.0)
                
                # 更新结束状态
                finished = finished | (next_tokens == self.processing_class.eos_token_id)
                
                # 对已经结束的样本，用 pad_token 填充
                next_tokens = next_tokens.masked_fill(finished, pad_token_id)
                
                generated_tokens.append(next_tokens)
                sampled_logprobs.append(token_logprobs)
                
                # 更新 current_ids 和 current_mask
                current_ids = torch.cat([current_ids, next_tokens.unsqueeze(-1)], dim=-1)
                current_mask = torch.cat([current_mask, (~finished).long().unsqueeze(-1)], dim=-1)
                
                # 如果所有样本都结束了，提前退出
                if finished.all():
                    break
        
        # 拼接生成的 tokens 和 logprobs
        if generated_tokens:
            generated_tokens = torch.stack(generated_tokens, dim=1)  # [batch_size, num_new_tokens]
            sampled_logprobs = torch.stack(sampled_logprobs, dim=1)  # [batch_size, num_new_tokens]
            num_new_tokens = generated_tokens.shape[1]
            
            # 拼接到原始序列
            generated_ids = torch.cat([left_padded_input_ids, generated_tokens], dim=-1)
        else:
            generated_ids = left_padded_input_ids
            sampled_logprobs = torch.zeros((batch_size, 0), device=input_ids.device)
            num_new_tokens = 0
        
        # 生成对应的 attention mask
        generated_attention_mask = torch.ones_like(generated_ids, dtype=attention_mask.dtype)
        generated_attention_mask[generated_ids == pad_token_id] = 0
        
        # 恢复模型原来的训练状态
        if was_training:
            model.train()
        
        return generated_ids, generated_attention_mask, sampled_logprobs, num_new_tokens

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 在评估模式下，使用 off-policy 方式（不进行采样生成）
        # 因为评估时需要确定性的结果
        is_training = model.training
        
        if self.use_on_policy and is_training:
            # ========== On-Policy RL-style Distillation ==========
            # 类似伪代码的流程:
            # 1. 用学生模型采样轨迹，记录 sampled_logprobs
            # 2. 用教师模型计算 teacher_logprobs
            # 3. 计算 reverse_kl = sampled_logprobs - teacher_logprobs
            # 4. 使用 policy gradient loss: -advantage * new_logprobs
            
            # 1. 提取 prompt（source 部分）
            labels = inputs.get("labels", None)
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            
            # 找到 prompt 的结束位置（labels 中 IGNORE_INDEX 的部分是 prompt）
            if labels is not None:
                prompt_lengths = []
                for i in range(labels.shape[0]):
                    ignore_positions = (labels[i] == IGNORE_INDEX).nonzero(as_tuple=True)[0]
                    if len(ignore_positions) > 0:
                        prompt_len = ignore_positions[-1].item() + 1
                    else:
                        prompt_len = labels.shape[1]
                    prompt_lengths.append(prompt_len)
                prompt_lengths = torch.tensor(prompt_lengths, device=input_ids.device)
            else:
                if attention_mask is not None:
                    prompt_lengths = attention_mask.sum(dim=1)
                else:
                    prompt_lengths = torch.full((input_ids.shape[0],), input_ids.shape[1], device=input_ids.device)
            
            # 提取 prompt
            batch_size = input_ids.shape[0]
            max_prompt_len = prompt_lengths.max().item()
            prompt_ids = input_ids[:, :max_prompt_len].clone()
            if attention_mask is not None:
                prompt_attention_mask = attention_mask[:, :max_prompt_len].clone()
            else:
                prompt_attention_mask = torch.ones((batch_size, max_prompt_len), device=input_ids.device, dtype=torch.long)
            
            for i in range(batch_size):
                prompt_len = prompt_lengths[i].item()
                if prompt_len < max_prompt_len:
                    prompt_ids[i, prompt_len:] = self.processing_class.pad_token_id
                    prompt_attention_mask[i, prompt_len:] = 0
            
            # 2. 使用学生模型进行自回归采样，并记录采样时的 log probabilities
            prompt_len = prompt_ids.shape[1]
            max_length = getattr(self.processing_class, 'model_max_length', 2048)
            effective_max_new_tokens = min(self.max_new_tokens, max_length - prompt_len)
            
            if effective_max_new_tokens <= 0:
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
                p = F.softmax(shift_target_logits.float(), dim=-1)
                q_log = F.log_softmax(shift_logits.float(), dim=-1)
                loss = nn.KLDivLoss(reduction='batchmean')(q_log, p)
                return (loss, outputs) if return_outputs else loss
            
            # 采样并获取 sampled_logprobs（类似伪代码中的 trajectories.loss_fn_inputs["logprobs"]）
            generated_ids, generated_attention_mask, sampled_logprobs, num_new_tokens = self._sample_from_model(
                model, prompt_ids, prompt_attention_mask, effective_max_new_tokens
            )
            
            if num_new_tokens == 0:
                print("[Debug] No new tokens generated, returning zero loss", flush=True)
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                outputs = model(**inputs)
                return (loss, outputs) if return_outputs else loss
            
            # 3. 计算教师模型的 log probabilities（类似伪代码中的 teacher_client.compute_logprobs）
            with torch.no_grad():
                teacher_outputs = self.target_model(
                    input_ids=generated_ids, 
                    attention_mask=generated_attention_mask, 
                    use_cache=False
                )
                teacher_logits = teacher_outputs.logits.float()
                
                # 获取生成部分的 logits
                # generated_ids 结构: [left_pad prompt new_tokens]
                # logits[i] 预测 token[i+1]
                # 所以预测 new_tokens 的 logits 位置是 [-(num_new_tokens+1):-1]
                teacher_logits_gen = teacher_logits[:, -(num_new_tokens+1):-1, :]  # [batch, num_new_tokens, vocab]
                teacher_log_probs = F.log_softmax(teacher_logits_gen, dim=-1)
                
                # 获取采样 token 对应的 log prob
                generated_tokens = generated_ids[:, -num_new_tokens:]  # [batch, num_new_tokens]
                teacher_logprobs = teacher_log_probs.gather(
                    dim=-1, 
                    index=generated_tokens.unsqueeze(-1)
                ).squeeze(-1)  # [batch, num_new_tokens]
            
            # 4. 重新计算学生模型的 log prob（需要梯度）
            student_outputs = model(
                input_ids=generated_ids, 
                attention_mask=generated_attention_mask, 
                use_cache=False
            )
            student_logits = student_outputs.logits.float()
            
            # 获取生成部分的 logits
            # generated_ids 结构: [left_pad prompt new_tokens]
            # logits[i] 预测 token[i+1]
            # 所以预测 new_tokens 的 logits 位置是 [-(num_new_tokens+1):-1]
            student_logits_gen = student_logits[:, -(num_new_tokens+1):-1, :]  # [batch, num_new_tokens, vocab]
            student_log_probs = F.log_softmax(student_logits_gen, dim=-1)
            
            # 获取采样 token 对应的 log prob（带梯度）
            new_logprobs = student_log_probs.gather(
                dim=-1, 
                index=generated_tokens.unsqueeze(-1)
            ).squeeze(-1)  # [batch, num_new_tokens]
            
            # 5. 计算 KL Loss（直接可微，不需要 REINFORCE）
            # 创建 mask 来忽略 padding 部分
            token_mask = (generated_tokens != self.processing_class.pad_token_id).float()
            num_valid_tokens = token_mask.sum().clamp(min=1)
            
            if self.kl_type == "reverse":
                # Reverse KL: KL(Q||P) = E_Q[log Q - log P]
                # loss = (new_logprobs - teacher_logprobs) 的均值
                # new_logprobs 有梯度，teacher_logprobs 没有梯度
                # 
                # 这和伪代码一致：
                # reverse_kl = sampled_logprobs - teacher_logprobs
                # 但我们用 new_logprobs（重新计算的，有梯度）代替 sampled_logprobs
                reverse_kl = new_logprobs - teacher_logprobs  # [batch, num_new_tokens]
                loss = (reverse_kl * token_mask).sum() / num_valid_tokens
                
            elif self.kl_type == "forward":
                # Forward KL: KL(P||Q) = E_P[log P - log Q]
                # 由于我们从 Q 采样，用 importance sampling:
                # KL(P||Q) ≈ E_Q[(P/Q) * (log P - log Q)]
                with torch.no_grad():
                    # importance weight = P(a)/Q(a) = exp(log P - log Q)
                    importance_weights = (teacher_logprobs - sampled_logprobs).exp()
                    importance_weights = importance_weights.clamp(max=10.0)  # 防止权重过大
                
                # loss = E_Q[w * (log P - log Q)] = E_Q[w * (-log Q + log P)]
                # 只有 -log Q 部分有梯度
                forward_kl = importance_weights * (teacher_logprobs - new_logprobs)
                loss = (forward_kl * token_mask).sum() / num_valid_tokens
            else:
                raise ValueError(f"Unknown KL type: {self.kl_type}. Must be 'forward' or 'reverse'")
            
            outputs = student_outputs
            
            # 应用后处理
            if (
                    self.args.average_tokens_across_devices
                    and (self.model_accepts_loss_kwargs or self.compute_loss_func)
                    and num_items_in_batch is not None
            ):
                loss *= self.accelerator.num_processes
            
            return (loss, outputs) if return_outputs else loss
            
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
