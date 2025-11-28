import logging
import os
import re
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
from accelerate.utils import set_seed

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments,
)
import sys

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    logging.warning(f"vLLM is not available. Import error: {e}")
    logging.warning("Please install vLLM to use this functionality. You may need to activate the correct conda environment.")
except Exception as e:
    VLLM_AVAILABLE = False
    error_msg = str(e)
    if "undefined symbol" in error_msg or "_C.abi3.so" in error_msg:
        logging.warning(f"vLLM import failed due to compatibility issue: {error_msg}")
        logging.warning("This is usually caused by PyTorch version mismatch. Try:")
        logging.warning("  1. Reinstall vLLM: pip uninstall vllm && pip install vllm")
        logging.warning("  2. Or check PyTorch version compatibility with vLLM 0.8.1")
    else:
        logging.warning(f"vLLM import failed with error: {type(e).__name__}: {error_msg}")
        logging.warning("Please check your vLLM installation.")


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@contextmanager
def _timer(name: str, timing_dict: Dict):
    """计时器上下文管理器，用于测量代码执行时间。"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        timing_dict[name] = end_time - start_time


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})


@dataclass
class TestArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    target_model_name_or_path: Optional[str] = field(default=None)
    lora_backend: Optional[str] = field(
        default=None,
        metadata={"help": "Backend for LoRA."}
    )
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be left padded (and possibly truncated)."},
    )


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


#access_token = "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"


def evaluation(data_args, test_args):
    if not VLLM_AVAILABLE:
        logging.error("vLLM is not available. Please install vLLM to use this functionality.")
        return
    
    # 初始化 vLLM with speculative decoding
    logging.warning("Initializing vLLM with speculative decoding...")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=40,
        top_p=0.95,
        max_tokens=512,
    )
    
    # 设置合理的 max_model_len，考虑到输入长度 + 生成长度（512 tokens）
    # 对于 GSM8K 任务，2048 应该足够（输入通常 < 500 tokens，生成 512 tokens）
    max_len = min(2048, test_args.model_max_length + 512) if test_args.model_max_length > 512 else 2048
    
    llm = LLM(
        model=test_args.target_model_name_or_path,
        tensor_parallel_size=1,
        speculative_model=test_args.output_dir,
        num_speculative_tokens=4,  # 对应原来的 gamma=4
        gpu_memory_utilization=0.95,  # 增加 GPU 内存利用率以支持更大的 KV cache
        max_model_len=max_len,  # 限制最大模型长度以避免 KV cache 溢出
    )

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    test_set = dataset['test']

    logging.warning("Formatting inputs...")
    questions = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
    answer = []

    # get numerical answer
    for example in test_set['answer']:
        ans = example.split('####')[-1]
        ans = ans.replace(',', '')  # handle numbers like 2,000
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    # 速度测试相关变量
    gen = 0
    cnt = 0
    total = 0
    ans_pred_list = []
    
    logging.warning(f"Starting evaluation with {len(questions)} questions...")
    
    for e in questions:
        timing_raw = {}
        with _timer('gen', timing_raw):
            outputs = llm.generate([e], sampling_params, use_tqdm=False)
        
        gen += timing_raw['gen']
        cnt += 1
        
        # 获取生成的文本
        generated_text = outputs[0].outputs[0].text
        ans_pred_list.append(extract_answer_number(generated_text))
        
        # 统计生成的token数
        token_count = len(outputs[0].outputs[0].token_ids)
        total += token_count
        
        print("Generated:", generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
              "{} s/sentence, {} tokens/s".format(gen / cnt, total/gen), flush=True)

    print("prediction", ans_pred_list)
    print("ground truth", answer)

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"GSM8K test accuracy: {100 * accuracy:.2f}%")
    print("total time: {}".format(gen))
    print("{} s/sentence, {} tokens/s".format(gen / cnt, total/gen))


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((DataArguments, TestArguments))
    data_args, test_args = parser.parse_args_into_dataclasses()
    evaluation(data_args, test_args)
