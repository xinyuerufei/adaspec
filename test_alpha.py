import logging
import os
import re
import math
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

sys.path.append("../..")
from sampling.speculative_sampling import my_speculative_sampling

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


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
    ######################
    #      target        #
    ######################
    target_model = AutoModelForCausalLM.from_pretrained(test_args.target_model_name_or_path, device_map="auto")
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
        test_args.target_model_name_or_path,
        model_max_length=test_args.model_max_length,
        padding_side="left",
    )

    ######################
    #      draft         #
    ######################
    draft_model = AutoModelForCausalLM.from_pretrained(test_args.output_dir, device_map="auto")
    draft_tokenizer = transformers.AutoTokenizer.from_pretrained(
        test_args.output_dir,
        model_max_length=test_args.model_max_length,
        padding_side="left",
    )

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    test_set = dataset['test']

    logging.warning("Formatting inputs...")
    question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
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

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = target_tokenizer(
                question[i * data_args.batch_size: (i + 1) * data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = target_tokenizer(
                question[i * data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch)

    target_model.eval()
    gen_kwargs = {
        "max_new_tokens": 512,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }
    ans_pred_list = []
    set_seed(42)
    accepted_count, rejected_count = 0, 0
    for step, batch in enumerate(question_data):
        with torch.no_grad():
            gen_kwargs["input_ids"] = batch["input_ids"].to('cuda')
            gen_kwargs["attention_mask"] = batch["attention_mask"].to('cuda')
            gamma = 4
            generated_tokens, acc, rej = my_speculative_sampling(
                gen_kwargs["input_ids"], draft_model, target_model,
                gen_kwargs["max_new_tokens"], target_tokenizer.eos_token_id, gamma,
                top_k=1
            )

        pred_tokens = generated_tokens[:, batch['input_len']:]
        decoded_pred = target_tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

        # Extract the numbers in sentences
        ans_pred_list += [extract_answer_number(sentence_pred) for sentence_pred in decoded_pred]
        accepted_count += acc
        rejected_count += rej

        eps = 1e-6
        alpha = accepted_count / (accepted_count + rejected_count + eps)
        print(decoded_pred, "alpha: ", alpha, "E(# generated tokens): ", (1 - alpha ** (gamma + 1)) / (1 - alpha),
              flush=True)

    print("prediction", ans_pred_list)
    print("ground truth", answer)

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"GSM8K test accuracy: {100 * accuracy:.2f}%")
    print(f"alpha: {100 * (accepted_count / (accepted_count + rejected_count)):.2f}%")


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