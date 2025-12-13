"""
使用 HuggingFace TRL 的 MiniLLMTrainer 进行知识蒸馏训练
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
import torch
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@dataclass
class ScriptArguments:
    """脚本参数（不包含与 MiniLLMConfig 重复的字段）"""
    draft_model_name_or_path: str = field(
        metadata={"help": "学生模型路径"}
    )
    target_model_name_or_path: str = field(
        metadata={"help": "教师模型路径"}
    )
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "数据集名称"}
    )
    # 注意：temperature, kd_temperature, max_new_tokens 等已在 MiniLLMConfig 中定义


def prepare_dataset(data_name: str, tokenizer):
    """准备数据集"""
    dataset = load_dataset(data_name, "main")
    
    def format_prompt(example):
        question = example["question"]
        prompt = f"{question}{QUESTION_PROMPT}"
        return {"prompt": prompt}
    
    train_dataset = dataset["train"].map(format_prompt)
    eval_dataset = dataset["test"].map(format_prompt)
    
    return train_dataset, eval_dataset


def main():
    # 解析参数
    parser = HfArgumentParser((ScriptArguments, MiniLLMConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    # 加载 tokenizer (使用教师模型的 tokenizer 以确保词表一致)
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.target_model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # MiniLLM 要求左填充
    
    vocab_size = len(tokenizer)
    logger.info(f"Tokenizer vocab size: {vocab_size}")
    
    # 手动加载学生模型并调整词表大小
    logger.info(f"加载学生模型: {script_args.draft_model_name_or_path}")
    student_model = AutoModelForCausalLM.from_pretrained(
        script_args.draft_model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    # 检查并调整学生模型词表大小
    if student_model.config.vocab_size != vocab_size:
        logger.info(f"调整学生模型词表大小: {student_model.config.vocab_size} -> {vocab_size}")
        student_model.resize_token_embeddings(vocab_size)
    
    # 手动加载教师模型
    logger.info(f"加载教师模型: {script_args.target_model_name_or_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        script_args.target_model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        trust_remote_code=True,
    )
    
    # 检查并调整教师模型词表大小（如果需要）
    if teacher_model.config.vocab_size != vocab_size:
        logger.info(f"调整教师模型词表大小: {teacher_model.config.vocab_size} -> {vocab_size}")
        teacher_model.resize_token_embeddings(vocab_size)
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset(script_args.data_name, tokenizer)
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"评估集大小: {len(eval_dataset)}")
    
    # 创建 trainer (传入模型对象而非路径)
    trainer = MiniLLMTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    logger.info(f"模型已保存到: {training_args.output_dir}")


if __name__ == "__main__":
    main()

