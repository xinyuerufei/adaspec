#!/bin/bash
set -e

# 激活 conda 环境
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate adaspec

# 定义参数
version="gsm8k-minillm-trl-qwen-0.5b"
student_model="Qwen/Qwen2.5-0.5B"
teacher_model="/mnt/blob/onpolicy-spec-ckpt/checkpoints/gsm8k-target-qwen-7b/checkpoint-5610"

# 创建日志目录
mkdir -p "./logs/$version"

nvidia-smi

# 使用 accelerate 启动训练
# 注意: GPU 3 有问题，只使用 GPU 0,1,2
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
    --config_file accelerate_configs/zero1_3gpu.yaml \
    train_minillm_trl.py \
    --draft_model_name_or_path "$student_model" \
    --target_model_name_or_path "$teacher_model" \
    --data_name gsm8k \
    \
    --bf16 True \
    --output_dir "/mnt/blob/onpolicy-spec-ckpt/checkpoints/$version" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --report_to tensorboard \
    --logging_dir "./logs/$version" \
    --max_completion_length 512 \
    --temperature 0.2 \
    --kd_temperature 1.0 \
    --num_generations 1 \
    --single_step_decomposition True \
    --rkl_advantage True \
    --disable_dropout True

