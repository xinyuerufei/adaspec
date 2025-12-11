#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <version> <model>"
    exit 1
fi

# Assign the arguments to variables
version="$1" # change this to your desired version name
model="$2" # this can be either "gpt2*" or "opt-*", or leave it blank, which means using the default model.

# Check if the ./logs folder exists
if [ ! -d "./logs" ]; then
    # If the folder doesn't exist, create it
    mkdir "./logs"
    echo "Created ./logs folder"
else
    echo "./logs folder already exists"
fi

# Check if the ./logs/$version folder exists
if [ ! -d "./logs/$version" ]; then
    # If the folder doesn't exist, create it
    mkdir "./logs/$version"
    echo "Created ./logs/$version folder"
else
    echo "./logs/$version folder already exists"
fi

nvidia-smi

accelerate launch --config_file accelerate_configs/zero1.yaml train.py \
    --draft_model_name_or_path $model \
    --target_model_name_or_path "/mnt/blob/onpolicy-spec-ckpt/checkpoints/gsm8k-target-qwen-7b/checkpoint-5610" \
    \
    --data_name gsm8k \
    \
    --use_on_policy True \
    --kl_type forward \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --top_k 40 \
    --top_p 0.95 \
    \
    --bf16 True \
    --output_dir "/mnt/blob/onpolicy-spec-ckpt/checkpoints/$version" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --eval_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --batch_eval_metrics True \
    --prediction_loss_only True \
    --save_only_model True \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 10 \
    --report_to tensorboard \
    --logging_dir "./logs/$version" \
    --gradient_checkpointing False
