#!/bin/bash

# these should be the same as the training parameters
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

python test_alpha.py \
    --target_model_name_or_path "/mnt/jxblob/onpolicy-spec-ckpt/checkpoints/gsm8k-target-qwen-7b/checkpoint-5610" \
    --data_name gsm8k \
    --output_dir "/mnt/jxblob/onpolicy-spec-ckpt/checkpoints/gsm8k-target-qwen-7b-ref-qwen-0.5b-onpolicy-3epoch" \
    --batch_size 1 \
    2>&1 | tee -a "./logs/gsm8k-target-qwen-7b-ref-qwen-0.5b-3epoch/log.txt"