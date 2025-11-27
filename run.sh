#!/bin/bash

# Define the lists
version="gsm8k-target-qwen-7b"
model="Qwen/Qwen2.5-7B"

remove_before_slash() {
    local input_string="$1"
    local output_string="${input_string##*/}"
    echo "$output_string"
}

sh run_train.sh "$version" "$model"