#! /bin/bash

# activate Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

deepspeed --num_gpus=8  \
    fastchat/train/train_mem.py \
    --model_name_or_path path/to/FastChat/7B-vicuna \
    --data_path path/to/FastChat/playground/data/alpaca-data-conversation.json \
    --fp16 True \
    --output_dir ./checkpoint \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed path/to/FastChat/deepspeed_conf_7B.json > ./log/train_7B.log 2>&1 &