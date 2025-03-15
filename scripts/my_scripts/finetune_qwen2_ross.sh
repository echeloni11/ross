#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1

RUN_NAME="qwen2-nuscenes20k-ross"

cd ~/projects/ross
echo "Running $RUN_NAME"

source /home/hanyim/miniconda3/etc/profile.d/conda.sh
conda activate ross

set -x

MASTER_ADDR=$(hostname)
MASTER_PORT=29805

torchrun --nproc-per-node=4 --nnodes 1 --node_rank 0 \
    --master_addr=$MASTER_ADDR  --master_port=$MASTER_PORT \
    \
    train.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path Qwen/Qwen2-7B-Instruct \
    --pretrain_mm_mlp_adapter ./checkpoints/ross-siglip-qwen2-7b-pt558k/mm_projector.bin \
    --pretrain_mm_inv_mlp_adapter ./checkpoints/ross-siglip-qwen2-7b-pt558k/mm_inv_projector.bin \
    --output_dir ./checkpoints/$RUN_NAME \
    --vision_tower google/siglip-so400m-patch14-384 \
    --version qwen_2 \
    --mm_pixel_decoder ./pretrained_vae \
    \
    --data_path ./train_scenes.jsonl \
    --image_folder "" \
    \
    --mm_projector_type mlp2x_gelu \
    --mm_inv_projector_type denoiser_vit3x \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 50 \
    --save_total_limit 3 \
    --save_only_model \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $RUN_NAME \
