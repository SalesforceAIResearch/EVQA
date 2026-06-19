#!/bin/bash

set -e

# Usage: ./sft.sh [3b|7b] [base|large] [resume_checkpoint_path|--from_scratch]
model_size=${1:-3b}
sam2_type=${2:-base}

# Parse resume options
from_scratch=false
resume_checkpoint=""
for arg in "$@"; do
    if [[ "$arg" == "--from_scratch" ]]; then
        from_scratch=true
    elif [[ -n "${3:-}" && "${3}" != "--from_scratch" && "$arg" == "${3}" ]]; then
        resume_checkpoint="$3"
    fi
done

# Validate model size
if [[ "$model_size" != "3b" && "$model_size" != "7b" ]]; then
    echo "Usage: $0 [3b|7b] [base|large] [resume_checkpoint_path]"
    echo "Error: Invalid model size. Must be '3b' or '7b'"
    exit 1
fi

command -v npu-smi &>/dev/null && nproc=$(npu-smi info -l | grep "NPU ID" | wc -l) || nproc=$(nvidia-smi --list-gpus | wc -l)

# Set model path based on size
if [[ "$model_size" == "3b" ]]; then
    model_name_or_path="model_zoo/UniPixel-3B"
elif [[ "$model_size" == "7b" ]]; then
    model_name_or_path="model_zoo/UniPixel-7B"
    # model_name_or_path="work_dirs/7b/stage3_1e_sam2_large"
fi

# Set SAM2 config and checkpoint based on type
if [[ "$sam2_type" == "base" ]]; then
    sam2_config='configs/sam2.1_hiera_b+'
    sam2_checkpoint='model_zoo/sam2.1/sam2.1_hiera_base_plus.pt'
    finetune_ckpt_path="work_dirs/${model_size}/finetune_1e_sam2_base_videoseg_eccv_merged_v2"
elif [[ "$sam2_type" == "large" ]]; then
    sam2_config='configs/sam2.1_hiera_l.yaml'
    sam2_checkpoint='model_zoo/sam2.1/sam2.1_hiera_large.pt'
    finetune_ckpt_path="work_dirs/${model_size}/finetune_1e_sam2_large_videoseg_eccv_merged_v2"
else
    echo "Usage: $0 [3b|7b] [base|large] [resume_checkpoint_path]"
    echo "Error: Invalid SAM2 type. Must be 'base' or 'large'"
    exit 1
fi

# Determine resume behavior
resume_arg=""
if [[ "$from_scratch" == "true" ]]; then
    echo -e "\e[1;36mTraining Mode:\e[0m From scratch (--from_scratch flag provided)"
    resume_arg=""
elif [[ -n "$resume_checkpoint" ]]; then
    # Explicit checkpoint path provided
    resume_arg="--resume_from_checkpoint $resume_checkpoint"
    echo -e "\e[1;36mResume from:\e[0m $resume_checkpoint"
else
    # No flags or path: auto-resume if checkpoint exists, otherwise train from scratch
    resume_arg="--resume_from_checkpoint True"
    echo -e "\e[1;36mResume mode:\e[0m Auto (will resume if checkpoint exists, otherwise train from scratch)"
fi

export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((nproc-1)))
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"
export NCCL_TIMEOUT=1800

echo -e "\e[1;35m╔══════════════════════════════════════════════════════════╗\e[0m"
echo -e "\e[1;35m║         UniPixel Finetuning Configuration                ║\e[0m"
echo -e "\e[1;35m╚══════════════════════════════════════════════════════════╝\e[0m"
echo -e "\e[1;32mModel Size:\e[0m       $model_size"
echo -e "\e[1;32mModel Path:\e[0m       $model_name_or_path"
echo -e "\e[1;34mSAM2 Type:\e[0m        $sam2_type"
echo -e "\e[1;34mSAM2 Config:\e[0m      $sam2_config"
echo -e "\e[1;34mSAM2 Checkpoint:\e[0m  $sam2_checkpoint"
echo -e "\e[1;33mOutput Dir:\e[0m       $finetune_ckpt_path"
echo -e "\e[1;32mDevice Count:\e[0m     $nproc ($CUDA_VISIBLE_DEVICES)"
echo -e "\e[1;35m══════════════════════════════════════════════════════════\e[0m"

torchrun --nproc_per_node 8 unipixel/train/train.py \
    --deepspeed scripts/zero0.json \
    --model_name_or_path $model_name_or_path \
    --base_model qwen2_5_vl \
    --conv_type chatml \
    --sam2_config $sam2_config \
    --sam2_checkpoint $sam2_checkpoint \
    --sam2_image_size 768 \
    --sam2_apply_postprocessing False \
    --sam2_inference_mode False \
    --sam2_hidden_tokens 2 \
    --sam2_batch_mode False \
    --sam2_enable_decoder True \
    --sam2_lr 5e-6 \
    --lora_enable True \
    --lora_type qkvo_all \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    --lora_bias none \
    --tuning_modules embedding,ref,ref_enc,msk,seg,sam2 \
    --datasets st_evidence_gen_mask:2,st_evidence_gen_qa:2,revos:5,mevis:5,lvvis:3,ref_youtube_vos:5,ref_davis17:10,ref_sav:3,groundmore:3,vicas:3,llava_instruct_665k_videogpt_plus_576k \
    --sample_frames 8 \
    --sample_type random \
    --sample_objects 5 \
    --num_threads 1 \
    --max_conv_turns 3 \
    --max_video_frames 500 \
    --max_video_len 300 \
    --max_num_words 200 \
    --max_num_tokens 40960 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --output_dir $finetune_ckpt_path \
    --save_full_model True \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 500 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --tf32 True \
    --bf16 True \
    --report_to wandb \
    $resume_arg

bash scripts/auto_eval.sh $finetune_ckpt_path

