#!/bin/bash

set -e

command -v npu-smi &>/dev/null && nproc=$(npu-smi info -l | grep "NPU ID" | wc -l) || nproc=$(nvidia-smi --list-gpus | wc -l)

export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((nproc-1)))
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

ckpt_path=$1

# bash /fsx/home/shijie.wang/code/st-evidence/eval_baseline/ours_st_evidence_multigpu.sh $ckpt_path 8 0

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset ref_sav_eval \
        --split valid \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/ref_sav_seg \
        --vis_pred_path $ckpt_path/ref_sav_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_seg.py ref_sav $ckpt_path/ref_sav_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset revos \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/revos_seg \
        --vis_pred_path $ckpt_path/revos_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_revos.py $ckpt_path/revos_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset mevis \
        --split valid_u \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/mevis_val_u_seg \
        --vis_pred_path $ckpt_path/mevis_val_u_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_seg.py mevis $ckpt_path/mevis_val_u_seg


# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset ref_davis17 \
        --split val \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/ref_davis17_seg \
        --vis_pred_path $ckpt_path/ref_davis17_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_seg.py ref_davis17 $ckpt_path/ref_davis17_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset ref_sav_eval \
        --split valid \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/ref_sav_seg \
        --vis_pred_path $ckpt_path/ref_sav_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_seg.py ref_sav $ckpt_path/ref_sav_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_seg.py \
        --dataset groundmore \
        --split test \
        --model_path $ckpt_path \
        --seg_pred_path $ckpt_path/groundmore_seg \
        --vis_pred_path $ckpt_path/groundmore_vis \
        --chunk $CHUNKS \
        --index $IDX \
        --verbose \
        --dump 100 &
done

wait

python unipixel/eval/eval_groundmore.py $ckpt_path/groundmore_seg

# ===========================================================================

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python unipixel/eval/infer_general.py \
        --dataset mvbench \
        --split test \
        --model_path $ckpt_path \
        --pred_path $ckpt_path/mvbench \
        --chunk $CHUNKS \
        --index $IDX &
done

wait

python unipixel/eval/eval_general.py $ckpt_path/mvbench
