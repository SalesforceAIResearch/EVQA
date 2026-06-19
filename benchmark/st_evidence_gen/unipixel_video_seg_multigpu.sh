#!/bin/bash
# Script to run unipixel_video_seg.py on multiple GPUs in parallel

# Check if JSON file argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <ref_exp_json_file> [num_gpus] [mode] [start_gpu]"
    echo ""
    echo "Arguments:"
    echo "  ref_exp_json_file    Path to ref-exp JSON file (required)"
    echo "  num_gpus             Number of GPUs to use (optional, default: 8)"
    echo "  mode                 Processing mode: seperate, concat, or both (optional, default: seperate)"
    echo "  start_gpu            Starting GPU index (optional, default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 results/qwen3vl/qwen3_vl_235b_a22b_instruct_st_evidence_ref_exp_1fps.json"
    echo "  $0 results/qwen3vl/qwen3_vl_235b_a22b_instruct_st_evidence_ref_exp_1fps.json 4"
    echo "  $0 results/qwen3vl/qwen3_vl_235b_a22b_instruct_st_evidence_ref_exp_1fps.json 8 seperate"
    echo "  $0 results/qwen3vl/qwen3_vl_235b_a22b_instruct_st_evidence_ref_exp_1fps.json 4 seperate 4  # Use GPUs 4-7"
    echo "  $0 results/qwen3vl/qwen3_vl_235b_a22b_instruct_st_evidence_ref_exp_1fps.json 8 both"
    exit 1
fi

# Configuration
REF_EXP_FILE="$1"
NUM_GPUS="${2:-8}"  # Default to 8 if not specified
MODE="${3:-seperate}"  # Default to seperate if not specified
START_GPU="${4:-0}"  # Default to GPU 0 if not specified
BATCH_SIZE=5
SKIP_EXISTING="--skip_existing"  # Skip already processed videos (comment out to reprocess all)

# Sampling strategy (only one should be uncommented)
TARGET_FPS=1.0  # Sample to target FPS (default)
# FRAME_NUM=16  # Sample exact number of frames
# EVERY_N_FRAMES=6  # Sample every N frames

# Validate mode
if [[ ! "$MODE" =~ ^(seperate|concat|both)$ ]]; then
    echo "Error: Invalid mode '$MODE'. Must be one of: seperate, concat, both"
    exit 1
fi

# Validate JSON file exists
if [ ! -f "${REF_EXP_FILE}" ]; then
    echo "Error: File '${REF_EXP_FILE}' not found!"
    exit 1
fi

# Build base command
BASE_CMD="python unipixel_video_seg.py ${REF_EXP_FILE} \
    --save_masks \
    --skip_viz \
    --mode ${MODE} \
    --batch_size ${BATCH_SIZE} \
    --chunk ${NUM_GPUS} \
    ${SKIP_EXISTING}"

# Add sampling strategy (only one will be active)
if [ ! -z "${TARGET_FPS}" ]; then
    BASE_CMD="${BASE_CMD} --target_fps ${TARGET_FPS}"
elif [ ! -z "${FRAME_NUM}" ]; then
    BASE_CMD="${BASE_CMD} --frame_num ${FRAME_NUM}"
elif [ ! -z "${EVERY_N_FRAMES}" ]; then
    BASE_CMD="${BASE_CMD} --every_n_frames ${EVERY_N_FRAMES}"
fi

echo "=========================================="
echo "Running on ${NUM_GPUS} GPUs (GPU ${START_GPU} to $((START_GPU+NUM_GPUS-1)))"
echo "Input file: ${REF_EXP_FILE}"
echo "Mode: ${MODE}"
echo "Batch size: ${BATCH_SIZE}"
if [ ! -z "${TARGET_FPS}" ]; then
    echo "Sampling: target_fps=${TARGET_FPS}"
elif [ ! -z "${FRAME_NUM}" ]; then
    echo "Sampling: frame_num=${FRAME_NUM}"
elif [ ! -z "${EVERY_N_FRAMES}" ]; then
    echo "Sampling: every_n_frames=${EVERY_N_FRAMES}"
else
    echo "Sampling: default (target_fps=1.0)"
fi
echo "=========================================="

# Launch jobs on each GPU
for i in $(seq 0 $((NUM_GPUS-1))); do
    GPU_ID=$((START_GPU + i))
    echo "Launching job on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${BASE_CMD} --index ${i} &
done

echo ""
echo "All jobs launched!"
echo ""
echo "Check running processes:"
echo "  ps aux | grep unipixel_video_seg"
echo ""
echo "To kill all jobs:"
echo "  pkill -f unipixel_video_seg"
echo ""

# Wait for all background jobs to complete
wait

echo "=========================================="
echo "All GPU jobs completed!"
echo "=========================================="

