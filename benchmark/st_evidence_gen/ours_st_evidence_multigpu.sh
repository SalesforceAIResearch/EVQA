#!/bin/bash
# Script to run ours_st_evidence.py on multiple GPUs in parallel

# Check if required arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [num_gpus] [start_gpu]"
    echo ""
    echo "Arguments:"
    echo "  model_path           Path to model checkpoint (required)"
    echo "  num_gpus             Number of GPUs to use (optional, default: 8)"
    echo "  start_gpu            Starting GPU index (optional, default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 /fsx/home/shijie.wang/code/UniPixel/work_dirs/3b/finetune_1e_sam2_base_plus"
    echo "  $0 /fsx/home/shijie.wang/code/UniPixel/work_dirs/3b/finetune_1e_sam2_base_plus 4"
    echo "  $0 /fsx/home/shijie.wang/code/UniPixel/work_dirs/3b/finetune_1e_sam2_base_plus 4 4  # Use GPUs 4-7"
    echo ""
    echo "Command will be:"
    echo "  python ours_st_evidence.py --fps 1.0 --max-frames 128 --model <model_path>"
    exit 1
fi

# Configuration
MODEL_PATH="$1"
NUM_GPUS="${2:-8}"  # Default to 8 if not specified
START_GPU="${3:-0}"  # Default to GPU 0 if not specified

# Fixed parameters
FPS=1.0
MAX_FRAMES=128

# Data paths (relative to current directory)
DATA_FILE="data/st_evidence_gen.csv"
VIDEO_DIR="data/videos_6fps"
OUTPUT_DIR="results/ours"

# Other settings
MAX_NEW_TOKENS=512
NUM_THREADS=8
SAVE_EVERY=10

# Validate model path/name
if [ -z "${MODEL_PATH}" ]; then
    echo "Error: Model path is required!"
    exit 1
fi

# Build base command (matching the user's command format)
BASE_CMD="python ours_st_evidence.py \
    --fps ${FPS} \
    --max-frames ${MAX_FRAMES} \
    --model ${MODEL_PATH} \
    --data-file ${DATA_FILE} \
    --video-dir ${VIDEO_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --num-threads ${NUM_THREADS} \
    --save-every ${SAVE_EVERY} \
    --resume \
    --chunk ${NUM_GPUS}"

echo "=========================================="
echo "ST EVIDENCE - PARALLEL INFERENCE"
echo "=========================================="
echo "Configuration:"
echo "  Model:        ${MODEL_PATH}"
echo "  GPUs:         ${NUM_GPUS} (GPU ${START_GPU} to $((START_GPU+NUM_GPUS-1)))"
echo "  FPS:          ${FPS}"
echo "  Max frames:   ${MAX_FRAMES}"
echo ""
echo "Data:"
echo "  CSV:          ${DATA_FILE}"
echo "  Videos:       ${VIDEO_DIR}"
echo "  Output:       ${OUTPUT_DIR}"
echo ""
echo "Command template:"
echo "  python ours_st_evidence.py --fps ${FPS} --max-frames ${MAX_FRAMES} --model ${MODEL_PATH} ..."
echo "=========================================="
echo ""

# Launch jobs on each GPU
echo "Launching ${NUM_GPUS} parallel jobs..."
echo ""
for i in $(seq 0 $((NUM_GPUS-1))); do
    GPU_ID=$((START_GPU + i))
    echo "  [GPU ${GPU_ID}] Chunk ${i}/${NUM_GPUS}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${BASE_CMD} --index ${i} &
    PID=$!
    echo "           PID: ${PID}"
done

echo ""
echo "=========================================="
echo "✅ All ${NUM_GPUS} jobs launched!"
echo "=========================================="
echo ""
echo "📊 Monitor progress:"
echo "  # Check how many samples processed"
echo "  ls ${OUTPUT_DIR}/*/masks/ 2>/dev/null | wc -l"
echo ""
echo "🔍 Check status:"
echo "  # List running processes"
echo "  ps aux | grep ours_st_evidence | grep -v grep"
echo ""
echo "  # Count running jobs"
echo "  ps aux | grep ours_st_evidence | grep -v grep | wc -l"
echo ""
echo "  # Watch GPU usage"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "⚠️  Stop all jobs:"
echo "  pkill -f ours_st_evidence"
echo ""
echo "=========================================="
echo "All outputs will be printed below..."
echo "=========================================="
echo ""

# Wait for all background jobs to complete
wait

echo ""
echo "=========================================="
echo "🎉 ALL GPU JOBS COMPLETED!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  ${OUTPUT_DIR}/"
echo ""
echo "Check results:"
echo "  ls -lh ${OUTPUT_DIR}/*.json"
echo ""
echo "=========================================="

