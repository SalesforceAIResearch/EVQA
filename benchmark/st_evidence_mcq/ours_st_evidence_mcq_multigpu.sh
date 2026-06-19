#!/bin/bash
# Script to run ours_st_evidence_mcq.py on multiple GPUs in parallel

# Check if required arguments are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [num_gpus] [start_gpu] [task]"
    echo ""
    echo "Arguments:"
    echo "  model_path           Path to model checkpoint (required)"
    echo "  num_gpus             Number of GPUs to use (optional, default: 8)"
    echo "  start_gpu            Starting GPU index (optional, default: 0)"
    echo "  task                 Task type: qa, time_evidence, spatial_evidence, or all (optional, default: all)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/model_checkpoint"
    echo "  $0 /path/to/model_checkpoint 4"
    echo "  $0 /path/to/model_checkpoint 4 4  # Use GPUs 4-7"
    echo "  $0 /path/to/model_checkpoint 8 0 all  # All tasks"
    echo ""
    echo "Command will be:"
    echo "  python ours_st_evidence_mcq.py --task all --split val --fps 1.0 --max-frames 128 --model <model_path>"
    exit 1
fi

# Configuration
MODEL_PATH="$1"
NUM_GPUS="${2:-8}"  # Default to 8 if not specified
START_GPU="${3:-0}"  # Default to GPU 0 if not specified
TASK="${4:-all}"  # Default to 'all' if not specified

# Fixed parameters
FPS=1.0
MAX_FRAMES=128
SPLIT="val"

# Data paths
DATA_FILE="data/st_evidence_mcq.csv"
DISTRACTORS_FILE="data/temp_options.json"
MASK_FILE="data/mask_options.json"
VIDEO_DIR="data/NextQA-Video"

# Other settings
NUM_THREADS=8

# Validate model path/name
if [ -z "${MODEL_PATH}" ]; then
    echo "Error: Model path is required!"
    exit 1
fi

# Validate task
if [[ ! "${TASK}" =~ ^(qa|time_evidence|spatial_evidence|all)$ ]]; then
    echo "Error: Invalid task '${TASK}'. Must be one of: qa, time_evidence, spatial_evidence, all"
    exit 1
fi

# Build base command
BASE_CMD="python ours_st_evidence_mcq.py \
    --task ${TASK} \
    --split ${SPLIT} \
    --model ${MODEL_PATH} \
    --data_file ${DATA_FILE} \
    --distractors_file ${DISTRACTORS_FILE} \
    --mask_file ${MASK_FILE} \
    --video_dir ${VIDEO_DIR} \
    --fps ${FPS} \
    --max_frames ${MAX_FRAMES} \
    --num_threads ${NUM_THREADS}"

echo "=========================================="
echo "ST EVIDENCE MCQ - PARALLEL INFERENCE"
echo "=========================================="
echo "Configuration:"
echo "  Model:        ${MODEL_PATH}"
echo "  GPUs:         ${NUM_GPUS} (GPU ${START_GPU} to $((START_GPU+NUM_GPUS-1)))"
echo "  Task:         ${TASK}"
echo "  Split:        ${SPLIT}"
echo "  FPS:          ${FPS}"
echo "  Max frames:   ${MAX_FRAMES}"
echo ""
echo "Data:"
echo "  CSV:          ${DATA_FILE}"
echo "  Videos:       ${VIDEO_DIR}"
echo "  Distractors:  ${DISTRACTORS_FILE}"
echo "  Masks:        ${MASK_FILE}"
echo ""
echo "Command template:"
echo "  python ours_st_evidence_mcq.py --task ${TASK} --split ${SPLIT} --model ${MODEL_PATH} ..."
echo "=========================================="
echo ""

# Launch jobs on each GPU
echo "Launching ${NUM_GPUS} parallel jobs..."
echo ""
echo "⚠️  NOTE: All GPUs will write to the SAME output file using file locking."
echo "          This ensures no data is lost across multiple processes."
echo ""

for i in $(seq 0 $((NUM_GPUS-1))); do
    GPU_ID=$((START_GPU + i))
    echo "  [GPU ${GPU_ID}] Starting MCQ ${TASK} task"
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${BASE_CMD} &
    PID=$!
    echo "           PID: ${PID}"
done

echo ""
echo "=========================================="
echo "✅ All ${NUM_GPUS} jobs launched!"
echo "=========================================="
echo ""
echo "📊 Monitor progress:"
echo "  # Check output file (all GPUs write to same file)"
if [ "${TASK}" = "all" ]; then
    echo "  watch -n 5 'python -c \"import json; f=open(\\\"result/ours/*_all_${SPLIT}_${FPS}fps.json\\\"); d=json.load(f); qa=sum(1 for e in d.values() if e.get(\\\"answer\\\") is not None); t=sum(1 for e in d.values() if e.get(\\\"evidence_t\\\") is not None); s=sum(1 for e in d.values() if e.get(\\\"evidence_s\\\") is not None); print(f\\\"QA: {qa}, Time: {t}, Spatial: {s}\\\"); f.close()\"'"
else
    echo "  watch -n 5 'python -c \"import json; f=open(\\\"result/ours/*_${TASK}_${SPLIT}_${FPS}fps.json\\\"); d=json.load(f); print(f\\\"Processed: {len(d)} entries\\\"); f.close()\"'"
fi
echo ""
echo "🔍 Check status:"
echo "  # List running processes"
echo "  ps aux | grep ours_st_evidence_mcq | grep -v grep"
echo ""
echo "  # Count running jobs"
echo "  ps aux | grep ours_st_evidence_mcq | grep -v grep | wc -l"
echo ""
echo "  # Watch GPU usage"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "⚠️  Stop all jobs:"
echo "  pkill -f ours_st_evidence_mcq"
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
echo "  result/ours/"
echo ""
echo "Check results:"
echo "  ls -lh result/ours/*.json"
echo ""
if [ "${TASK}" = "all" ]; then
    echo "Verify all tasks completed:"
    echo "  python -c \"import json; f=open('result/ours/*_all_${SPLIT}_${FPS}fps.json'); d=json.load(f); qa=sum(1 for e in d.values() if e.get('answer') is not None); t=sum(1 for e in d.values() if e.get('evidence_t') is not None); s=sum(1 for e in d.values() if e.get('evidence_s') is not None); print(f'QA: {qa}, Time: {t}, Spatial: {s}, Total entries: {len(d)}'); f.close()\""
else
    echo "Count processed entries:"
    echo "  python -c \"import json; f=open('result/ours/*_${TASK}_${SPLIT}_${FPS}fps.json'); d=json.load(f); print(f'Processed: {len(d)} entries'); f.close()\""
fi
echo ""
echo "=========================================="

