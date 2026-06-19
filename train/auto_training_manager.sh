#!/bin/bash
# Auto training manager with data link fixing and restart on failure

EVQA_DIR="/fsx/home/shijie.wang/code/EVQA"
UNIPIXEL_DIR="/fsx/home/shijie.wang/code/UniPixel"
OUTPUT_DIR="work_dirs/3b/finetune_1e_sam2_base_plus_videoseg_eccv_merged_v2"
TRAIN_CMD="micromamba run -n unipixel bash scripts/finetune_eccv_merged_v2.sh 3b base"

cd $EVQA_DIR

# Function to check and create data links
check_data_links() {
    echo "[$(date)] Checking data links..."

    DATASETS="revos mevis lvvis ref_youtube_vos ref_davis17 ref_sav groundmore vicas sav llava_instruct videogpt_plus general"

    for dataset in $DATASETS; do
        if [ ! -e "data/$dataset" ]; then
            if [ -e "$UNIPIXEL_DIR/data/$dataset" ]; then
                echo "[$(date)] Creating link for $dataset"
                ln -sf "$UNIPIXEL_DIR/data/$dataset" "data/$dataset"
            else
                echo "[$(date)] WARNING: $dataset not found in UniPixel"
            fi
        fi
    done
}

# Function to check if training is running
is_training_running() {
    ps aux | grep 'train.py' | grep -v grep | wc -l
}

# Function to check GPU utilization
check_gpu_health() {
    # Get GPU utilization of all GPUs
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{s+=$1} END {print s/NR}')

    # If average GPU utilization < 5% for extended period, might be stuck
    echo "$GPU_UTIL"
}

# Main monitoring loop
echo "[$(date)] Auto Training Manager Started"
check_data_links

CONSECUTIVE_FAILURES=0
MAX_RETRIES=5

while true; do
    PROC_COUNT=$(is_training_running)

    if [ "$PROC_COUNT" -eq 0 ]; then
        echo "[$(date)] Training not running. Starting..."

        if [ $CONSECUTIVE_FAILURES -ge $MAX_RETRIES ]; then
            echo "[$(date)] ERROR: Too many failures ($CONSECUTIVE_FAILURES). Stopping."
            exit 1
        fi

        # Check data links before starting
        check_data_links

        # Start training in background
        nohup zsh -i -c "cd $EVQA_DIR && $TRAIN_CMD" > /tmp/training_${$}.log 2>&1 &
        TRAIN_PID=$!

        echo "[$(date)] Training started with PID: $TRAIN_PID"
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))

        sleep 300  # Wait 5 minutes before checking again
    else
        echo "[$(date)] Training running ($PROC_COUNT processes)"
        CONSECUTIVE_FAILURES=0  # Reset on successful run

        # Check GPU health
        GPU_UTIL=$(check_gpu_health)
        echo "[$(date)] Average GPU Util: $GPU_UTIL%"

        # Check for checkpoints
        if [ -f "$OUTPUT_DIR/trainer_state.json" ]; then
            STEP=$(grep '"global_step"' "$OUTPUT_DIR/trainer_state.json" | head -1 | awk '{print $2}' | tr -d ',')
            echo "[$(date)] Current step: $STEP"
        fi

        sleep 600  # Check every 10 minutes when running
    fi
done
