#!/bin/bash

# Training monitor script
OUTPUT_DIR="work_dirs/3b/finetune_1e_sam2_base_plus_videoseg_eccv_merged_v2"

echo "=== Training Monitor Started at $(date) ==="

while true; do
    echo ""
    echo "=== Check at $(date) ==="

    # Check if training processes are running
    PROC_COUNT=$(ps aux | grep 'train.py' | grep -v grep | wc -l)
    echo "Training processes: $PROC_COUNT"

    if [ "$PROC_COUNT" -eq 0 ]; then
        echo "ERROR: Training processes not found!"
        exit 1
    fi

    # Check GPU utilization
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -8

    # Check latest log entries
    echo ""
    echo "Latest debug log (rank 0):"
    tail -3 "$OUTPUT_DIR/debug.0.log" 2>/dev/null || echo "No log yet"

    # Check for checkpoints
    CKPT_COUNT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | wc -l)
    echo ""
    echo "Checkpoints saved: $CKPT_COUNT"

    if [ "$CKPT_COUNT" -gt 0 ]; then
        echo "Latest checkpoint:"
        ls -lhtr "$OUTPUT_DIR" | grep checkpoint | tail -1
    fi

    # Check for trainer state
    if [ -f "$OUTPUT_DIR/trainer_state.json" ]; then
        echo ""
        echo "Training state:"
        grep -E '"global_step"|"epoch"' "$OUTPUT_DIR/trainer_state.json" | head -2
    fi

    sleep 300  # Check every 5 minutes
done
