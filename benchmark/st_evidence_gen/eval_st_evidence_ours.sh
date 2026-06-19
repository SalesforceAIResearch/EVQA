#!/bin/bash

# Script to run evaluation for all results in the 'ours' folder

python eval_st_evidence.py \
    --pred_file results/ours/3b_finetune_1e_sam2_base_plus_videoseg_st_evidence_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/ours/3b_finetune_1e_sam2_base_plus_videoseg_st_evidence_1.0fps \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/ours/7b_finetune_1e_sam2_base_plus_videoseg_st_evidence_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/ours/7b_finetune_1e_sam2_base_plus_videoseg_st_evidence_1.0fps \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/ours/3b_finetune_1e_sam2_base_plus_videoseg_eccv_merged_v2_st_evidence_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/ours/3b_finetune_1e_sam2_base_plus_videoseg_eccv_merged_v2_st_evidence_1.0fps \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/ours/7b_finetune_1e_sam2_base_plus_videoseg_eccv_merged_v2_st_evidence_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/ours/7b_finetune_1e_sam2_base_plus_videoseg_eccv_merged_v2_st_evidence_1.0fps \
    --num_workers 32

echo "All 'ours' evaluations complete!"
