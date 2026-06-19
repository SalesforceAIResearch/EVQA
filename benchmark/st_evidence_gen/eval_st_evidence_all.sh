#!/bin/bash

# Simple script to run evaluation for all results folders except 'ours'

python eval_st_evidence.py \
    --pred_file results/gemini/gemini_2_5_flash_st_evidence_single_1fps.json \
    --eval_masks \
    --pred_mask_dir results/gemini/gemini_2_5_flash_st_evidence_single_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/gemini/gemini_2_5_pro_st_evidence_single_1fps.json \
    --eval_masks \
    --pred_mask_dir results/gemini/gemini_2_5_pro_st_evidence_single_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/internvl/internvl3_5_4b_st_evidence_single_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/internvl/internvl3_5_4b_st_evidence_single_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/internvl/internvl3_5_8b_st_evidence_multi_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/internvl/internvl3_5_8b_st_evidence_multi_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/internvl/internvl3_5_8b_st_evidence_single_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/internvl/internvl3_5_8b_st_evidence_single_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/llava_ov/llava_onevision_1.5_8b_instruct_st_evidence_multi_turn_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/llava_ov/llava_onevision_1.5_8b_instruct_st_evidence_multi_turn_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/llava_ov/llava_onevision_1.5_8b_instruct_st_evidence_single_turn_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/llava_ov/llava_onevision_1.5_8b_instruct_st_evidence_single_turn_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/openai/o3_st_evidence_single_turn_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/openai/o3_st_evidence_single_turn_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/qwen2_5vl/qwen2.5_vl_3b_instruct_st_evidence_single_turn_1fps.json \
    --eval_masks \
    --pred_mask_dir results/qwen2_5vl/qwen2.5_vl_3b_instruct_st_evidence_single_turn_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/qwen2_5vl/qwen2.5_vl_72b_instruct_st_evidence_single_turn_1fps.json \
    --eval_masks \
    --pred_mask_dir results/qwen2_5vl/qwen2.5_vl_72b_instruct_st_evidence_single_turn_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/qwen2_5vl/qwen2.5_vl_7b_instruct_st_evidence_single_turn_1fps.json \
    --eval_masks \
    --pred_mask_dir results/qwen2_5vl/qwen2.5_vl_7b_instruct_st_evidence_single_turn_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/qwen3vl/qwen3_vl_235b_a22b_instruct_st_evidence_single_turn_1fps.json \
    --eval_masks \
    --pred_mask_dir results/qwen3vl/qwen3_vl_235b_a22b_instruct_st_evidence_single_turn_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/qwen3vl/qwen3_vl_30b_a3b_instruct_st_evidence_single_turn_1fps.json \
    --eval_masks \
    --pred_mask_dir results/qwen3vl/qwen3_vl_30b_a3b_instruct_st_evidence_single_turn_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/qwen3vl/qwen3_vl_4b_instruct_st_evidence_single_turn_1fps.json \
    --eval_masks \
    --pred_mask_dir results/qwen3vl/qwen3_vl_4b_instruct_st_evidence_single_turn_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/qwen3vl/qwen3_vl_8b_instruct_st_evidence_single_turn_1fps.json \
    --eval_masks \
    --pred_mask_dir results/qwen3vl/qwen3_vl_8b_instruct_st_evidence_single_turn_1fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/videollama3/videollama3_7b_st_evidence_multi_turn_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/videollama3/videollama3_7b_st_evidence_multi_turn_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/videollama3/videollama3_7b_st_evidence_single_turn_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/videollama3/videollama3_7b_st_evidence_single_turn_1.0fps/concat \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/sa2va/sa2va_qwen2_5_vl_7b_st_evidence_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/sa2va/sa2va_qwen2_5_vl_7b_st_evidence_1.0fps \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/unipixel/unipixel_3b_st_evidence_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/unipixel/unipixel_3b_st_evidence_1.0fps \
    --num_workers 32

python eval_st_evidence.py \
    --pred_file results/unipixel/unipixel_7b_st_evidence_1.0fps.json \
    --eval_masks \
    --pred_mask_dir results/unipixel/unipixel_7b_st_evidence_1.0fps \
    --num_workers 32

echo "All evaluations complete!"
