"""
ST-Evidence Evaluation Script

This script evaluates predictions on the ST-Evidence dataset by calculating:
1. QA Accuracy: Percentage of correct answers
2. Temporal IoU metrics: mIoU, TIoU@0.3, TIoU@0.5
3. Temporal IoP metrics: mIoP, TIoP@0.3, TIoP@0.5
4. Mask Quality metrics (optional): J score (region IoU), F score (contour), J&F score

Usage:
    python eval_st_evidence.py --pred_file <path_to_predictions.json>
    python eval_st_evidence.py --pred_file <path_to_predictions.json> --eval_masks --pred_mask_dir <mask_directory>
    
Example:
    python eval_st_evidence.py --pred_file results/gemini/gemini_2_5_pro_st_evidence_single_1fps.json
    python eval_st_evidence.py --pred_file results/internvl/internvl3_5_4b.json --eval_masks --pred_mask_dir results/internvl/internvl3_5_4b/concat

The ground truth file defaults to st_evidence/eccv/st_evidence_final.csv
Ground truth masks default to st_evidence/data/mask_annos_latest_img
"""

import argparse
import json
import csv
import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from datetime import datetime

def parse_segments(segment_str):
    """
    Parse segment string from CSV to list of [start, end] pairs.
    """
    if not segment_str or segment_str.strip() == '':
        return []
    
    try:
        segments = ast.literal_eval(segment_str)
        if not isinstance(segments, list):
            return []
        
        # Convert to list of [start, end] pairs
        result = []
        for seg in segments:
            if isinstance(seg, list) and len(seg) >= 2:
                result.append([float(seg[0]), float(seg[1])])
        return result
    except (ValueError, SyntaxError, TypeError):
        return []

def load_ground_truth(csv_path):
    """
    Load ground truth from CSV file.
    Returns three dicts: gt_answers, gt_segments, and candidates
    """
    gt_answers = {}
    gt_segments = {}
    candidates = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry_id = row['entry_id']
            
            # Parse answer (e.g., "2", "Yes", etc.)
            gt_answers[entry_id] = row['answer']
            
            # Parse candidates
            candidates_str = row.get('candidates', '[]')
            try:
                candidates_list = ast.literal_eval(candidates_str)
                candidates[entry_id] = candidates_list if isinstance(candidates_list, list) else []
            except:
                candidates[entry_id] = []
            
            # Parse temporal segments
            segment_str = row.get('segment', '')
            gt_segments[entry_id] = parse_segments(segment_str)
    
    return gt_answers, gt_segments, candidates

def load_predictions(json_path):
    """
    Load predictions from JSON file.
    Returns three items: pred_answers dict, pred_segments dict, and error_stats dict
    """
    pred_answers = {}
    pred_segments = {}
    error_stats = {
        'parse_errors': 0,
        'none_responses': 0,
        'empty_segments': 0,
        'malformed_segments': 0
    }
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for entry_id, entry_data in data.items():
        parsed_response = entry_data.get('parsed_response', None)
        
        if parsed_response is None:
            pred_answers[entry_id] = None
            pred_segments[entry_id] = []
            error_stats['none_responses'] += 1
        else:
            # Get predicted answer
            pred_answers[entry_id] = parsed_response.get('answer', None)
            
            # Get predicted segments with robust parsing
            segments = parsed_response.get('segments', [])
            parsed_segs = []
            had_error = False
            
            if isinstance(segments, list) and len(segments) > 0:
                # Check if it's a flat list of numbers [start1, end1, start2, end2, ...]
                if all(isinstance(x, (int, float)) for x in segments):
                    # Pair up the numbers
                    for i in range(0, len(segments), 2):
                        if i + 1 < len(segments):
                            try:
                                parsed_segs.append([float(segments[i]), float(segments[i+1])])
                            except (ValueError, TypeError):
                                had_error = True
                                continue
                else:
                    # Process each element (nested list or dict format)
                    for s in segments:
                        try:
                            # Handle different formats
                            if isinstance(s, list) and len(s) >= 2:
                                # Standard format: [[start, end], ...]
                                parsed_segs.append([float(s[0]), float(s[1])])
                            elif isinstance(s, dict):
                                # Dict format: [{"start": x, "end": y}, ...]
                                if 'start' in s and 'end' in s:
                                    parsed_segs.append([float(s['start']), float(s['end'])])
                                elif 'start_time' in s and 'end_time' in s:
                                    parsed_segs.append([float(s['start_time']), float(s['end_time'])])
                                else:
                                    had_error = True
                            else:
                                had_error = True
                        except (ValueError, TypeError, KeyError, IndexError):
                            # Skip malformed segments
                            had_error = True
                            continue
                
                if had_error:
                    error_stats['malformed_segments'] += 1
            elif isinstance(segments, list) and len(segments) == 0:
                error_stats['empty_segments'] += 1
            else:
                error_stats['parse_errors'] += 1
            
            pred_segments[entry_id] = parsed_segs
    
    return pred_answers, pred_segments, error_stats

def calculate_accuracy(gt_answers, pred_answers, candidates_map):
    """
    Calculate answer accuracy.
    Maps answer choices (A, B, C, etc.) to actual answers using candidates_map.
    All examples in ground truth are counted, including:
    - Missing entries (not in prediction file): counted as incorrect
    - None responses: counted as incorrect
    """
    correct = 0
    total = 0
    
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    
    for entry_id, gt_answer in gt_answers.items():
        # Count ALL examples in ground truth
        total += 1
        
        # Missing entries are counted as incorrect
        if entry_id not in pred_answers:
            continue
        
        pred_answer = pred_answers[entry_id]
        
        # None responses are counted as incorrect
        if pred_answer is None:
            continue
        
        # Map predicted letter to actual answer
        pred_answer_str = str(pred_answer).strip()
        
        # If the prediction is a letter (A, B, C), map it to the actual answer
        if pred_answer_str in letter_to_idx and entry_id in candidates_map:
            idx = letter_to_idx[pred_answer_str]
            candidates = candidates_map[entry_id]
            if 0 <= idx < len(candidates):
                pred_answer_text = candidates[idx]
            else:
                pred_answer_text = pred_answer_str
        else:
            pred_answer_text = pred_answer_str
        
        # Compare with ground truth
        if str(pred_answer_text).strip() == str(gt_answer).strip():
            correct += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, correct, total

def calculate_intersection(segments1, segments2):
    """Calculates the total intersection duration between two lists of time segments."""
    total_intersection = 0
    for s1, e1 in segments1:
        for s2, e2 in segments2:
            intersection_start = max(s1, s2)
            intersection_end = min(e1, e2)
            if intersection_end > intersection_start:
                total_intersection += (intersection_end - intersection_start)
    return total_intersection

def calculate_total_duration(segments):
    """Calculates the total duration of a list of time segments."""
    if not segments:
        return 0
    return sum(max(0, end - start) for start, end in segments)

def calculate_temporal_iou(pred_segs, gt_segs):
    """Calculate temporal IoU for a single entry."""
    if not pred_segs or not gt_segs:
        return 0.0
    
    gt_duration = calculate_total_duration(gt_segs)
    pred_duration = calculate_total_duration(pred_segs)
    total_intersection = calculate_intersection(gt_segs, pred_segs)
    
    union = gt_duration + pred_duration - total_intersection
    return total_intersection / union if union > 0 else 0.0

def calculate_temporal_iop(pred_segs, gt_segs):
    """Calculate temporal IoP (Intersection over Prediction) for a single entry."""
    if not pred_segs or not gt_segs:
        return 0.0
    
    pred_duration = calculate_total_duration(pred_segs)
    total_intersection = calculate_intersection(gt_segs, pred_segs)
    
    return total_intersection / pred_duration if pred_duration > 0 else 0.0

def calculate_metrics_aggregate(gt_segments, pred_segments):
    """
    METHOD: TOTAL AGGREGATION
    Calculates metrics based on the total intersection over the total union of all segments.
    Format errors (empty/missing predictions) are counted as 0 scores.
    """
    iou_scores = []
    iop_scores = []
    
    for entry_id, gt_segs in gt_segments.items():
        # If entry is missing from predictions, count as 0
        if entry_id not in pred_segments:
            iou_scores.append(0.0)
            iop_scores.append(0.0)
            continue
        
        pred_segs = pred_segments[entry_id]
        
        # Use the new standalone functions
        iou = calculate_temporal_iou(pred_segs, gt_segs)
        iop = calculate_temporal_iop(pred_segs, gt_segs)
        
        iou_scores.append(iou)
        iop_scores.append(iop)
    
    return iou_scores, iop_scores

def load_mask(mask_path):
    """Load a mask image and convert to binary numpy array."""
    if not os.path.exists(mask_path):
        return None
    try:
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        # Binarize: any non-zero value becomes 1
        return (mask_array > 0).astype(np.uint8)
    except Exception as e:
        return None

def calculate_j_score(pred_mask, gt_mask):
    """
    Calculate J score (Jaccard index / IoU) for a single mask pair.
    If gt_mask is all zeros (no ground truth), returns 1.0 if pred is also all zeros, else 0.0.
    """
    if pred_mask is None or gt_mask is None:
        return 0.0
    
    # If ground truth is all black (no annotation)
    if gt_mask.sum() == 0:
        # Return 1.0 if prediction is also all black, 0.0 otherwise
        return 1.0 if pred_mask.sum() == 0 else 0.0
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0
    return intersection / union

def _seg2bmap(seg):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide boundaries.
    Uses XOR operations on shifted masks (UniPixel implementation).
    """
    seg = seg.astype(bool)
    seg[seg > 0] = 1
    
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    
    return b


def calculate_f_score(pred_mask, gt_mask, bound_th=0.008):
    """
    Calculate F score (contour-based F-measure) using UniPixel's implementation.
    
    Args:
        pred_mask: Binary prediction mask
        gt_mask: Binary ground truth mask
        bound_th: Distance threshold as fraction of image diagonal (default 0.008, same as UniPixel)
    
    Returns:
        F score between 0 and 1
    """
    if pred_mask is None or gt_mask is None:
        return 0.0
    
    # Early exit for empty masks (optimization)
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    if gt_sum == 0:
        return 1.0 if pred_sum == 0 else 0.0
    
    if pred_sum == 0:
        return 0.0
    
    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(pred_mask)
    gt_boundary = _seg2bmap(gt_mask)
    
    # Count boundary pixels
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)
    
    # Handle edge cases (following UniPixel's logic)
    if n_fg == 0 and n_gt > 0:
        return 0.0  # precision=1, recall=0 -> F=0
    elif n_fg > 0 and n_gt == 0:
        return 0.0  # precision=0, recall=1 -> F=0
    elif n_fg == 0 and n_gt == 0:
        return 1.0  # Both have no boundaries
    
    # Calculate distance threshold in pixels
    bound_pix = int(np.ceil(bound_th * np.linalg.norm(pred_mask.shape)))
    
    # Create disk kernel for dilation
    from skimage.morphology import disk
    kernel = disk(bound_pix).astype(np.uint8)
    
    # Dilate boundaries
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), kernel)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), kernel)
    
    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil
    
    # Compute precision and recall
    precision = np.sum(fg_match) / float(n_fg)
    recall = np.sum(gt_match) / float(n_gt)
    
    # Compute F measure
    if precision + recall == 0:
        return 0.0
    else:
        return 2 * precision * recall / (precision + recall)

def evaluate_single_entry(args):
    """
    Worker function to evaluate masks for a single entry (for multiprocessing).
    
    Args:
        args: Tuple of (entry_id, entry_data, gt_base_path, pred_mask_base_dir)
    
    Returns:
        Tuple of (entry_id, j_score, f_score, mask_count) or None if GT doesn't exist
    """
    entry_id, entry_data, gt_base_path_str, pred_mask_base_dir = args
    
    gt_base_path = Path(gt_base_path_str)
    
    # Get corresponding GT mask folder (must exist)
    gt_mask_path = gt_base_path / entry_id
    
    if not gt_mask_path.exists():
        # Skip this entry if GT mask folder not found (shouldn't happen)
        return None
    
    # Get the mask folder path for this entry
    mask_folder = entry_data.get('mask_folder', None) if entry_data else None
    
    # If mask_folder not in JSON, try to construct from pred_mask_base_dir
    if not mask_folder and pred_mask_base_dir:
        pred_mask_path = Path(pred_mask_base_dir) / entry_id / 'masks'
    elif mask_folder:
        pred_mask_path = Path(mask_folder)
    else:
        # No prediction masks - return score of 0
        return (entry_id, 0.0, 0.0, 0)
    
    # If prediction mask folder doesn't exist, return score of 0
    if not pred_mask_path.exists():
        return (entry_id, 0.0, 0.0, 0)
    
    # Get all predicted masks
    pred_masks = sorted(pred_mask_path.glob('*.png'))
    
    # Early exit if no prediction masks
    if not pred_masks:
        return (entry_id, 0.0, 0.0, 0)
    
    entry_j_scores = []
    entry_f_scores = []
    
    for pred_mask_file in pred_masks:
        # Load predicted mask
        pred_mask = load_mask(pred_mask_file)
        if pred_mask is None:
            continue
        
        # Get corresponding GT mask (same filename)
        gt_mask_file = gt_mask_path / pred_mask_file.name
        
        # Load GT mask (use all black if not found)
        if gt_mask_file.exists():
            gt_mask = load_mask(gt_mask_file)
        else:
            # Create all-black mask with same size as prediction
            gt_mask = np.zeros_like(pred_mask)
        
        if gt_mask is None:
            gt_mask = np.zeros_like(pred_mask)

        # Quick check: if both masks are empty, scores are 1.0 (skip expensive computation)
        pred_empty = pred_mask.sum() == 0
        gt_empty = gt_mask.sum() == 0

        if pred_empty and gt_empty:
            # Both empty -> perfect match
            j = 1.0
            f = 1.0
        elif gt_empty:
            # GT empty but pred not -> wrong prediction
            j = 0.0
            f = 0.0
        elif pred_empty:
            # Pred empty but GT not -> missed detection
            j = 0.0
            f = 0.0
        else:
            # Both non-empty -> calculate normally
            j = calculate_j_score(pred_mask, gt_mask)
            f = calculate_f_score(pred_mask, gt_mask)
        
        entry_j_scores.append(j)
        entry_f_scores.append(f)
    
    # Calculate mean scores for this entry
    if entry_j_scores:
        j_mean = np.mean(entry_j_scores)
        f_mean = np.mean(entry_f_scores)
        return (entry_id, j_mean, f_mean, len(entry_j_scores))
    
    return None


def evaluate_masks(pred_file, gt_mask_dir, pred_mask_base_dir=None, num_workers=None, gt_entries=None):
    """
    Evaluate mask quality using J, F, and J&F scores with multiprocessing.
    
    Args:
        pred_file: Path to prediction JSON file
        gt_mask_dir: Path to ground truth mask directory
        pred_mask_base_dir: Optional base directory for predicted masks (e.g., results/internvl/model_name/concat)
        num_workers: Number of parallel workers (default: cpu_count())
        gt_entries: Set of entry IDs from ground truth (if provided, only evaluate these)
    
    Returns:
        Dictionary with J, F, and J&F scores per entry
    """
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    if num_workers is None:
        num_workers = cpu_count()
    
    # If gt_entries provided, only evaluate those entries
    if gt_entries is not None:
        entries_to_eval = gt_entries
    else:
        entries_to_eval = predictions.keys()
    
    # Prepare arguments for parallel processing
    args_list = [
        (entry_id, predictions.get(entry_id), gt_mask_dir, pred_mask_base_dir)
        for entry_id in entries_to_eval
    ]
    
    # Process in parallel with progress bar
    print(f"Processing {len(args_list)} entries with {num_workers} workers...")
    
    j_scores = {}
    f_scores = {}
    jf_scores = {}
    mask_counts = {}
    
    with Pool(processes=num_workers) as pool:
        # Use imap with chunksize for better load balancing
        chunksize = max(1, len(args_list) // (num_workers * 4))
        results = list(tqdm(
            pool.imap(evaluate_single_entry, args_list, chunksize=chunksize),
            total=len(args_list),
            desc="Evaluating masks",
            unit="entry"
        ))
    
    # Collect results
    skipped_count = 0
    for result in results:
        if result is not None:
            entry_id, j_mean, f_mean, count = result
            j_scores[entry_id] = j_mean
            f_scores[entry_id] = f_mean
            jf_scores[entry_id] = (j_mean + f_mean) / 2
            mask_counts[entry_id] = count
        else:
            # GT mask folder doesn't exist (shouldn't happen but track it)
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} entries without GT masks (this shouldn't happen)")
    
    return {
        'j_scores': j_scores,
        'f_scores': f_scores,
        'jf_scores': jf_scores,
        'mask_counts': mask_counts
    }

def print_metrics(iou_scores, iop_scores, accuracy, error_stats, mask_results=None):
    """Helper function to calculate and print final metrics from score lists."""
    if not iou_scores:
        print("No scores to evaluate for temporal metrics.")
        return
    
    # ANSI color codes
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ORANGE = '\033[38;5;208m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Calculate final metrics
    mIoU = 100 * sum(iou_scores) / len(iou_scores) if iou_scores else 0
    TIoU_0_3 = 100 * sum(1 for score in iou_scores if score >= 0.3) / len(iou_scores) if iou_scores else 0
    TIoU_0_5 = 100 * sum(1 for score in iou_scores if score >= 0.5) / len(iou_scores) if iou_scores else 0
    
    mIoP = 100 * sum(iop_scores) / len(iop_scores) if iop_scores else 0
    TIoP_0_3 = 100 * sum(1 for score in iop_scores if score >= 0.3) / len(iop_scores) if iop_scores else 0
    TIoP_0_5 = 100 * sum(1 for score in iop_scores if score >= 0.5) / len(iop_scores) if iop_scores else 0
    
    # Calculate t-mean (average of IoU and IoP for each sample, then mean)
    t_mean_scores = [(iou + iop) / 2 for iou, iop in zip(iou_scores, iop_scores)]
    t_mean = 100 * sum(t_mean_scores) / len(t_mean_scores) if t_mean_scores else 0
    
    # Get mask metrics if available
    mean_j = 0
    mean_f = 0
    mean_jf = 0
    if mask_results is not None and mask_results['j_scores']:
        j_scores_list = list(mask_results['j_scores'].values())
        f_scores_list = list(mask_results['f_scores'].values())
        jf_scores_list = list(mask_results['jf_scores'].values())
        
        mean_j = 100 * np.mean(j_scores_list)
        mean_f = 100 * np.mean(f_scores_list)
        mean_jf = 100 * np.mean(jf_scores_list)
    
    # Calculate total format errors
    total_errors = (error_stats['none_responses'] + error_stats['empty_segments'] + 
                   error_stats['malformed_segments'] + error_stats['parse_errors'])
    
    print(f'\nNumber of samples: {len(iou_scores)}')
    print(f'Format errors: {total_errors} (none: {error_stats["none_responses"]}, '
          f'empty: {error_stats["empty_segments"]}, malformed: {error_stats["malformed_segments"]}, '
          f'parse: {error_stats["parse_errors"]})')
    
    # Print detailed temporal metrics
    print(f'\nDetailed Temporal Metrics:')
    print(f'  TIoP@0.3: {TIoP_0_3:.1f}%  |  TIoP@0.5: {TIoP_0_5:.1f}%  |  TIoU@0.3: {TIoU_0_3:.1f}%  |  TIoU@0.5: {TIoU_0_5:.1f}%')
    
    print('\n' + '-'*130)
    
    # Color the metric headers
    acc_header = f'{BOLD}{CYAN}Acc{RESET}'
    miou_header = f'{BOLD}{YELLOW}mIoU{RESET}'
    miop_header = f'{BOLD}{GREEN}mIoP{RESET}'
    tmean_header = f'{BOLD}{MAGENTA}t-mean{RESET}'
    j_header = f'{BOLD}{RED}J{RESET}'
    f_header = f'{BOLD}{BLUE}F{RESET}'
    jf_header = f'{BOLD}{ORANGE}J&F{RESET}'
    
    # Print header row with colored titles (accounting for ANSI codes in spacing)
    print(f'{"Metric":<12} {acc_header:<18} {miou_header:<20} {miop_header:<20} {tmean_header:<22} {j_header:<18} {f_header:<18} {jf_header:<18}')
    print('-'*130)
    
    # Format colored values (accounting for ANSI codes in spacing)
    acc_val = f'{BOLD}{CYAN}{accuracy:.1f}{RESET}'
    miou_val = f'{BOLD}{YELLOW}{mIoU:.1f}{RESET}'
    miop_val = f'{BOLD}{GREEN}{mIoP:.1f}{RESET}'
    tmean_val = f'{BOLD}{MAGENTA}{t_mean:.1f}{RESET}'
    j_val = f'{BOLD}{RED}{mean_j:.1f}{RESET}' if mask_results else '-'
    f_val = f'{BOLD}{BLUE}{mean_f:.1f}{RESET}' if mask_results else '-'
    jf_val = f'{BOLD}{ORANGE}{mean_jf:.1f}{RESET}' if mask_results else '-'
    
    print(f'{"Score (%)":<12} {acc_val:<18} {miou_val:<20} {miop_val:<20} {tmean_val:<22} {j_val:<18} {f_val:<18} {jf_val:<18}')
    print('='*130)

def main():
    parser = argparse.ArgumentParser(description='Evaluate ST-Evidence predictions')
    parser.add_argument('--gt_file', type=str, 
                        default='data/st_evidence_gen.csv',
                        help='Path to ground truth CSV file')
    parser.add_argument('--pred_file', type=str, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--eval_masks', action='store_true',
                        help='Evaluate mask quality (J, F, J&F scores)')
    parser.add_argument('--gt_mask_dir', type=str,
                        default='data/mask_annos_latest_img',
                        help='Path to ground truth masks directory')
    parser.add_argument('--pred_mask_dir', type=str, default=None,
                        help='Base directory for predicted masks (e.g., results/internvl/model_name/concat). If not specified, uses mask_folder from JSON.')
    parser.add_argument('--mask_mode', type=str, default='concat', choices=['concat', 'seperate'],
                        help='Mode to use for predicted masks (concat: concatenate all masks, seperate: use seperate mask)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers for mask evaluation (default: all CPU cores)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file to save evaluation results (default: {pred_file}_eval.json)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Loading ground truth from: {args.gt_file}")
    print(f"Loading predictions from: {args.pred_file}")
    print(f"{'='*80}\n")
    
    # Load data
    gt_answers, gt_segments, candidates = load_ground_truth(args.gt_file)
    pred_answers, pred_segments, error_stats = load_predictions(args.pred_file)
    
    print(f"Loaded {len(gt_answers)} ground truth entries")
    print(f"Loaded {len(pred_answers)} prediction entries")
    
    # Find common entries
    common_entries = set(gt_answers.keys()) & set(pred_answers.keys())
    print(f"Found {len(common_entries)} common entries for evaluation")
    
    # Calculate accuracy
    accuracy, correct, total = calculate_accuracy(gt_answers, pred_answers, candidates)
    print(f"\nQA Accuracy: {correct}/{total} = {accuracy:.1f}%")
    
    # Calculate temporal metrics
    print("\nCalculating temporal IoU metrics (format errors counted as 0)...")
    iou_scores, iop_scores = calculate_metrics_aggregate(gt_segments, pred_segments)
    
    # Build per-sample scores for correlation analysis
    per_sample_scores = {}
    letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    
    for entry_id in common_entries:
        pred_segs = pred_segments.get(entry_id, [])
        gt_segs = gt_segments.get(entry_id, [])
        
        iou = calculate_temporal_iou(pred_segs, gt_segs)
        iop = calculate_temporal_iop(pred_segs, gt_segs)
        
        # Calculate QA correctness using EXACT same logic as calculate_accuracy
        qa_correct = False
        if entry_id in pred_answers and pred_answers[entry_id] is not None:
            pred_answer = pred_answers[entry_id]
            gt_answer = gt_answers[entry_id]
            pred_answer_str = str(pred_answer).strip()
            
            # Map letter to actual answer if applicable (same logic as calculate_accuracy)
            if pred_answer_str in letter_to_idx and entry_id in candidates:
                idx = letter_to_idx[pred_answer_str]
                candidates_list = candidates[entry_id]
                if 0 <= idx < len(candidates_list):
                    pred_answer_text = str(candidates_list[idx]).strip()
                else:
                    pred_answer_text = pred_answer_str
            else:
                pred_answer_text = pred_answer_str
            
            # Compare answers (case-sensitive, matching calculate_accuracy)
            qa_correct = (pred_answer_text == str(gt_answer).strip())
        
        per_sample_scores[entry_id] = {
            'qa_correct': qa_correct,
            'temporal_iou': iou,
            'temporal_iop': iop,
            'temporal_avg': (iou + iop) / 2.0
        }
    
    # Evaluate masks if requested
    mask_results = None
    if args.eval_masks:
        print("\nEvaluating mask quality...")
        if args.pred_mask_dir:
            print(f"Using prediction mask directory: {args.pred_mask_dir}")
        else:
            args.pred_mask_dir = args.pred_file.replace('.json', '')
            if args.mask_mode == 'seperate':
                args.pred_mask_dir = args.pred_mask_dir + '/seperate'
            elif args.mask_mode == 'concat':
                args.pred_mask_dir = args.pred_mask_dir + '/concat'
            else:
                print(f"Invalid mask mode: {args.mask_mode}")
                return
            print(f"Using prediction mask directory: {args.pred_mask_dir}")
        try:
            # Pass GT entries to ensure we only evaluate GT examples
            gt_entry_ids = set(gt_answers.keys())
            mask_results = evaluate_masks(args.pred_file, args.gt_mask_dir, args.pred_mask_dir, args.num_workers, gt_entry_ids)
            
            # Count how many had predictions vs missing
            total_evaluated = len(mask_results['j_scores'])
            num_with_masks = sum(1 for count in mask_results['mask_counts'].values() if count > 0)
            num_missing = total_evaluated - num_with_masks
            print(f"Evaluated {total_evaluated} entries: {num_with_masks} with masks, {num_missing} missing (scored as 0)")
        except Exception as e:
            print(f"Error evaluating masks: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results
    print_metrics(iou_scores, iop_scores, accuracy, error_stats, mask_results)
    
    # Save results to JSON file
    output_path = args.output_file
    if output_path is None:
        # Default: add _eval.json to prediction file
        pred_path = Path(args.pred_file)
        output_path = pred_path.parent / f"{pred_path.stem}_eval.json"
    
    # Calculate all metrics
    mIoU = 100 * sum(iou_scores) / len(iou_scores) if iou_scores else 0
    TIoU_0_3 = 100 * sum(1 for score in iou_scores if score >= 0.3) / len(iou_scores) if iou_scores else 0
    TIoU_0_5 = 100 * sum(1 for score in iou_scores if score >= 0.5) / len(iou_scores) if iou_scores else 0
    
    mIoP = 100 * sum(iop_scores) / len(iop_scores) if iop_scores else 0
    TIoP_0_3 = 100 * sum(1 for score in iop_scores if score >= 0.3) / len(iop_scores) if iop_scores else 0
    TIoP_0_5 = 100 * sum(1 for score in iop_scores if score >= 0.5) / len(iop_scores) if iop_scores else 0
    
    t_mean_scores = [(iou + iop) / 2 for iou, iop in zip(iou_scores, iop_scores)]
    t_mean = 100 * sum(t_mean_scores) / len(t_mean_scores) if t_mean_scores else 0
    
    results = {
        'metadata': {
            'gt_file': args.gt_file,
            'pred_file': args.pred_file,
            'num_samples': len(iou_scores),
            'timestamp': datetime.now().isoformat()
        },
        'qa_metrics': {
            'accuracy': round(accuracy, 2),
            'correct': correct,
            'total': total
        },
        'temporal_metrics': {
            'mIoU': round(mIoU, 2),
            'mIoP': round(mIoP, 2),
            't_mean': round(t_mean, 2),
            'TIoU@0.3': round(TIoU_0_3, 2),
            'TIoU@0.5': round(TIoU_0_5, 2),
            'TIoP@0.3': round(TIoP_0_3, 2),
            'TIoP@0.5': round(TIoP_0_5, 2)
        },
        'format_errors': {
            'total': error_stats['none_responses'] + error_stats['empty_segments'] + 
                    error_stats['malformed_segments'] + error_stats['parse_errors'],
            'none_responses': error_stats['none_responses'],
            'empty_segments': error_stats['empty_segments'],
            'malformed_segments': error_stats['malformed_segments'],
            'parse_errors': error_stats['parse_errors']
        }
    }
    
    # Add mask metrics if available
    if mask_results is not None and mask_results['j_scores']:
        j_scores_list = list(mask_results['j_scores'].values())
        f_scores_list = list(mask_results['f_scores'].values())
        jf_scores_list = list(mask_results['jf_scores'].values())
        
        results['mask_metrics'] = {
            'J': round(100 * np.mean(j_scores_list), 2),
            'F': round(100 * np.mean(f_scores_list), 2),
            'J&F': round(100 * np.mean(jf_scores_list), 2),
            'num_entries': len(j_scores_list),
            'total_masks': sum(mask_results['mask_counts'].values())
        }
        
        # Add per-sample mask scores to per_sample_scores
        for entry_id in mask_results['j_scores']:
            if entry_id in per_sample_scores:
                per_sample_scores[entry_id]['spatial_j'] = mask_results['j_scores'][entry_id]
                per_sample_scores[entry_id]['spatial_f'] = mask_results['f_scores'][entry_id]
                per_sample_scores[entry_id]['spatial_jf'] = mask_results['jf_scores'][entry_id]
    
    # Add per-sample scores to results for correlation analysis
    results['per_sample_scores'] = per_sample_scores
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")

if __name__ == '__main__':
    main()

