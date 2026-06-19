#!/usr/bin/env python3
"""
Evaluation script for ST-Evidence MCQ results

Calculates accuracy for three tasks:
- QA: Video question answering
- Temporal Evidence: Time segment selection
- Spatial Evidence: Spatial region selection

Usage:
    python eval_st_evidence_mcq.py result/gemini/gemini-2.5-flash_all_val_1fps.json
    python eval_st_evidence_mcq.py result/internvl/InternVL3_5-8B_all_val_1.0fps.json
    python eval_st_evidence_mcq.py result/videollama3/videollama3_7b_all_val_1.0fps.json --verbose
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def calculate_accuracy(results_dict, task_key='answer', gt_key='gt_answer'):
    """Calculate accuracy for a specific task."""
    correct = 0
    total = 0
    valid = 0  # Entries with valid predictions
    
    for entry_id, data in results_dict.items():
        pred = data.get(task_key)
        gt = data.get(gt_key)
        
        # Skip if ground truth is missing
        if gt is None or gt == '' or gt == 'None':
            continue
            
        total += 1
        
        # Check if prediction exists
        if pred is None or pred == '' or pred == 'None':
            continue
            
        valid += 1
        
        # Check if correct
        if pred == gt:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    coverage = (valid / total * 100) if total > 0 else 0.0
    
    return {
        'correct': correct,
        'total': total,
        'valid': valid,
        'accuracy': accuracy,
        'coverage': coverage
    }


def get_task_type_stats(results_dict):
    """Get statistics broken down by task type (if available in the data)."""
    # This would require additional metadata about question types
    # For now, we'll just return overall stats
    pass


def print_results(results_dict, verbose=False):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("ST-EVIDENCE MCQ EVALUATION RESULTS")
    print("="*80)
    
    # Calculate metrics for each task
    qa_metrics = calculate_accuracy(results_dict, 'answer', 'gt_answer')
    temporal_metrics = calculate_accuracy(results_dict, 'evidence_t', 'gt_evidence_t')
    spatial_metrics = calculate_accuracy(results_dict, 'evidence_s', 'gt_evidence_s')
    
    # Print overall statistics
    print(f"\nTotal Entries: {len(results_dict)}")
    
    # QA Task
    print("\n" + "-"*80)
    print("VIDEO QUESTION ANSWERING (QA)")
    print("-"*80)
    print(f"Total Questions:  {qa_metrics['total']}")
    print(f"Valid Predictions: {qa_metrics['valid']} ({qa_metrics['coverage']:.1f}%)")
    print(f"Correct:          {qa_metrics['correct']}")
    print(f"Accuracy:         {qa_metrics['accuracy']:.2f}%")
    
    # Temporal Evidence Task
    print("\n" + "-"*80)
    print("TEMPORAL EVIDENCE SELECTION")
    print("-"*80)
    print(f"Total Questions:   {temporal_metrics['total']}")
    print(f"Valid Predictions: {temporal_metrics['valid']} ({temporal_metrics['coverage']:.1f}%)")
    print(f"Correct:           {temporal_metrics['correct']}")
    print(f"Accuracy:          {temporal_metrics['accuracy']:.2f}%")
    
    # Spatial Evidence Task
    print("\n" + "-"*80)
    print("SPATIAL EVIDENCE SELECTION")
    print("-"*80)
    print(f"Total Questions:   {spatial_metrics['total']}")
    print(f"Valid Predictions: {spatial_metrics['valid']} ({spatial_metrics['coverage']:.1f}%)")
    print(f"Correct:           {spatial_metrics['correct']}")
    print(f"Accuracy:          {spatial_metrics['accuracy']:.2f}%")
    
    # Overall Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total_questions = qa_metrics['total'] + temporal_metrics['total'] + spatial_metrics['total']
    total_correct = qa_metrics['correct'] + temporal_metrics['correct'] + spatial_metrics['correct']
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0.0
    
    print(f"Overall Accuracy (all tasks): {overall_accuracy:.2f}%")
    print(f"  - QA:       {qa_metrics['accuracy']:.2f}%")
    print(f"  - Temporal: {temporal_metrics['accuracy']:.2f}%")
    print(f"  - Spatial:  {spatial_metrics['accuracy']:.2f}%")
    
    # Verbose output: show incorrect predictions
    if verbose:
        print("\n" + "="*80)
        print("INCORRECT PREDICTIONS (Sample)")
        print("="*80)
        
        # Show first 10 incorrect QA predictions
        qa_incorrect = []
        temporal_incorrect = []
        spatial_incorrect = []
        
        for entry_id, data in results_dict.items():
            # QA
            if data.get('answer') != data.get('gt_answer') and data.get('gt_answer') is not None:
                qa_incorrect.append((entry_id, data.get('answer'), data.get('gt_answer')))
            
            # Temporal
            if data.get('evidence_t') != data.get('gt_evidence_t') and data.get('gt_evidence_t') is not None:
                temporal_incorrect.append((entry_id, data.get('evidence_t'), data.get('gt_evidence_t')))
            
            # Spatial
            if data.get('evidence_s') != data.get('gt_evidence_s') and data.get('gt_evidence_s') is not None:
                spatial_incorrect.append((entry_id, data.get('evidence_s'), data.get('gt_evidence_s')))
        
        print(f"\nQA Incorrect ({len(qa_incorrect)} total):")
        for i, (entry_id, pred, gt) in enumerate(qa_incorrect[:10]):
            print(f"  {entry_id}: Predicted={pred}, Ground Truth={gt}")
        
        print(f"\nTemporal Incorrect ({len(temporal_incorrect)} total):")
        for i, (entry_id, pred, gt) in enumerate(temporal_incorrect[:10]):
            print(f"  {entry_id}: Predicted={pred}, Ground Truth={gt}")
        
        print(f"\nSpatial Incorrect ({len(spatial_incorrect)} total):")
        for i, (entry_id, pred, gt) in enumerate(spatial_incorrect[:10]):
            print(f"  {entry_id}: Predicted={pred}, Ground Truth={gt}")
    
    print("\n" + "="*80)
    
    return {
        'qa': qa_metrics,
        'temporal': temporal_metrics,
        'spatial': spatial_metrics,
        'overall_accuracy': overall_accuracy
    }


def save_metrics_summary(results_dict, output_file):
    """Save metrics to a JSON file."""
    qa_metrics = calculate_accuracy(results_dict, 'answer', 'gt_answer')
    temporal_metrics = calculate_accuracy(results_dict, 'evidence_t', 'gt_evidence_t')
    spatial_metrics = calculate_accuracy(results_dict, 'evidence_s', 'gt_evidence_s')
    
    total_questions = qa_metrics['total'] + temporal_metrics['total'] + spatial_metrics['total']
    total_correct = qa_metrics['correct'] + temporal_metrics['correct'] + spatial_metrics['correct']
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0.0
    
    summary = {
        'total_entries': len(results_dict),
        'qa': {
            'accuracy': qa_metrics['accuracy'],
            'correct': qa_metrics['correct'],
            'total': qa_metrics['total'],
            'valid': qa_metrics['valid'],
            'coverage': qa_metrics['coverage']
        },
        'temporal': {
            'accuracy': temporal_metrics['accuracy'],
            'correct': temporal_metrics['correct'],
            'total': temporal_metrics['total'],
            'valid': temporal_metrics['valid'],
            'coverage': temporal_metrics['coverage']
        },
        'spatial': {
            'accuracy': spatial_metrics['accuracy'],
            'correct': spatial_metrics['correct'],
            'total': spatial_metrics['total'],
            'valid': spatial_metrics['valid'],
            'coverage': spatial_metrics['coverage']
        },
        'overall_accuracy': overall_accuracy
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Metrics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ST-Evidence MCQ results")
    parser.add_argument('results_file', type=str, help='Path to results JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed incorrect predictions')
    parser.add_argument('--save-metrics', '-s', type=str, default=None, 
                        help='Save metrics summary to JSON file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"❌ Error: File not found: {args.results_file}")
        return
    
    # Load results
    print(f"📂 Loading results from: {args.results_file}")
    with open(args.results_file, 'r') as f:
        results_dict = json.load(f)
    
    print(f"✓ Loaded {len(results_dict)} entries")
    
    # Print evaluation results
    metrics = print_results(results_dict, verbose=args.verbose)
    
    # Save metrics if requested
    if args.save_metrics:
        save_metrics_summary(results_dict, args.save_metrics)
    else:
        # Auto-save to same directory with _metrics.json suffix
        results_path = Path(args.results_file)
        metrics_file = results_path.parent / (results_path.stem + '_metrics.json')
        save_metrics_summary(results_dict, str(metrics_file))


if __name__ == "__main__":
    main()

