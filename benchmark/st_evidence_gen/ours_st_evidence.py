#!/usr/bin/env python3
# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

"""
Run ST Evidence Task using UniPixel model.
This script processes videos to answer questions and provide spatio-temporal evidence.

Frames are uniformly sampled across the entire video (matching unipixel_st_evidence.py).

Usage:
    python run_st_evidence.py \
        --model PolyU-ChenLab/UniPixel-3B \
        --data-file st_evidence/eccv/st_evidence_final.csv \
        --video-dir st_evidence/data/videos_6fps \
        --fps 1.0 \
        --max-frames 128
    
    # Results will be saved to results/ours/ by default
"""

import argparse
import ast
import fcntl
import glob
import json
import os
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from decord import VideoReader
import decord
from PIL import Image
from tqdm import tqdm

from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_image, load_frames_with_stride
from unipixel.utils.transforms import get_sam2_transform


def safe_save_json(result_dict, save_file, max_retries=5):
    """
    Safely save JSON file with file locking and reload to prevent race conditions.
    
    This ensures multiple GPUs can write to the same file without data loss:
    1. Acquire exclusive lock on the file
    2. Reload existing results (from other GPUs)
    3. Merge with current results
    4. Write back and release lock
    """
    for attempt in range(max_retries):
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            
            # Open file in read-write mode, create if doesn't exist
            mode = 'r+' if os.path.exists(save_file) else 'w+'
            with open(save_file, mode) as f:
                # Acquire exclusive lock (blocks until available)
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    # Reload existing results from file (may have updates from other GPUs)
                    if mode == 'r+' and os.path.getsize(save_file) > 0:
                        f.seek(0)
                        try:
                            existing_results = json.load(f)
                            # Merge: existing results from other GPUs + our new results
                            existing_results.update(result_dict)
                            result_dict = existing_results
                        except json.JSONDecodeError:
                            # File corrupted, use our results
                            pass
                    
                    # Write merged results
                    f.seek(0)
                    f.truncate()
                    json.dump(result_dict, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                    
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Success!
            return result_dict  # Return merged results
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait a bit and retry
                time.sleep(0.1 * (attempt + 1))
                continue
            else:
                # Final attempt failed
                print(f"⚠️  Warning: Failed to save JSON after {max_retries} attempts: {e}")
                raise
    
    return result_dict


def load_video_with_llm_only(path, sample_frames=-1, sample_type='uniform', sample_for_llm_only=False, 
                             fps=None, max_frames=None):
    """
    Load video frames with optional sample_for_llm_only behavior and fps/max_frames controls.
    
    Two mutually exclusive modes:
    1. FPS Mode (fps is specified): Load frames at specific fps, optionally limit with max_frames
       - sample_frames is ignored in this mode
       - All frames at target fps are loaded
    
    2. Sample Mode (fps is None): Sample specific number of frames uniformly
       - sample_frames determines how many frames to sample
       - Original video fps is not considered
    
    When sample_for_llm_only=True (applies to both modes):
    - Loads ALL frames from the mode
    - Returns sampled frame indices for LLM processing
    - This allows SAM2 to process all frames while LLM sees only sampled frames
    
    Args:
        path: Path to video file or frame directory
        sample_frames: Number of frames to sample (only used when fps=None)
        sample_type: 'uniform' or 'random' sampling
        sample_for_llm_only: If True, load all frames but sample for LLM; if False, load only sampled frames
        fps: Target FPS for loading frames (e.g., 6 to load at 6fps). If specified, sample_frames is ignored
        max_frames: Maximum number of frames to load (after fps sampling). If None, load all
        
    Returns:
        frames: All frames (if sample_for_llm_only=True) or sampled frames (if False) [T, C, H, W]
        sampled_paths: List of PIL Images at sampled indices (for LLM)
        inds: Sampled frame indices
    """
    assert sample_type in ('uniform', 'random')
    
    frame_mode = os.path.isdir(path)
    
    if frame_mode:
        # Directory of frames
        paths = []
        for ext in ('jpg', 'png'):
            paths.extend(glob.glob(os.path.join(path, f'*.{ext}')))
        paths.sort(key=lambda p: int(re.sub(r'^\D*', '', os.path.splitext(os.path.basename(p))[0])))
        total_frames = len(paths)
        video_fps = None  # Unknown for frame directories
    else:
        # Video file
        if VideoReader is None:
            raise ImportError("decord is required for video loading. Install with: pip install decord")
        decord.bridge.set_bridge('torch')
        vr = VideoReader(path, num_threads=1)
        total_frames = len(vr)
        video_fps = vr.get_avg_fps()
    
    # MODE 1: FPS Mode (fps is specified)
    if fps is not None and video_fps is not None:
        # Load frames at specific FPS
        every_n_frames = max(1, int(round(video_fps / fps)))
        keep_indices = list(range(0, total_frames, every_n_frames))
        
        # Apply max_frames limit if specified
        if max_frames is not None and max_frames > 0:
            keep_indices = keep_indices[:max_frames]
        
        # In FPS mode, keep_indices ARE the frames to load (no further sampling)
        load_indices = keep_indices
        sampled_indices = keep_indices  # For LLM, same as loaded frames
    
    # MODE 2: Sample Mode (fps is None)
    else:
        # Sample N frames uniformly from the video
        if sample_frames > 0 and total_frames > sample_frames:
            if sample_type == 'uniform':
                sampled_indices = np.arange(0, total_frames, (total_frames - 1) / (sample_frames - 1))[:sample_frames].round().astype(int).tolist()
            else:
                seps = np.arange(0, total_frames, (total_frames - 1) / sample_frames)[:sample_frames + 1].round().astype(int).tolist()
                sampled_indices = [random.choice(range(sep, max(sep + 1, seps[i + 1]))) for i, sep in enumerate(seps[:-1])]
            assert len(sampled_indices) == sample_frames
        else:
            sampled_indices = list(range(total_frames))
        
        # In sample mode, we either load all frames or just sampled frames
        if sample_for_llm_only:
            load_indices = list(range(total_frames))  # Load all frames
        else:
            load_indices = sampled_indices  # Load only sampled frames
    
    # Load frames
    if frame_mode:
        frames = torch.cat([load_image(paths[i]) for i in load_indices])
        sampled_paths = [paths[i] for i in sampled_indices]
    else:
        # Video file
        frames = vr.get_batch(load_indices)
        # Create PIL Images for sampled frames (for LLM)
        sampled_frames = vr.get_batch(sampled_indices)
        sampled_paths = [Image.fromarray(t.numpy()) for t in sampled_frames]
    
    return frames, sampled_paths, sampled_indices


def load_data(csv_path, video_dir):
    """
    Load QA data from CSV file.
    
    Expected CSV columns:
    - entry_id: Unique identifier
    - video_id: Video identifier
    - video_path: Relative path to video
    - question: Question text
    - candidates: List of answer options
    """
    df = pd.read_csv(csv_path)
    data = []
    
    print(f"📂 Loading data from: {csv_path}")
    print(f"🎬 Video directory: {video_dir}")
    
    for idx, row in df.iterrows():
        entry_id = row['entry_id']
        video_id = row['video_id']
        video_path = os.path.join(video_dir, row['video_path'])
        question = row['question']
        
        # Parse candidates
        try:
            candidates = ast.literal_eval(row['candidates']) if isinstance(row['candidates'], str) else row['candidates']
        except Exception as e:
            print(f"⚠️  Warning: Failed to parse candidates for {entry_id}: {e}")
            continue
        
        # Check if video exists
        if not os.path.isfile(video_path):
            print(f"⚠️  Warning: Video not found: {video_path}")
            continue
        
        data.append({
            'entry_id': entry_id,
            'video_id': video_id,
            'video_path': video_path,
            'question': question,
            'options': candidates
        })
    
    print(f"✅ Loaded {len(data)} valid QA pairs\n")
    return data


def format_question_with_options(question, options):
    """Format question with multiple choice options."""
    option_letters = ['A', 'B', 'C', 'D', 'E']
    question_text = question + '\nOptions:\n'
    for i, option in enumerate(options):
        question_text += f"({option_letters[i]}) {option}\n"
    return question_text.strip()


def create_st_evidence_prompt(question, options):
    """
    Create single-turn prompt for ST Evidence task.
    This matches the format used in UniPixel training.
    """
    question_with_options = format_question_with_options(question, options)
    prompt = f"{question_with_options} Answer the question and provide evidence in the form of temporal ([[start1, end1], [start2, end2], ...]) and spatial evidence (masks)."
    return prompt


def parse_temporal_segments(text):
    """Parse temporal segments from model response."""
    # Pattern to match [[start1, end1], [start2, end2], ...]
    pattern = r'\[\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\](?:\s*,\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\])*\s*\]'
    match = re.search(pattern, text)
    if match:
        # Extract all [start, end] pairs
        segs = re.findall(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
        return [[float(s), float(e)] for s, e in segs]
    return []


def extract_answer_letter(text):
    """Extract answer letter (A, B, C, D, E) from response."""
    # Look for patterns like "Answer: A" or just "A." at the beginning
    match = re.search(r'(?:Answer\s*:\s*)?([A-E])\.?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def save_masks_to_disk(masks, output_dir, entry_id):
    """
    Save predicted masks to disk.
    
    Structure: output_dir / entry_id / masks / XXXXX.png
    """
    mask_dir = os.path.join(output_dir, entry_id, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    
    # masks shape: [num_frames, H, W]
    num_frames = masks.shape[0]
    for i in range(num_frames):
        mask = masks[i]
        # Convert to uint8 (0-255)
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        # Save as PNG
        frame_filename = f"{str(i).zfill(5)}.png"
        mask_path = os.path.join(mask_dir, frame_filename)
        cv2.imwrite(mask_path, mask_uint8)
    
    return mask_dir


def process_single_item(item, model, processor, sam2_transform, device, args, mask_output_dir):
    """
    Process a single video item.
    
    Returns dict with:
    - answer: Predicted answer letter
    - segments: Temporal segments [[start, end], ...]
    - mask_path: Path to saved masks
    """
    result = {'answer': None, 'segments': None, 'mask_path': None}
    
    try:
        # Load video - ALL frames for mask generation, sampled frames for LLM
        # (matching unipixel_st_evidence.py approach)
        decord.bridge.set_bridge('torch')
        vr = VideoReader(item['video_path'], num_threads=args.num_threads)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        # Calculate sample_frames (matching unipixel_st_evidence.py)
        if args.fps is not None:
            video_duration = total_frames / original_fps
            sample_frames = int(video_duration * args.fps)
            sample_frames = max(1, min(sample_frames, min(args.max_frames, total_frames)))
        else:
            sample_frames = min(args.max_frames, total_frames)
        
        # Load frames: uniformly spread across entire video
        frames, images, inds = load_frames_with_stride(
            item['video_path'],
            every_n_frames=1,  # Load all frames first, then uniformly sample
            sample_frames=sample_frames,
            sample_type='uniform',
            sample_for_llm_only=True,  # All frames for SAM2, sampled for LLM
            num_threads=args.num_threads
        )
        
        num_frames = len(images)
        
        # Create prompt
        prompt = create_st_evidence_prompt(item['question'], item['options'])
        
        # Prepare messages
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'video',
                'video': images,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * num_frames
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]
        
        # Prepare input
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images_proc, videos_proc, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = processor(text=[text], images=images_proc, videos=videos_proc, return_tensors='pt', **kwargs)
        
        # Add SAM2 frames
        data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
        data['frame_size'] = [frames.shape[1:3]]
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **data.to(device),
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_k=None,
                top_p=None,
                repetition_penalty=None,
                max_new_tokens=args.max_new_tokens
            )
        
        # Decode response
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        
        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        
        # Parse answer letter
        answer = extract_answer_letter(response)
        if answer:
            result['answer'] = answer
        
        # Parse temporal segments
        segments = parse_temporal_segments(response)
        if segments:
            result['segments'] = segments
        
        # Extract and save masks
        if hasattr(model, 'seg') and len(model.seg) >= 1:
            # Get all object masks
            all_masks = [model.seg[obj_idx][0] for obj_idx in range(len(model.seg))]  # List of [num_frames, H, W]
            combined_masks_tensor = torch.stack(all_masks, dim=0)  # [num_objects, num_frames, H, W]
            merged_mask = torch.max(combined_masks_tensor, dim=0)[0]  # [num_frames, H, W]
            
            # Convert to numpy and binarize
            out = merged_mask.to(torch.uint8).cpu().numpy()
            out[out > 0] = 255  # Binarize: 0 or 255
            
            # Save masks
            mask_dir = save_masks_to_disk(out, mask_output_dir, item['entry_id'])
            result['mask_path'] = mask_dir
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return result


def main():
    parser = argparse.ArgumentParser(description='Run ST Evidence Task')
    
    # Model settings
    parser.add_argument('--model', type=str,
                        default='PolyU-ChenLab/UniPixel-3B',
                        help='UniPixel model path or HuggingFace model ID')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for inference (auto, cuda, cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Data type for inference')
    
    # Data settings
    parser.add_argument('--data-file', type=str,
                        default='data/st_evidence_gen.csv',
                        help='CSV file with video QA data')
    parser.add_argument('--video-dir', type=str,
                        default='data/videos_6fps',
                        help='Root directory containing video files')
    
    # Video processing
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second to sample (default: 1.0)')
    parser.add_argument('--max-frames', type=int, default=128,
                        help='Maximum frames to extract (default: 128)')
    parser.add_argument('--num-threads', type=int, default=8,
                        help='Number of threads for video decoding')
    
    # Generation settings
    parser.add_argument('--max-new-tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--do-sample', action='store_true',
                        help='Whether to use sampling for generation')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Temperature for sampling')
    
    # Output settings
    parser.add_argument('--output-dir', type=str,
                        default='results/ours',
                        help='Output directory for results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results file')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N samples')
    
    # Processing settings
    parser.add_argument('--start-idx', type=int, default=None,
                        help='Start index for processing')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='End index for processing')
    
    # Distributed processing (for parallel inference)
    parser.add_argument('--chunk', type=int, default=1,
                        help='Number of chunks to split the dataset')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of current chunk (0 to chunk-1)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = Path(__file__).parent
    if not os.path.isabs(args.data_file):
        args.data_file = os.path.join(script_dir, args.data_file)
    if not os.path.isabs(args.video_dir):
        args.video_dir = os.path.join(script_dir, args.video_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    
    print("=" * 80)
    print("ST EVIDENCE TASK")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data file: {args.data_file}")
    print(f"Video directory: {args.video_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"FPS: {args.fps}, Max frames: {args.max_frames}")
    print("=" * 80 + "\n")
    
    # Load data
    data = load_data(args.data_file, args.video_dir)
    total_samples = len(data)
    
    # Apply chunk-based splitting (for parallel inference)
    if args.chunk > 1:
        chunk_size = (total_samples + args.chunk - 1) // args.chunk
        start = args.index * chunk_size
        end = min(start + chunk_size, total_samples)
        data = data[start:end]
        print(f"🔍 Processing chunk {args.index + 1}/{args.chunk}: indices {start} to {end-1} ({len(data)} items)\n")
    # Apply manual subset filters (for debugging)
    elif args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else total_samples
        data = data[start:end]
        print(f"🔍 Processing subset: indices {start} to {end-1} ({len(data)} items)\n")
    
    # Determine output paths
    model_name = args.model.split('/')[-1].lower().replace('-', '_')
    
    # Add prefix if model name contains "3b"
    if '3b' in args.model.lower():
        output_name = f'3b_{model_name}_st_evidence_{args.fps}fps'
    elif '7b' in args.model.lower():
        output_name = f'7b_{model_name}_st_evidence_{args.fps}fps'
    else:
        output_name = f'{model_name}_st_evidence_{args.fps}fps'
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f'{output_name}.json')
    mask_output_dir = os.path.join(args.output_dir, output_name)
    os.makedirs(mask_output_dir, exist_ok=True)
    
    print(f"💾 Results will be saved to: {save_file}")
    print(f"🎭 Masks will be saved to: {mask_output_dir}\n")
    
    # Load existing results if resuming
    result_dict = {}
    if args.resume and os.path.exists(save_file):
        print(f"📂 Loading existing results from {save_file}")
        with open(save_file, 'r') as f:
            existing_results = json.load(f)
            if isinstance(existing_results, dict):
                result_dict = existing_results
            else:
                # Handle old format (list)
                for entry in existing_results:
                    if 'entry_id' in entry:
                        result_dict[entry['entry_id']] = entry
        print(f"✅ Loaded {len(result_dict)} existing results\n")
    
    # Filter items to process
    items_to_process = []
    skipped_count = 0
    retry_partial_count = 0
    
    for item in data:
        entry_id = item['entry_id']
        if entry_id in result_dict:
            parsed_response = result_dict[entry_id].get('parsed_response')
            if parsed_response is not None:
                # Check if all fields are valid
                if (parsed_response.get('answer') is not None and
                    parsed_response.get('segments') is not None and
                    parsed_response.get('mask_path') is not None):
                    skipped_count += 1
                    continue
                else:
                    retry_partial_count += 1
        
        items_to_process.append(item)
    
    print("📊 Processing summary:")
    print(f"  Total items: {len(data)}")
    print(f"  Already complete: {skipped_count}")
    print(f"  Retrying partial: {retry_partial_count}")
    print(f"  New items: {len(items_to_process) - retry_partial_count}")
    print(f"  To process: {len(items_to_process)}\n")
    
    if not items_to_process:
        print("✅ All items already processed!")
        return
    
    # Load model (build_model automatically handles LoRA adapters)
    print(f"🔄 Loading model: {args.model}...")
    model, processor = build_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        is_trainable=False,
        merge_adapter=False  # Keep adapter separate for inference
    )
    device = next(model.parameters()).device
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    print(f"✅ Model loaded on device: {device}\n")
    
    # Process items
    processed_count = 0
    
    for item in tqdm(items_to_process, desc="Processing"):
        entry_id = item['entry_id']
        video_id = item['video_id']
        
        print(f"\n{'=' * 80}")
        print(f"Entry ID: {entry_id}")
        print(f"Video: {Path(item['video_path']).name}")
        print(f"Question: {item['question'][:80]}...")
        
        try:
            # Process item
            parsed_result = process_single_item(
                item, model, processor, sam2_transform, device, args, mask_output_dir
            )
            
            # Store result
            result_dict[entry_id] = {
                'video_id': video_id,
                'parsed_response': parsed_result
            }
            
            # Report status
            has_answer = parsed_result.get('answer') is not None
            has_segments = parsed_result.get('segments') is not None
            has_mask_path = parsed_result.get('mask_path') is not None
            
            status_parts = []
            if has_answer:
                status_parts.append(f"Answer: {parsed_result['answer']}")
            if has_segments:
                status_parts.append(f"Segments: {len(parsed_result['segments'])}")
            if has_mask_path:
                status_parts.append("Masks: ✓")
            
            if has_answer and has_segments and has_mask_path:
                print(f"✅ Complete: {', '.join(status_parts)}")
            elif has_answer or has_segments or has_mask_path:
                print(f"⚠️  Partial: {', '.join(status_parts)}")
            else:
                print(f"❌ Failed")
            
            processed_count += 1
            
            # Save periodically with safe locking
            if processed_count % args.save_every == 0:
                result_dict = safe_save_json(result_dict, save_file)
                print(f"💾 Checkpoint saved ({processed_count}/{len(items_to_process)}, total: {len(result_dict)} entries)")
        
        except Exception as e:
            print(f"❌ Error processing {entry_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Store failed result
            result_dict[entry_id] = {
                'video_id': video_id,
                'parsed_response': {'answer': None, 'segments': None, 'mask_path': None},
                'error': str(e)
            }
    
    # Final save with safe locking
    result_dict = safe_save_json(result_dict, save_file)
    
    print("\n" + "=" * 80)
    print("🎉 PROCESSING COMPLETED!")
    print("=" * 80)
    
    # Calculate statistics
    final_total = len(result_dict)
    complete_count = sum(1 for entry in result_dict.values()
                        if entry.get('parsed_response') and
                        all(entry['parsed_response'].get(k) is not None
                            for k in ['answer', 'segments', 'mask_path']))
    partial_count = sum(1 for entry in result_dict.values()
                       if entry.get('parsed_response') and
                       any(entry['parsed_response'].get(k) is None
                           for k in ['answer', 'segments', 'mask_path']) and
                       any(entry['parsed_response'].get(k) is not None
                           for k in ['answer', 'segments', 'mask_path']))
    failed_count = final_total - complete_count - partial_count
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"  Total entries: {final_total}")
    print(f"  ✅ Complete: {complete_count}")
    print(f"  ⚠️  Partial: {partial_count}")
    print(f"  ❌ Failed: {failed_count}")
    if final_total > 0:
        print(f"  📍 Success rate: {(complete_count/final_total)*100:.1f}%")
    print(f"\n💾 Results: {save_file}")
    print(f"🎭 Masks: {mask_output_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
