#!/usr/bin/env python3
"""
SA2VA Three-Stage ST Evidence Task

Stages:
1. Answer the multiple-choice question
2. Identify temporal evidence segments
3. Generate segmentation masks (instead of referring expressions)
"""

import argparse
import json
import os
import re
import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Import SA2VA dependencies
import sys
import torch
from transformers import AutoModel, AutoProcessor

# Try to find SA2VA path (flexible configuration)
sa2va_paths = [
    os.environ.get('SA2VA_PATH'),  # Environment variable (highest priority)
    '/fsx/home/shijie.wang/code/Sa2VA',  # Original absolute path
    '../../../Sa2VA',  # Relative path (3 levels up)
    '../../Sa2VA',  # Relative path (2 levels up)
]

sa2va_found = False
for sa2va_path in sa2va_paths:
    if sa2va_path and os.path.exists(sa2va_path):
        sys.path.insert(0, sa2va_path)
        sa2va_found = True
        print(f"✓ Found SA2VA at: {sa2va_path}")
        break

if not sa2va_found:
    print("⚠️  Warning: SA2VA not found in expected locations.")
    print("   Set SA2VA_PATH environment variable or ensure Sa2VA is in the expected location.")
    print(f"   Tried: {[p for p in sa2va_paths if p]}")

from third_parts import VideoReader


def generate_sample_indices(total_frames, target_fps=1.0, video_fps=30.0, max_frames=32):
    """
    Generate sample indices based on target fps and maximum number of frames.
    
    Args:
        total_frames: Total number of frames in the video
        target_fps: Target sampling rate (frames per second)
        video_fps: Original video fps
        max_frames: Maximum number of frames to sample
    
    Returns:
        List of frame indices to sample
    """
    # Calculate interval between sampled frames
    interval = int(video_fps / target_fps)
    target_frame_num = int(total_frames / interval)
    
    if target_frame_num > max_frames:
        # uniformly sample max_frames frames using the new interval
        new_interval = int(total_frames / max_frames)
        indices = list(range(0, total_frames, new_interval))
        # Ensure we don't exceed max_frames
        if len(indices) > max_frames:
            indices = indices[:max_frames]
        return indices
    else:
        return list(range(0, total_frames, interval))


def resize_frame(frame, max_size=512):
    """
    Resize a PIL Image frame to have max dimension of max_size while maintaining aspect ratio.
    
    Args:
        frame: PIL Image
        max_size: Maximum size for width or height
    
    Returns:
        Resized PIL Image
    """
    w, h = frame.size
    
    # Calculate new dimensions maintaining aspect ratio
    if w > h:
        if w > max_size:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_w, new_h = w, h
    else:
        if h > max_size:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w, new_h = w, h
    
    # Resize if dimensions changed
    if (new_w, new_h) != (w, h):
        frame = frame.resize((new_w, new_h), Image.LANCZOS)
    
    return frame


def load_data(csv_path, video_dir):
    """Load data from CSV file."""
    import ast
    df = pd.read_csv(csv_path)
    data = []
    
    for _, row in df.iterrows():
        entry_id = row['entry_id']
        video_id = row['video_id']  # Read full video_id from CSV
        video_path = os.path.join(video_dir, row['video_path'])
        question = row['question']
        
        # Parse candidates (options)
        try:
            candidates = ast.literal_eval(row['candidates']) if isinstance(row['candidates'], str) else row['candidates']
        except:
            print(f"Warning: Failed to parse candidates for {entry_id}, skipping")
            continue
        
        data.append({
            'entry_id': entry_id,
            'video_id': video_id,
            'video_path': video_path,
            'question': question,
            'options': candidates
        })
    
    print(f"📊 Loaded {len(data)} QA pairs")
    return data


def create_answer_prompt(question, options):
    """Create prompt for stage 1: Answer the question"""
    option_letters = ['A', 'B', 'C', 'D', 'E']
    option_text = '\n'.join([f"{option_letters[i]}: {opt}" for i, opt in enumerate(options)])
    
    prompt = f"""Question: {question}

Options:
{option_text}

Please provide your answer as a single letter (A, B, C, D, or E) that corresponds to the correct option."""
    
    return prompt


def create_temporal_evidence_prompt():
    """Create prompt for stage 2: Identify temporal evidence"""
    prompt = """Now, identify the specific time segments in the video that serve as evidence for your answer.

Please provide the temporal evidence as a list of time segments in seconds, formatted as: [[start1, end1], [start2, end2], ...]

For example: [[2.5, 5.8], [7.1, 9.4]]"""
    
    return prompt


def create_segmentation_prompt():
    """Create prompt for stage 3: Generate segmentation masks"""
    prompt = """Please segment all the key objects in the video to serve as evidence for your answer."""

    return prompt


def parse_list_response(response_text):
    """Parse nested list from response text (for segments)."""
    import ast
    try:
        # Try to find nested list pattern first
        match = re.search(r'\[\[.*?\]\]', response_text, re.DOTALL)
        if match:
            result = ast.literal_eval(match.group(0))
            return result
    except:
        pass
    
    # If nested list not found, try to parse a simple list
    try:
        match = re.search(r'\[[\d\s,.\-]+\]', response_text)
        if match:
            result = ast.literal_eval(match.group(0))
            # Check if it's a simple list (not nested)
            if isinstance(result, list) and len(result) > 0:
                # If it's a simple list of numbers, wrap it in another list
                if all(isinstance(x, (int, float)) for x in result):
                    # Check if it looks like [start, end] format
                    if len(result) == 2:
                        return [result]  # Wrap single segment
                    # Otherwise, try to parse as pairs
                    elif len(result) % 2 == 0:
                        return [[result[i], result[i+1]] for i in range(0, len(result), 2)]
    except:
        pass
    
    return None


def generate_text_response(model, processor, frames, prompt, frame_indices=None, messages=None):
    """Generate text response from SA2VA model.
    
    Args:
        frames: All frames (for mask model)
        prompt: Current prompt text
        frame_indices: Indices of frames to send to LLM (optional, defaults to first 5)
        messages: Previous conversation messages (for multi-turn)
        
    Returns:
        response: Model's response text
        updated_messages: Updated conversation messages
    """
    try:
        result = model.predict_forward(
            video=frames,
            text=f"<image>{prompt}",
            past_text='',
            mask_prompts=None,
            processor=processor,
            frame_indices=frame_indices,
            messages=messages
        )
        response = result.get('prediction', '')
        
        # Update messages for next turn
        if messages is None:
            # First turn - messages were created inside predict_forward
            updated_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]  # Simplified since video already added
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        else:
            # Subsequent turns - append assistant response
            updated_messages = copy.deepcopy(messages)
            updated_messages.append({
                "role": "assistant",
                "content": response
            })
        
        return response, updated_messages
    except Exception as e:
        print(f"  ❌ Error generating text response: {e}")
        return None, messages


def generate_masks_sa2va(model, processor, frames, prompt, frame_indices=None, messages=None):
    """Generate segmentation masks from SA2VA model.
    
    Args:
        frames: All frames (for mask model)
        prompt: Current prompt text
        frame_indices: Indices of frames to send to LLM (optional, defaults to first 5)
        messages: Previous conversation messages (for multi-turn)
        
    Returns:
        response: Model's response text
        masks: List of segmentation masks
        updated_messages: Updated conversation messages
    """
    try:
        result = model.predict_forward(
            video=frames,
            text=f"<image>{prompt}",
            past_text='',
            mask_prompts=None,
            processor=processor,
            frame_indices=frame_indices,
            messages=messages
        )
        
        response = result.get('prediction', '')
        prediction_masks = result.get('prediction_masks', [])
        
        masks = []
        if prediction_masks:
            # SA2VA returns list of masks, each is [num_frames, H, W]
            for mask_set in prediction_masks:
                masks.append(np.array(mask_set))
        
        # Update messages for next turn
        if messages is None:
            # First turn - messages were created inside predict_forward
            updated_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                },
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        else:
            # Subsequent turns - append assistant response
            updated_messages = copy.deepcopy(messages)
            updated_messages.append({
                "role": "assistant",
                "content": response
            })
        
        return response, masks, updated_messages
    except Exception as e:
        print(f"  ❌ Error generating masks: {e}")
        return None, [], messages


def process_item_multi_turn(item, model, processor, args, mask_output_dir, skip_stage2=False):
    """Process single item in multi-turn mode.
    
    Args:
        skip_stage2: If True, skip stage 2 (temporal evidence generation)
    """
    result = {'answer': None, 'segments': None, 'mask_path': None}
    
    try:
        # Load video frames
        print("  🎬 Loading video frames...")
        reader = VideoReader(item['video_path'])
        original_fps = reader.fps
        total_frames = len(reader)
        
        # Load ALL frames (for mask generation)
        all_frames = []
        for idx in range(total_frames):
            frame = reader[idx]
            frame_rgb = frame[..., ::-1]  # BGR to RGB
            all_frames.append(Image.fromarray(frame_rgb))
        
        # Store original size for mask resizing later
        original_frame_size = all_frames[0].size if all_frames else (0, 0)
        
        # Resize ALL frames to max size 512 while keeping aspect ratio
        all_frames_resized = [resize_frame(frame, max_size=512) for frame in all_frames]
        resized_size = all_frames_resized[0].size if all_frames_resized else (0, 0)
        
        # Generate sample indices for LLM (using the same logic as sa2va.py)
        sample_indices = generate_sample_indices(
            total_frames=total_frames,
            target_fps=args.fps,
            video_fps=original_fps,
            max_frames=args.max_frames
        )
        
        print(f"  🎬 Loaded {total_frames} frames (all for masks)")
        print(f"  📐 Resized frames from {original_frame_size} to {resized_size}")
        print(f"  🎯 LLM will see {len(sample_indices)} sampled frames: {sample_indices[:10]}{'...' if len(sample_indices) > 10 else ''}")
        
        # Initialize conversation messages
        messages = None
        
        # Stage 1: Answer the question
        print("  📝 Stage 1: Answering question...")
        answer_prompt = create_answer_prompt(item['question'], item['options'])
        answer_response, messages_stage1 = generate_text_response(
            model, processor, all_frames_resized, answer_prompt, 
            frame_indices=sample_indices, messages=messages
        )
        
        if answer_response is None:
            return result
        
        print(f"  Answer response: {answer_response[:100]}")
        
        # Parse answer
        answer = None
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in answer_response.upper():
                answer = letter
                break
        
        if answer:
            result['answer'] = answer
            print(f"  ✅ Answer: {answer}")
        else:
            print(f"  ⚠️ Failed to extract answer")
            return result
        
        # Stage 2: Get temporal evidence (with context from Stage 1 ONLY)
        if skip_stage2:
            print("  ⏭️  Stage 2: Skipped (--skip-stage2)")
        else:
            print("  📝 Stage 2: Identifying temporal evidence...")
            temporal_prompt = create_temporal_evidence_prompt()
            segments_response, messages_stage2 = generate_text_response(
                model, processor, all_frames_resized, temporal_prompt, 
                frame_indices=sample_indices, messages=copy.deepcopy(messages_stage1)
            )
            
            if segments_response is not None:
                print(f"  Segments response: {segments_response[:100]}")
                
                # Parse segments
                segments = parse_list_response(segments_response)
                if segments:
                    result['segments'] = segments
                    print(f"  ✅ Segments: {segments}")
                else:
                    print(f"  ⚠️ Failed to parse segments")
            else:
                print(f"  ⚠️ Failed to generate segments response")
        
        # Stage 3: Generate segmentation masks (with context from Stage 1 ONLY, not Stage 2)
        print("  📝 Stage 3: Generating segmentation masks...")
        seg_prompt = create_segmentation_prompt()
        seg_response, masks, messages_stage3 = generate_masks_sa2va(
            model, processor, all_frames_resized, seg_prompt, 
            frame_indices=sample_indices, messages=copy.deepcopy(messages_stage1)
        )
        
        if seg_response:
            print(f"  Segmentation response: {seg_response[:100]}")
        
        if len(masks) > 0:
            # Save masks in structure: {entry_id}/masks/
            entry_id = item['entry_id']
            video_entry_dir = os.path.join(mask_output_dir, entry_id)
            video_mask_dir = os.path.join(video_entry_dir, 'masks')
            os.makedirs(video_mask_dir, exist_ok=True)
            
            # Combine all object masks
            # SA2VA masks: each is [num_frames, H, W]
            if len(masks) == 1:
                merged_mask = masks[0]
            else:
                # Stack and merge multiple object masks
                stacked_masks = np.stack(masks, axis=0)  # [num_objects, num_frames, H, W]
                merged_mask = np.max(stacked_masks, axis=0)  # [num_frames, H, W]
            
            # Convert to uint8 and binarize
            out = (merged_mask * 255).astype(np.uint8)
            out[out > 0] = 255  # Binarize: 0 or 255
            
            # Save each frame as PNG, resized back to original size
            for frm_idx in range(out.shape[0]):
                # Resize mask back to original frame size using NEAREST interpolation for binary masks
                mask_resized = cv2.resize(out[frm_idx], original_frame_size, interpolation=cv2.INTER_NEAREST)
                
                frame_name = f"{str(frm_idx).zfill(5)}.png"
                mask_path_frame = os.path.join(video_mask_dir, frame_name)
                cv2.imwrite(mask_path_frame, mask_resized)
            
            # Store mask directory path (full path to masks folder)
            result['mask_path'] = video_mask_dir
            print(f"  ✅ Generated {len(masks)} object masks, saved to {video_mask_dir}")
            print(f"  📐 Masks resized to original size: {original_frame_size}")
        else:
            print(f"  ⚠️ No masks generated")
            return result
        
    except Exception as e:
        print(f"  ❌ Multi-turn processing error: {e}")
        import traceback
        traceback.print_exc()
        return result
    
    return result


def main():
    parser = argparse.ArgumentParser(description='SA2VA three-stage ST evidence task')
    parser.add_argument('--model', type=str, 
                        default='Sa2VA-Qwen2_5-VL-7B',
                        help='SA2VA model path')
    parser.add_argument('--data-file', type=str,
                        default='data/st_evidence_gen.csv',
                        help='CSV file with video data')
    parser.add_argument('--video-dir', type=str, 
                        default='data/videos_6fps',
                        help='Video directory')
    parser.add_argument('--fps', type=float, default=1.0, 
                        help='Frames per second to sample (default: 1.0)')
    parser.add_argument('--max-frames', type=int, default=16, 
                        help='Max frames to extract (default: 16)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--skip-stage2', action='store_true',
                        help='Skip stage 2 (temporal evidence generation)')
    
    args = parser.parse_args()
    
    # Print configuration
    print(f"\n{'='*80}")
    print(f"SA2VA Three-Stage ST Evidence Task")
    if args.skip_stage2:
        print(f"⏭️  Stage 2 (temporal evidence) will be SKIPPED")
    else:
        print(f"✅ All three stages will be executed")
    print(f"{'='*80}\n")
    
    # Load data
    data = load_data(args.data_file, args.video_dir)
    
    # Determine save file path
    model_name = args.model.split('/')[-1].lower().replace('-', '_')
    save_dir = 'results/sa2va'
    os.makedirs(save_dir, exist_ok=True)
    
    # JSON file name (without extension)
    json_basename = f'{model_name}_st_evidence_{args.fps}fps'
    save_file = os.path.join(save_dir, f'{json_basename}.json')
    
    # Mask output directory: same name as JSON (without .json)
    mask_output_dir = os.path.join(save_dir, json_basename)
    os.makedirs(mask_output_dir, exist_ok=True)
    
    print(f"\n💾 Results will be saved to: {save_file}")
    print(f"🎭 Masks will be saved to: {mask_output_dir}")
    
    # Load existing results if resuming
    result_dict = {}
    if os.path.exists(save_file):
        print(f"\n📂 Loading existing results from {save_file}")
        with open(save_file, 'r') as f:
            existing_results = json.load(f)
            # Results are stored as dict with entry_id as keys
            if isinstance(existing_results, dict):
                result_dict = existing_results
            else:
                # Handle old format (list)
                for entry in existing_results:
                    if 'entry_id' in entry:
                        result_dict[entry['entry_id']] = entry
        print(f"✅ Loaded {len(result_dict)} existing results")
    
    # Filter items to process (skip complete and partial entries)
    items_to_process = []
    skipped_complete_count = 0
    skipped_partial_count = 0
    
    for item in data:
        entry_id = item['entry_id']
        if entry_id in result_dict:
            parsed_response = result_dict[entry_id].get('parsed_response')
            if parsed_response is not None:
                # Check if required fields are present based on skip_stage2 flag
                if args.skip_stage2:
                    # When skipping stage 2, only check answer and mask_path
                    is_complete = (parsed_response.get('answer') is not None and 
                                   parsed_response.get('mask_path') is not None)
                else:
                    # Normal mode: check all three fields
                    is_complete = (parsed_response.get('answer') is not None and 
                                   parsed_response.get('segments') is not None and 
                                   parsed_response.get('mask_path') is not None)
                
                if is_complete:
                    # Complete result, skip
                    skipped_complete_count += 1
                    continue
                else:
                    # Partial result, skip (don't retry)
                    skipped_partial_count += 1
                    continue
            else:
                # Failed result (parsed_response is None), retry
                items_to_process.append(item)
        else:
            # New item
            items_to_process.append(item)
    
    print(f"\n📊 Processing summary:")
    print(f"  Total items: {len(data)}")
    print(f"  Already complete: {skipped_complete_count}")
    print(f"  Skipped partial: {skipped_partial_count}")
    print(f"  New items: {len(items_to_process)}")
    print(f"  To process: {len(items_to_process)}")
    
    if not items_to_process:
        print("✅ All done!")
        return
    
    # Load SA2VA model
    print(f"\n🔄 Loading {args.model}...")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().to(args.device)
    
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    print(f"✅ Model loaded on device: {args.device}")
    
    # Process items
    processed_count = 0
    
    for item in tqdm(items_to_process):
        entry_id = item['entry_id']
        video_id = item['video_id']
        
        print(f"\n{'='*80}")
        print(f"Processing: {entry_id}")
        print(f"Video: {Path(item['video_path']).name}")
        print(f"Question: {item['question']}")
        
        try:
            # Process in multi-turn mode
            parsed_result = process_item_multi_turn(item, model, processor, args, mask_output_dir, skip_stage2=args.skip_stage2)
            
            # Store result (matching qwen3vl format)
            result_dict[entry_id] = {
                'video_id': video_id,
                'parsed_response': parsed_result
            }
            
            # Report status
            has_answer = parsed_result.get('answer') is not None
            has_segments = parsed_result.get('segments') is not None
            has_mask_path = parsed_result.get('mask_path') is not None
            
            if args.skip_stage2:
                # When skipping stage 2, only check answer and mask_path
                if has_answer and has_mask_path:
                    print(f"✅ Successfully processed {entry_id}")
                elif has_answer or has_mask_path:
                    successful_parts = []
                    if has_answer:
                        successful_parts.append('answer')
                    if has_mask_path:
                        successful_parts.append('mask_path')
                    print(f"⚠️ Partially processed {entry_id} (successful: {', '.join(successful_parts)})")
                else:
                    print(f"❌ Failed to process {entry_id}")
            else:
                # Normal mode: check all three fields
                if has_answer and has_segments and has_mask_path:
                    print(f"✅ Successfully processed {entry_id}")
                elif has_answer or has_segments or has_mask_path:
                    successful_parts = []
                    if has_answer:
                        successful_parts.append('answer')
                    if has_segments:
                        successful_parts.append('segments')
                    if has_mask_path:
                        successful_parts.append('mask_path')
                    print(f"⚠️ Partially processed {entry_id} (successful: {', '.join(successful_parts)})")
                else:
                    print(f"❌ Failed to process {entry_id}")
            
            processed_count += 1
            
            # Save periodically
            if processed_count % 10 == 0:
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                print(f"💾 Saved checkpoint ({processed_count}/{len(items_to_process)} processed)")
                
        except Exception as e:
            print(f"❌ Error processing {entry_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Store failed result (matching qwen3vl format)
            result_dict[entry_id] = {
                'video_id': video_id,
                'parsed_response': {'answer': None, 'segments': None, 'mask_path': None},
                'error': str(e)
            }
            continue
    
    # Final save (as dict with entry_id keys)
    with open(save_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print("\n🎉 Processing completed!")
    
    # Calculate final statistics
    final_total = len(result_dict)
    complete_count = sum(1 for entry in result_dict.values() 
                        if entry.get('parsed_response') and
                        all(entry['parsed_response'].get(k) is not None 
                            for k in ['answer', 'segments', 'mask_path']))
    partial_count = sum(1 for entry in result_dict.values()
                       if entry.get('parsed_response') and
                       any(entry['parsed_response'].get(k) is None 
                           for k in ['answer', 'segments', 'mask_path']))
    failed_count = final_total - complete_count - partial_count
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"📊 Total entries: {final_total}")
    print(f"✅ Complete responses: {complete_count}")
    print(f"⚠️  Partial responses: {partial_count}")
    print(f"❌ Failed responses: {failed_count}")
    print(f"📍 Success rate: {(complete_count/final_total)*100:.1f}%" if final_total > 0 else "📍 Success rate: 0%")
    print(f"💾 Results saved to: {save_file}")
    print(f"🎭 Masks saved to: {mask_output_dir}")


if __name__ == '__main__':
    main()

