# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import nncore
import numpy as np
import pandas as pd
import torch
from decord import VideoReader
import decord
from tqdm import tqdm

from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_frames_with_stride
from unipixel.utils.transforms import get_sam2_transform


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


def generate_text_response(model, processor, frames, images, prompt, device, messages=None):
    """Generate text response from the model with optional chat history."""
    num_frames = len(images)
    
    if messages is None:
        # First message - include video
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
    else:
        # Continuation - only add text (video already in context)
        messages.append({
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': prompt
            }]
        })
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images_proc, videos_proc, kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    data = processor(text=[text], images=images_proc, videos=videos_proc, return_tensors='pt', **kwargs)
    
    # For text-only response, we don't need SAM2 frames
    output_ids = model.generate(
        **data.to(device),
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_new_tokens=512)
    
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    # Add assistant response to messages for next turn
    messages.append({
        'role': 'assistant',
        'content': [{
            'type': 'text',
            'text': response
        }]
    })
    
    return response, messages


def generate_masks(model, processor, frames, images, prompt, device, messages=None):
    """Generate segmentation masks from the model with optional chat history."""
    num_frames = len(images)
    
    if messages is None:
        # First message - include video
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
    else:
        # Continuation - only add text (video already in context)
        messages.append({
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': prompt
            }]
        })
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images_proc, videos_proc, kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    data = processor(text=[text], images=images_proc, videos=videos_proc, return_tensors='pt', **kwargs)
    
    # Add SAM2 frames for segmentation
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
    data['frame_size'] = [frames.shape[1:3]]
    
    output_ids = model.generate(
        **data.to(device),
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        max_new_tokens=512)
    
    output_ids = output_ids[0, data.input_ids.size(1):]
    if output_ids[-1] == processor.tokenizer.eos_token_id:
        output_ids = output_ids[:-1]
    
    response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
    
    # Get masks from model
    masks = model.seg if len(model.seg) >= 1 else []
    
    return response, masks


def process_item_multi_turn(item, model, processor, device, args, mask_output_dir):
    """Process single item in multi-turn mode."""
    result = {'answer': None, 'segments': None, 'mask_path': None}
    
    try:
        # Load video - ALL frames for mask generation, sampled frames for LLM
        decord.bridge.set_bridge('torch')
        vr = VideoReader(item['video_path'], num_threads=args.num_threads)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        # Calculate sample_frames
        if args.fps is not None:
            video_duration = total_frames / original_fps
            sample_frames = int(video_duration * args.fps)
            sample_frames = max(1, min(sample_frames, min(args.max_frames, total_frames)))
        else:
            sample_frames = min(args.max_frames, total_frames)
        
        frames, images, inds = load_frames_with_stride(
            item['video_path'],
            every_n_frames=1,
            sample_frames=sample_frames,
            sample_type='uniform',
            sample_for_llm_only=True,
            num_threads=args.num_threads
        )
        
        print(f"  🎬 Loaded {total_frames} frames (all for masks), {len(images)} sampled for LLM")
        
        # Initialize messages (will be built up across stages)
        messages = None
        
        # Stage 1: Answer the question
        print("  📝 Stage 1: Answering question...")
        answer_prompt = create_answer_prompt(item['question'], item['options'])
        answer_response, messages_stage1 = generate_text_response(model, processor, frames, images, answer_prompt, device, messages)
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
        
        # Save a copy of Stage 1 messages for Stage 3 (before Stage 2 modifies it)
        import copy
        messages_stage1_copy = copy.deepcopy(messages_stage1)
        
        # Stage 2: Get temporal evidence (with context from Stage 1)
        print("  📝 Stage 2: Identifying temporal evidence...")
        temporal_prompt = create_temporal_evidence_prompt()
        segments_response, messages_stage2 = generate_text_response(model, processor, frames, images, temporal_prompt, device, messages_stage1)
        print(f"  Segments response: {segments_response[:100]}")
        
        # Parse segments
        segments = parse_list_response(segments_response)
        if segments:
            result['segments'] = segments
            print(f"  ✅ Segments: {segments}")
        else:
            print(f"  ⚠️ Failed to parse segments (continuing to Stage 3)")
        
        # Stage 3: Generate segmentation masks (with context from Stage 1 only, not Stage 2)
        print("  📝 Stage 3: Generating segmentation masks...")
        seg_prompt = create_segmentation_prompt()
        seg_response, masks = generate_masks(model, processor, frames, images, seg_prompt, device, messages_stage1_copy)
        print(f"  Segmentation response: {seg_response[:100]}")
        
        if len(masks) > 0:
            # Save masks in structure: {entry_id}/masks/
            entry_id = item['entry_id']
            video_entry_dir = os.path.join(mask_output_dir, entry_id)
            video_mask_dir = os.path.join(video_entry_dir, 'masks')
            nncore.mkdir(video_mask_dir)
            
            # Combine all object masks
            all_masks = [masks[obj_idx][0] for obj_idx in range(len(masks))]  # List of [num_frames, H, W]
            combined_masks_tensor = torch.stack(all_masks, dim=0)  # [num_objects, num_frames, H, W]
            merged_mask = torch.max(combined_masks_tensor, dim=0)[0]  # [num_frames, H, W]
            
            # Convert to numpy and binarize
            out = merged_mask.to(torch.uint8).cpu().numpy()
            out[out > 0] = 255  # Binarize: 0 or 255
            
            # Save each frame as PNG
            for frm_idx in range(out.shape[0]):
                frame_name = f"{str(frm_idx).zfill(5)}.png"
                mask_path_frame = os.path.join(video_mask_dir, frame_name)
                cv2.imwrite(mask_path_frame, out[frm_idx])
            
            # Store mask directory path (full path to masks folder)
            result['mask_path'] = video_mask_dir
            print(f"  ✅ Generated {len(masks)} object masks, saved to {video_mask_dir}")
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
    parser = argparse.ArgumentParser(description='UniPixel three-stage ST evidence task')
    parser.add_argument('--model', type=str, 
                        default='PolyU-ChenLab/UniPixel-3B',
                        help='UniPixel model path')
    parser.add_argument('--data-file', type=str,
                        default='data/st_evidence_gen.csv',
                        help='CSV file with video data')
    parser.add_argument('--video-dir', type=str, 
                        default='data/videos_6fps',
                        help='Video directory')
    parser.add_argument('--fps', type=float, default=1.0, 
                        help='Frames per second to sample (default: 1.0)')
    parser.add_argument('--max-frames', type=int, default=128, 
                        help='Max frames to extract (default: 128)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for inference')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Data type for inference')
    parser.add_argument('--num-threads', type=int, default=8,
                        help='Number of threads for video decoding')
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.data_file, args.video_dir)
    
    # Determine save file path
    model_name = args.model.split('/')[-1].lower().replace('-', '_')
    save_dir = 'results/unipixel'
    nncore.mkdir(save_dir)
    
    # JSON file name (without extension)
    json_basename = f'{model_name}_st_evidence_{args.fps}fps'
    save_file = os.path.join(save_dir, f'{json_basename}.json')
    
    # Mask output directory: same name as JSON (without .json)
    mask_output_dir = os.path.join(save_dir, json_basename)
    nncore.mkdir(mask_output_dir)
    
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
    
    # Filter items to process (skip complete entries)
    items_to_process = []
    skipped_count = 0
    retry_partial_count = 0
    
    for item in data:
        entry_id = item['entry_id']
        if entry_id in result_dict:
            parsed_response = result_dict[entry_id].get('parsed_response')
            if parsed_response is not None:
                # Check if all three fields are present and valid
                if (parsed_response.get('answer') is not None and 
                    parsed_response.get('segments') is not None and 
                    parsed_response.get('mask_path') is not None):
                    # Complete result, skip
                    skipped_count += 1
                    continue
                else:
                    # Partial result, retry
                    retry_partial_count += 1
                    items_to_process.append(item)
            else:
                # Failed result, retry
                items_to_process.append(item)
        else:
            # New item
            items_to_process.append(item)
    
    print(f"\n📊 Processing summary:")
    print(f"  Total items: {len(data)}")
    print(f"  Already complete: {skipped_count}")
    print(f"  Retrying partial: {retry_partial_count}")
    print(f"  New items: {len(items_to_process) - retry_partial_count}")
    print(f"  To process: {len(items_to_process)}")
    
    if not items_to_process:
        print("✅ All done!")
        return
    
    # Build model
    print(f"\n🔄 Loading {args.model}...")
    model, processor = build_model(args.model, device=args.device, dtype=args.dtype)
    device = next(model.parameters()).device
    print(f"✅ Model loaded on device: {device}")
    
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
            # Process in multi-turn mode (UniPixel doesn't have chat history, so always multi-turn)
            parsed_result = process_item_multi_turn(item, model, processor, device, args, mask_output_dir)
            
            # Store result (matching qwen3vl format)
            result_dict[entry_id] = {
                'video_id': video_id,
                'parsed_response': parsed_result
            }
            
            # Report status
            has_answer = parsed_result.get('answer') is not None
            has_segments = parsed_result.get('segments') is not None
            has_mask_path = parsed_result.get('mask_path') is not None
            
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

