# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v3 as iio
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
from unipixel.utils.visualizer import draw_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Generate video masks using UniPixel')
    parser.add_argument('ref_exp_file', type=str, help='Path to ref-exp JSON file')
    parser.add_argument('--csv_file', type=str,
                        default='data/st_evidence_gen.csv',
                        help='Path to st_evidence_final.csv file')
    parser.add_argument('--video_dir', type=str,
                        default='data/videos_6fps',
                        help='Base directory for videos')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated masks (default: auto-generated from ref_exp_file)')
    parser.add_argument('--model_path', type=str, default='PolyU-ChenLab/UniPixel-3B',
                        help='Path to UniPixel model')
    parser.add_argument('--mode', type=str, choices=['seperate', 'concat', 'both'], default='both',
                        help='Mode for generating masks: seperate, concat, or both')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for inference')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Data type for inference')
    parser.add_argument('--save_masks', action='store_true',
                        help='Save individual mask PNG files (one per frame) in addition to visualizations')
    parser.add_argument('--skip_viz', action='store_true',
                        help='Skip saving GIF visualizations (only save mask PNGs if --save_masks is set)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for seperate mode inference (default: 1, set higher for speedup)')
    parser.add_argument('--every_n_frames', type=int, default=None,
                        help='Stride for sampling frames for LLM (1=all frames, 4=every 4th frame, etc.)')
    parser.add_argument('--target_fps', type=float, default=None,
                        help='Target FPS for sampling frames for LLM (e.g., 1.0 for 1 FPS)')
    parser.add_argument('--frame_num', type=int, default=None,
                        help='Exact number of frames to sample for LLM')
    parser.add_argument('--num_threads', type=int, default=8,
                        help='Number of threads for video decoding')
    parser.add_argument('--chunk', type=int, default=1,
                        help='Total number of chunks to split the data into (for multi-GPU)')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of this chunk (0 to chunk-1, for multi-GPU)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that have already been processed (check if mask directory exists)')
    return parser.parse_args()


def build_prompt_individual(ref_expression):
    """Build prompt for individual ref expression."""
    return f"Please segment {ref_expression}"


def build_prompt_concatenated(ref_expressions):
    """Build prompt for concatenated ref expressions."""
    if len(ref_expressions) == 1:
        return f"Please segment {ref_expressions[0]}"
    elif len(ref_expressions) == 2:
        return f"Please segment {ref_expressions[0]} and {ref_expressions[1]}"
    else:
        refs = ", ".join(ref_expressions[:-1])
        return f"Please segment {refs}, and {ref_expressions[-1]}"


def generate_mask(model, processor, frames, images, prompt, device):
    """Generate mask for given frames and prompt."""
    num_frames = len(images)
    messages = [{
        'role': 'user',
        'content': [{
            'type': 'video',
            'video': images,  # Sampled frames for LLM
            'min_pixels': 128 * 28 * 28,
            'max_pixels': 256 * 28 * 28 * num_frames
        }, {
            'type': 'text',
            'text': prompt
        }]
    }]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images_proc, videos_proc, kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    data = processor(text=[text], images=images_proc, videos=videos_proc, return_tensors='pt', **kwargs)
    
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


def generate_mask_batch(model, processor, frames, images, prompts, device):
    """Generate masks for multiple prompts in batch."""
    num_frames = len(images)
    batch_size = len(prompts)
    
    # Prepare batch data
    all_texts = []
    all_images_proc = []
    all_videos_proc = []
    all_kwargs_list = []
    
    for prompt in prompts:
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'video',
                'video': images,  # Sampled frames for LLM
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]
        
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images_proc, videos_proc, kwargs = process_vision_info(messages, return_video_kwargs=True)
        
        all_texts.append(text)
        all_images_proc.append(images_proc)
        all_videos_proc.append(videos_proc)
        all_kwargs_list.append(kwargs)
    
    # Process batch
    # Note: processor may not support true batching with videos, so we process sequentially
    # but collect results to return together
    batch_responses = []
    batch_masks = []
    
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    
    for i in range(batch_size):
        data = processor(text=[all_texts[i]], images=all_images_proc[i], 
                        videos=all_videos_proc[i], return_tensors='pt', **all_kwargs_list[i])
        
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
        masks = model.seg if len(model.seg) >= 1 else []
        
        batch_responses.append(response)
        batch_masks.append(masks)
    
    return batch_responses, batch_masks


def process_video(entry_id, video_path, ref_expressions, model, processor, device, 
                  args, seperate_dir, concat_dir):
    """Process a single video and generate masks."""
    full_video_path = os.path.join(args.video_dir, video_path)
    
    if not os.path.exists(full_video_path):
        print(f"Warning: Video not found: {full_video_path}")
        return None
    
    # Check if already processed (if skip_existing is enabled)
    if args.skip_existing:
        skip_seperate = False
        skip_concat = False
        
        if args.mode in ['seperate', 'both']:
            video_seperate_dir = os.path.join(seperate_dir, entry_id)
            mask_dir = os.path.join(video_seperate_dir, 'masks')
            # Check if mask directory exists and has PNG files
            if os.path.exists(mask_dir) and len([f for f in os.listdir(mask_dir) if f.endswith('.png')]) > 0:
                skip_seperate = True
        
        if args.mode in ['concat', 'both']:
            video_concat_dir = os.path.join(concat_dir, entry_id)
            mask_dir = os.path.join(video_concat_dir, 'masks')
            # Check if mask directory exists and has PNG files
            if os.path.exists(mask_dir) and len([f for f in os.listdir(mask_dir) if f.endswith('.png')]) > 0:
                skip_concat = True
        
        # If all required modes are already processed, skip this video
        if args.mode == 'seperate' and skip_seperate:
            print(f"  Skipping {entry_id} (seperate already processed)")
            return None
        elif args.mode == 'concat' and skip_concat:
            print(f"  Skipping {entry_id} (concat already processed)")
            return None
        elif args.mode == 'both' and skip_seperate and skip_concat:
            print(f"  Skipping {entry_id} (both modes already processed)")
            return None
    
    # Load video - ALL frames for mask generation, sampled frames for LLM
    try:
        decord.bridge.set_bridge('torch')
        vr = VideoReader(full_video_path, num_threads=args.num_threads)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        # Calculate sample_frames based on the sampling strategy
        if args.frame_num is not None:
            # Exact number of frames
            sample_frames = min(args.frame_num, total_frames)
            sampling_info = f"frame_num={args.frame_num}"
        elif args.target_fps is not None:
            # Target FPS: calculate frames based on video duration
            video_duration = total_frames / original_fps
            sample_frames = int(video_duration * args.target_fps)
            sample_frames = max(1, min(sample_frames, total_frames))  # At least 1, at most total_frames
            sampling_info = f"target_fps={args.target_fps}"
        elif args.every_n_frames is not None:
            # Every N frames stride
            sample_frames = total_frames // args.every_n_frames
            if total_frames % args.every_n_frames != 0:
                sample_frames += 1
            sampling_info = f"every_n_frames={args.every_n_frames}"
        else:
            # Should not reach here due to validation, but handle anyway
            sample_frames = total_frames
            sampling_info = "all frames"
        
        frames, images, inds = load_frames_with_stride(
            full_video_path,
            every_n_frames=1,  # Include ALL frames for mask generation
            sample_frames=sample_frames,  # Sample subset for LLM
            sample_type='uniform',
            sample_for_llm_only=True,  # KEY: frames=ALL, images=sampled
            num_threads=args.num_threads
        )
        
        print(f"  Loaded {total_frames} frames (all for masks), {len(images)} sampled for LLM ({sampling_info}) from {video_path}")
    except Exception as e:
        print(f"Error loading video {full_video_path}: {e}")
        return None
    
    results = {
        'entry_id': entry_id,
        'video_path': video_path,
        'ref_expressions': ref_expressions,
        'num_frames': len(images),
        'seperate': {},
        'concat': {}
    }
    
    # Mode 1: Seperate ref expressions (run individually then combine)
    if args.mode in ['seperate', 'both']:
        video_seperate_dir = os.path.join(seperate_dir, entry_id)
        
        # Skip if already processed (checked earlier, but handle partial processing)
        if args.skip_existing:
            mask_dir = os.path.join(video_seperate_dir, 'masks')
            if os.path.exists(mask_dir) and len([f for f in os.listdir(mask_dir) if f.endswith('.png')]) > 0:
                print(f"  Skipping seperate mode for {entry_id} (already processed)")
                results['seperate'] = {'skipped': True, 'reason': 'already processed'}
                if args.mode == 'seperate':
                    return results
                else:
                    pass  # Continue to concat mode
            else:
                nncore.mkdir(video_seperate_dir)
        else:
            nncore.mkdir(video_seperate_dir)
        
        collected_masks = []  # Collect masks from each inference
        collected_responses = []
        
        # Process in batches for speedup
        batch_size = min(args.batch_size, len(ref_expressions))
        
        for batch_start in range(0, len(ref_expressions), batch_size):
            batch_end = min(batch_start + batch_size, len(ref_expressions))
            batch_ref_exprs = ref_expressions[batch_start:batch_end]
            batch_prompts = [build_prompt_individual(ref_expr) for ref_expr in batch_ref_exprs]
            
            try:
                if len(batch_prompts) == 1:
                    # Single inference
                    response, masks = generate_mask(model, processor, frames, images, 
                                                   batch_prompts[0], device)
                    batch_responses = [response]
                    batch_masks = [masks]
                else:
                    # Batch inference
                    batch_responses, batch_masks = generate_mask_batch(model, processor, frames, images,
                                                                       batch_prompts, device)
                
                # Collect masks and responses
                for idx, (ref_expr, prompt, response, masks) in enumerate(zip(batch_ref_exprs, batch_prompts, 
                                                                              batch_responses, batch_masks)):
                    if len(masks) >= 1:
                        collected_masks.append(masks[0][0])  # [num_frames, H, W]
                        collected_responses.append({
                            'ref_expression': ref_expr,
                            'prompt': prompt,
                            'response': response
                        })
                    else:
                        print(f"Warning: No masks generated for '{ref_expr}'")
                        
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                for idx, ref_expr in enumerate(batch_ref_exprs):
                    results['seperate'][batch_start + idx] = {
                        'ref_expression': ref_expr,
                        'error': str(e)
                    }
        
        # After collecting all masks, save the combined result
        if len(collected_masks) > 0:
            # Save visualization GIF/PNG (unless skipped) - combine all masks
            output_path = None
            if not args.skip_viz:
                # Stack collected masks: [num_objects, num_frames, H, W]
                combined_masks = torch.stack(collected_masks, dim=0).unsqueeze(0)  # [1, num_objects, num_frames, H, W]
                imgs = draw_mask(frames, [combined_masks[0]])  # Pass [num_objects, num_frames, H, W]
                output_path = os.path.join(video_seperate_dir, 
                                          'all_refs.gif' if len(imgs) > 1 else 'all_refs.png')
                iio.imwrite(output_path, imgs, duration=100, loop=0)
            
            # Save combined mask PNGs if requested
            mask_dir = None
            if args.save_masks:
                mask_dir = os.path.join(video_seperate_dir, 'masks')
                nncore.mkdir(mask_dir)
                
                # Combine all masks into one: merge all objects
                # Stack and take max across objects dimension to merge
                combined_masks_tensor = torch.stack(collected_masks, dim=0)  # [num_objects, num_frames, H, W]
                merged_mask = torch.max(combined_masks_tensor, dim=0)[0]  # [num_frames, H, W]
                
                # Convert to numpy and binarize
                out = merged_mask.to(torch.uint8).cpu().numpy()
                out[out > 0] = 255  # Binarize: 0 or 255
                
                # Save each frame as PNG
                for frm_idx in range(out.shape[0]):
                    frame_name = f"{str(frm_idx).zfill(5)}.png"
                    mask_path = os.path.join(mask_dir, frame_name)
                    cv2.imwrite(mask_path, out[frm_idx])
            
            results['seperate'] = {
                'ref_expressions': ref_expressions,
                'num_objects': len(collected_masks),
                'num_frames': collected_masks[0].shape[0] if len(collected_masks) > 0 else len(images),
                'output_path': output_path,
                'mask_dir': mask_dir,
                'individual_results': collected_responses
            }
        else:
            results['seperate'] = {
                'ref_expressions': ref_expressions,
                'num_objects': 0,
                'num_frames': 0,
                'output_path': None,
                'mask_dir': None,
                'individual_results': []
            }
    
    # Mode 2: Concat ref expressions
    if args.mode in ['concat', 'both']:
        video_concat_dir = os.path.join(concat_dir, entry_id)
        
        # Skip if already processed (checked earlier, but handle partial processing)
        if args.skip_existing:
            mask_dir = os.path.join(video_concat_dir, 'masks')
            if os.path.exists(mask_dir) and len([f for f in os.listdir(mask_dir) if f.endswith('.png')]) > 0:
                print(f"  Skipping concat mode for {entry_id} (already processed)")
                results['concat'] = {'skipped': True, 'reason': 'already processed'}
                return results
            else:
                nncore.mkdir(video_concat_dir)
        else:
            nncore.mkdir(video_concat_dir)
        
        prompt = build_prompt_concatenated(ref_expressions)
        
        try:
            response, masks = generate_mask(model, processor, frames, images, 
                                           prompt, device)
            
            # Save results if masks were generated
            if len(masks) >= 1:
                # Save visualization GIF/PNG (unless skipped)
                output_path = None
                if not args.skip_viz:
                    imgs = draw_mask(frames, masks)
                    output_path = os.path.join(video_concat_dir, 
                                              'all_refs.gif' if len(imgs) > 1 else 'all_refs.png')
                    iio.imwrite(output_path, imgs, duration=100, loop=0)
                
                # Save combined mask PNGs if requested
                mask_dir = None
                if args.save_masks:
                    mask_dir = os.path.join(video_concat_dir, 'masks')
                    nncore.mkdir(mask_dir)
                    
                    # Combine all masks into one: merge all objects
                    if len(masks) > 0:
                        # Stack all object masks and merge
                        all_masks = [masks[obj_idx][0] for obj_idx in range(len(masks))]  # List of [num_frames, H, W]
                        combined_masks_tensor = torch.stack(all_masks, dim=0)  # [num_objects, num_frames, H, W]
                        merged_mask = torch.max(combined_masks_tensor, dim=0)[0]  # [num_frames, H, W]
                        
                        # Convert to numpy and binarize
                        out = merged_mask.to(torch.uint8).cpu().numpy()
                        out[out > 0] = 255  # Binarize: 0 or 255
                        
                        # Save each frame as PNG
                        for frm_idx in range(out.shape[0]):
                            frame_name = f"{str(frm_idx).zfill(5)}.png"
                            mask_path = os.path.join(mask_dir, frame_name)
                            cv2.imwrite(mask_path, out[frm_idx])
                
                results['concat'] = {
                    'prompt': prompt,
                    'response': response,
                    'num_masks': len(masks),
                    'num_frames': masks[0][0].shape[0] if len(masks) > 0 else len(images),
                    'output_path': output_path,
                    'mask_dir': mask_dir
                }
            else:
                results['concat'] = {
                    'prompt': prompt,
                    'response': response,
                    'num_masks': 0,
                    'num_frames': 0,
                    'output_path': None,
                    'mask_dir': None
                }
        except Exception as e:
            print(f"Error processing concat ref expressions: {e}")
            results['concat'] = {
                'error': str(e)
            }
    
    return results


def main():
    args = parse_args()
    
    # Validate sampling parameters: only one can be set
    sampling_params = [args.every_n_frames, args.target_fps, args.frame_num]
    num_set = sum(x is not None for x in sampling_params)
    
    if num_set > 1:
        print("Error: Only one of --every_n_frames, --target_fps, or --frame_num can be set")
        return None
    
    # Default to target_fps=1 if none are set
    if num_set == 0:
        args.target_fps = 1.0
        print("Info: No sampling parameter set, defaulting to --target_fps 1.0")
    
    # Determine output directories based on ref_exp_file
    if args.output_dir is None:
        # Get the directory and filename of the ref_exp_file
        ref_exp_path = Path(args.ref_exp_file)
        ref_exp_dir = ref_exp_path.parent
        ref_exp_name = ref_exp_path.stem  # filename without extension
        
        # Create base output directory, with mode as subfolder
        base_dir = ref_exp_dir / ref_exp_name
        seperate_dir = base_dir / 'seperate'
        concat_dir = base_dir / 'concat'
    else:
        # Use user-specified output directory
        seperate_dir = Path(args.output_dir) / 'seperate'
        concat_dir = Path(args.output_dir) / 'concat'
    
    print("=" * 80)
    print(f"Loading ref-exp file: {args.ref_exp_file}")
    print(f"Loading CSV file: {args.csv_file}")
    print(f"Video directory: {args.video_dir}")
    if args.mode in ['seperate', 'both']:
        print(f"Seperate output directory: {seperate_dir}")
    if args.mode in ['concat', 'both']:
        print(f"Concat output directory: {concat_dir}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_path}")
    print("=" * 80)
    
    # Load ref-exp JSON
    with open(args.ref_exp_file, 'r') as f:
        ref_exp_data = json.load(f)
    
    # Load CSV
    df = pd.read_csv(args.csv_file)
    
    # Build model
    print("\nBuilding UniPixel model...")
    model, processor = build_model(args.model_path, device=args.device, dtype=args.dtype)
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    
    # Create output directories
    if args.mode in ['seperate', 'both']:
        nncore.mkdir(str(seperate_dir))
    if args.mode in ['concat', 'both']:
        nncore.mkdir(str(concat_dir))
    
    # Process each entry
    all_results = []
    items = list(ref_exp_data.items())
    
    # Apply chunking for multi-GPU processing
    if args.chunk > 1:
        items = [items[i::args.chunk] for i in range(args.chunk)][args.index]
        print(f"\nChunking: Processing chunk {args.index + 1}/{args.chunk} ({len(items)} items)")
    
    print(f"\nProcessing {len(items)} items...")
    
    for entry_id, entry_data in tqdm(items, desc="Processing videos"):
        # Get video path from CSV
        row = df[df['entry_id'] == entry_id]
        
        if row.empty:
            print(f"Warning: Entry {entry_id} not found in CSV")
            continue
        
        video_path = row.iloc[0]['video_path']
        
        # Get ref expressions (support both 'ref_expressions' and 'referring_expressions' keys)
        ref_expressions = None
        if 'parsed_response' in entry_data:
            parsed_response = entry_data['parsed_response']
            if parsed_response is not None and isinstance(parsed_response, dict):
                if 'ref_expressions' in parsed_response:
                    ref_expressions = parsed_response['ref_expressions']
                elif 'referring_expressions' in parsed_response:
                    ref_expressions = parsed_response['referring_expressions']
        
        if ref_expressions is None:
            print(f"Warning: No ref_expressions/referring_expressions found for {entry_id}")
            continue
        
        # Validate type
        if not isinstance(ref_expressions, list):
            print(f"Warning: ref_expressions/referring_expressions is not a list for {entry_id}, got {type(ref_expressions)}")
            continue
        
        if not ref_expressions:
            print(f"Warning: Empty ref_expressions/referring_expressions for {entry_id}")
            continue
        
        # Filter out non-string or empty ref expressions
        valid_ref_expressions = [ref for ref in ref_expressions if isinstance(ref, str) and ref.strip()]
        if not valid_ref_expressions:
            print(f"Warning: No valid string ref_expressions/referring_expressions for {entry_id}")
            continue
        
        if len(valid_ref_expressions) < len(ref_expressions):
            print(f"Warning: Filtered out {len(ref_expressions) - len(valid_ref_expressions)} invalid ref_expressions for {entry_id}")
        
        ref_expressions = valid_ref_expressions
        
        # Process video
        result = process_video(entry_id, video_path, ref_expressions, model, processor, 
                              device, args, str(seperate_dir), str(concat_dir))
        
        if result:
            all_results.append(result)
    
    # Save results summary based on mode
    summaries_saved = []
    
    # Add chunk suffix to filename if using multi-GPU
    chunk_suffix = f"_chunk_{args.index}" if args.chunk > 1 else ""
    
    if args.mode in ['seperate', 'both']:
        summary_path = seperate_dir / f'results_summary{chunk_suffix}.json'
        with open(str(summary_path), 'w') as f:
            json.dump(all_results, f, indent=2)
        summaries_saved.append(str(summary_path))
    
    if args.mode in ['concat', 'both']:
        summary_path = concat_dir / f'results_summary{chunk_suffix}.json'
        with open(str(summary_path), 'w') as f:
            json.dump(all_results, f, indent=2)
        summaries_saved.append(str(summary_path))
    
    print(f"\n{'=' * 80}")
    if args.chunk > 1:
        print(f"Processing complete for chunk {args.index + 1}/{args.chunk}!")
    else:
        print(f"Processing complete!")
    print(f"Processed {len(all_results)} videos")
    if args.mode in ['seperate', 'both']:
        print(f"Seperate results saved to: {seperate_dir}")
    if args.mode in ['concat', 'both']:
        print(f"Concat results saved to: {concat_dir}")
    for summary_path in summaries_saved:
        print(f"Summary saved to: {summary_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

