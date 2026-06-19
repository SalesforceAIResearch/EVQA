#!/usr/bin/env python3
# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

"""
Run inference on a single video with a question using UniPixel model.

Usage:
    # QA Mode (with question and optional options):
    python ours_st_evidence_single.py \
        --model PolyU-ChenLab/UniPixel-3B \
        --video /path/to/video.mp4 \
        --question "What is happening in the video?" \
        --options "Option A" "Option B" "Option C" "Option D"
    
    # Or without options (open-ended question):
    python ours_st_evidence_single.py \
        --video /path/to/video.mp4 \
        --question "Describe what happens in the video"
    
    # Custom Prompt Mode (for direct segmentation):
    python ours_st_evidence_single.py \
        --video /path/to/video.mp4 \
        --prompt "Please segment the door" \
        --save-masks --output-dir ./output
    
    # Use base UniPixel model (without adapter parameters):
    python ours_st_evidence_single.py \
        --model PolyU-ChenLab/UniPixel-3B \
        --base-model \
        --video /path/to/video.mp4 \
        --prompt "Please segment the person" \
        --save-masks --output-dir ./output
    
    # Use fine-tuned model (default, with adapter support):
    python ours_st_evidence_single.py \
        --model /path/to/finetuned_model \
        --video /path/to/video.mp4 \
        --prompt "Please segment the fire extinguisher" \
        --save-masks --output-dir ./output
"""

import argparse
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader
import decord

from unipixel.dataset.utils import process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_frames_with_stride
from unipixel.utils.transforms import get_sam2_transform


def format_question_with_options(question, options):
    """Format question with multiple choice options."""
    option_letters = ['A', 'B', 'C', 'D', 'E']
    question_text = question + '\nOptions:\n'
    for i, option in enumerate(options):
        question_text += f"({option_letters[i]}) {option}\n"
    return question_text.strip()


def create_st_evidence_prompt(question, options=None):
    """
    Create single-turn prompt for ST Evidence task.
    This matches the format used in UniPixel training.
    """
    if options:
        question_with_options = format_question_with_options(question, options)
        prompt = f"{question_with_options} Answer the question and provide evidence in the form of temporal ([[start1, end1], [start2, end2], ...]) and spatial evidence (masks)."
    else:
        prompt = f"{question} Provide your answer with temporal evidence ([[start1, end1], [start2, end2], ...]) and spatial evidence (masks) if applicable."
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


def save_masks_to_disk(masks, output_dir, name="masks"):
    """
    Save predicted masks to disk.
    
    Structure: output_dir / name / XXXXX.png
    """
    mask_dir = os.path.join(output_dir, name)
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


def save_masks_with_overlay(masks, frames, output_dir, original_fps=30.0, mask_color=(0, 255, 0), alpha=0.5):
    """
    Save masks overlaid on video frames, and save as MP4 video.
    
    Args:
        masks: numpy array of shape [num_frames, H, W] with values 0 or 255
        frames: torch tensor of shape [num_frames, H, W, C] (RGB)
        output_dir: directory to save outputs
        original_fps: FPS for the output video
        mask_color: BGR color for the mask overlay (default: green)
        alpha: transparency of the overlay (0-1)
    
    Returns:
        Paths to mask_dir, overlay_dir, and video_path
    """
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlays")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    num_frames = masks.shape[0]
    
    # Convert frames to numpy if tensor
    if torch.is_tensor(frames):
        frames_np = frames.cpu().numpy()
    else:
        frames_np = frames
    
    # Get frame dimensions for video writer
    frame_h, frame_w = frames_np[0].shape[:2]
    
    # Initialize video writer
    video_path = os.path.join(output_dir, "overlay_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, original_fps, (frame_w, frame_h))
    
    for i in range(num_frames):
        mask = masks[i]
        frame = frames_np[i]  # [H, W, C] RGB
        
        # Convert to uint8 (0-255)
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # Save mask
        frame_filename = f"{str(i).zfill(5)}.png"
        mask_path = os.path.join(mask_dir, frame_filename)
        cv2.imwrite(mask_path, mask_uint8)
        
        # Create overlay
        # Resize mask to frame size if needed
        mask_h, mask_w = mask.shape[:2]
        
        if mask_h != frame_h or mask_w != frame_w:
            mask_resized = cv2.resize(mask_uint8, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask_uint8
        
        # Convert frame from RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Create colored mask overlay
        overlay = frame_bgr.copy()
        mask_bool = mask_resized > 0
        overlay[mask_bool] = (
            (1 - alpha) * overlay[mask_bool] + 
            alpha * np.array(mask_color)
        ).astype(np.uint8)
        
        # Save overlay frame
        overlay_path = os.path.join(overlay_dir, frame_filename)
        cv2.imwrite(overlay_path, overlay)
        
        # Write frame to video
        video_writer.write(overlay)
    
    # Release video writer
    video_writer.release()
    
    return mask_dir, overlay_dir, video_path


def process_custom_prompt(video_path, prompt, model, processor, sam2_transform, device, args, mask_output_dir=None):
    """
    Process a video with a custom prompt for direct segmentation.
    
    This is for prompts like "locate the door", "segment the person", etc.
    The model will generate masks for the specified object/region.
    
    Args:
        video_path: Path to video file
        prompt: Custom prompt (e.g., "locate the door", "segment the person")
        model: UniPixel model
        processor: Model processor
        sam2_transform: SAM2 transform
        device: Device for inference
        args: Arguments with fps, max_frames, etc.
        mask_output_dir: Directory to save masks (optional)
    
    Returns dict with:
    - response: Raw model response
    - mask_path: Path to saved masks (if mask_output_dir is provided)
    - masks: Raw mask numpy array
    - inference_time: Time taken for model inference (seconds)
    """
    result = {'response': None, 'mask_path': None, 'overlay_path': None, 'video_path': None, 'masks': None, 'inference_time': None}
    
    try:
        # Load video - ALL frames for mask generation, sampled frames for LLM
        decord.bridge.set_bridge('torch')
        vr = VideoReader(video_path, num_threads=args.num_threads)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        # Calculate sample_frames
        if args.fps is not None:
            video_duration = total_frames / original_fps
            sample_frames = int(video_duration * args.fps)
            sample_frames = max(1, min(sample_frames, min(args.max_frames, total_frames)))
        else:
            sample_frames = min(args.max_frames, total_frames)
        
        # Load frames: uniformly spread across entire video
        frames, images, inds = load_frames_with_stride(
            video_path,
            every_n_frames=1,
            sample_frames=sample_frames,
            sample_type='uniform',
            sample_for_llm_only=True,
            num_threads=args.num_threads
        )
        
        num_frames = len(images)
        
        # Debug: Print frame counts
        print(f"\n📊 Frame counts:")
        print(f"   Original video: {total_frames} frames @ {original_fps:.2f} fps")
        print(f"   SAM2 input (frames): {frames.shape[0]} frames")
        print(f"   LLM input (images): {len(images)} frames (sampled)")
        
        # Use the prompt directly as provided by user
        full_prompt = prompt
        print(f"\n📝 Prompt: {full_prompt}")
        
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
                'text': full_prompt
            }]
        }]
        
        # Prepare input
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # DEBUG: Print what's actually being sent to the model
        print(f"\n🔍 DEBUG - Actual text sent to model:")
        print(f"   {text[:500]}...")
        images_proc, videos_proc, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = processor(text=[text], images=images_proc, videos=videos_proc, return_tensors='pt', **kwargs)
        
        # Add SAM2 frames
        data['frames'] = [sam2_transform(frames).to(model.sam2.dtype)]
        data['frame_size'] = [frames.shape[1:3]]
        
        # Generate
        print(f"\n⏱️  Starting model inference...")
        start_time = time.time()
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
        inference_time = time.time() - start_time
        print(f"✅ Model inference completed in {inference_time:.4f} seconds")
        result['inference_time'] = inference_time
        
        # Decode response
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        
        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        result['response'] = response
        
        # Extract and save masks
        if hasattr(model, 'seg') and len(model.seg) >= 1:
            # Get all object masks
            all_masks = [model.seg[obj_idx][0] for obj_idx in range(len(model.seg))]
            
            # Debug: Print mask info
            print(f"\n🎭 Mask info:")
            print(f"   Number of objects detected: {len(model.seg)}")
            for obj_idx, mask in enumerate(all_masks):
                print(f"   Object {obj_idx}: mask shape = {mask.shape}")
            
            combined_masks_tensor = torch.stack(all_masks, dim=0)
            merged_mask = torch.max(combined_masks_tensor, dim=0)[0]
            
            print(f"   Merged mask shape: {merged_mask.shape}")
            print(f"   SAM2 input frames: {frames.shape[0]}")
            
            # Convert to numpy and binarize
            out = merged_mask.to(torch.uint8).cpu().numpy()
            out[out > 0] = 255
            
            result['masks'] = out
            
            # Save masks with overlay if output directory is provided
            if mask_output_dir:
                mask_dir, overlay_dir, video_path = save_masks_with_overlay(
                    out, frames, mask_output_dir, original_fps=original_fps
                )
                result['mask_path'] = mask_dir
                result['overlay_path'] = overlay_dir
                result['video_path'] = video_path
        else:
            print("\n⚠️  No masks generated by the model")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return result


def process_single_video(video_path, question, options, model, processor, sam2_transform, device, args, mask_output_dir=None):
    """
    Process a single video item.
    
    This function matches the exact logic from ours_st_evidence.py:process_single_item
    
    Returns dict with:
    - response: Raw model response
    - answer: Predicted answer letter
    - segments: Temporal segments [[start, end], ...]
    - mask_path: Path to saved masks (if mask_output_dir is provided)
    - masks: Raw mask numpy array
    - inference_time: Time taken for model inference (seconds)
    """
    result = {'response': None, 'answer': None, 'segments': None, 'mask_path': None, 'overlay_path': None, 'video_path': None, 'masks': None, 'inference_time': None}
    
    try:
        # Load video - ALL frames for mask generation, sampled frames for LLM
        # (matching ours_st_evidence.py approach)
        decord.bridge.set_bridge('torch')
        vr = VideoReader(video_path, num_threads=args.num_threads)
        total_frames = len(vr)
        original_fps = vr.get_avg_fps()
        
        # Calculate sample_frames (matching ours_st_evidence.py)
        if args.fps is not None:
            video_duration = total_frames / original_fps
            sample_frames = int(video_duration * args.fps)
            sample_frames = max(1, min(sample_frames, min(args.max_frames, total_frames)))
        else:
            sample_frames = min(args.max_frames, total_frames)
        
        # Load frames: uniformly spread across entire video
        frames, images, inds = load_frames_with_stride(
            video_path,
            every_n_frames=1,  # Load all frames first, then uniformly sample
            sample_frames=sample_frames,
            sample_type='uniform',
            sample_for_llm_only=True,  # All frames for SAM2, sampled for LLM
            num_threads=args.num_threads
        )
        
        num_frames = len(images)
        
        # Debug: Print frame counts
        print(f"\n📊 Frame counts:")
        print(f"   Original video: {total_frames} frames @ {original_fps:.2f} fps")
        print(f"   SAM2 input (frames): {frames.shape[0]} frames")
        print(f"   LLM input (images): {len(images)} frames (sampled)")
        print(f"   Sampled indices: {inds[:5]}...{inds[-5:] if len(inds) > 5 else ''}")
        
        # Create prompt
        prompt = create_st_evidence_prompt(question, options)
        print(prompt)
        
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
        print(f"\n⏱️  Starting model inference...")
        start_time = time.time()
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
        inference_time = time.time() - start_time
        print(f"✅ Model inference completed in {inference_time:.4f} seconds")
        result['inference_time'] = inference_time
        
        # Decode response
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        
        response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        result['response'] = response
        
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
            
            # Debug: Print mask info
            print(f"\n🎭 Mask info:")
            print(f"   Number of objects detected: {len(model.seg)}")
            for obj_idx, mask in enumerate(all_masks):
                print(f"   Object {obj_idx}: mask shape = {mask.shape}")
            
            combined_masks_tensor = torch.stack(all_masks, dim=0)  # [num_objects, num_frames, H, W]
            merged_mask = torch.max(combined_masks_tensor, dim=0)[0]  # [num_frames, H, W]
            
            print(f"   Merged mask shape: {merged_mask.shape}")
            print(f"   SAM2 input frames: {frames.shape[0]}")
            
            # Convert to numpy and binarize
            out = merged_mask.to(torch.uint8).cpu().numpy()
            out[out > 0] = 255  # Binarize: 0 or 255
            
            result['masks'] = out
            
            # Save masks with overlay if output directory is provided
            if mask_output_dir:
                mask_dir, overlay_dir, video_path = save_masks_with_overlay(
                    out, frames, mask_output_dir, original_fps=original_fps
                )
                result['mask_path'] = mask_dir
                result['overlay_path'] = overlay_dir
                result['video_path'] = video_path
        
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return result


def main():
    parser = argparse.ArgumentParser(description='Run ST Evidence Task on a single video')
    
    # Required arguments
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    
    # Mode selection: either --question (QA mode) or --prompt (segmentation mode)
    parser.add_argument('--question', type=str, default=None,
                        help='Question to ask about the video (QA mode)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt for direct segmentation (e.g., "locate the door")')
    
    # Optional arguments for QA mode
    parser.add_argument('--options', nargs='+', default=None,
                        help='Answer options for multiple choice (e.g., --options "A" "B" "C" "D")')
    
    # Model settings
    parser.add_argument('--model', type=str,
                        default='PolyU-ChenLab/UniPixel-3B',
                        help='UniPixel model path or HuggingFace model ID')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for inference (auto, cuda, cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Data type for inference')
    parser.add_argument('--base-model', action='store_true',
                        help='Load as base UniPixel model (without adapter parameters). Use for original UniPixel models.')
    
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
    parser.add_argument('--save-masks', action='store_true',
                        help='Save masks to disk')
    parser.add_argument('--output-dir', type=str, default='./inference_output',
                        help='Output directory for masks')
    
    args = parser.parse_args()
    
    # Verify video exists
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    # Validate mode: must have either --question or --prompt
    if args.question is None and args.prompt is None:
        print("Error: Must provide either --question (QA mode) or --prompt (segmentation mode)")
        return
    
    if args.question is not None and args.prompt is not None:
        print("Error: Cannot use both --question and --prompt. Choose one mode.")
        return
    
    # Determine mode
    is_prompt_mode = args.prompt is not None
    
    print("=" * 80)
    if is_prompt_mode:
        print("SEGMENTATION MODE - Custom Prompt")
    else:
        print("QA MODE - ST Evidence Task")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Video: {args.video}")
    if is_prompt_mode:
        print(f"Prompt: {args.prompt}")
    else:
        print(f"Question: {args.question}")
        if args.options:
            print(f"Options: {args.options}")
    print(f"FPS: {args.fps}, Max frames: {args.max_frames}")
    print("=" * 80 + "\n")
    
    # Load model
    print(f"🔄 Loading model: {args.model}...")
    if args.base_model:
        # Load as base UniPixel model (without adapter parameters)
        model, processor = build_model(
            args.model,
            device=args.device,
            dtype=args.dtype
        )
    else:
        # Load with adapter support (for fine-tuned models)
        model, processor = build_model(
            args.model,
            device=args.device,
            dtype=args.dtype,
            is_trainable=False,
            merge_adapter=False
        )
    device = next(model.parameters()).device
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)
    print(f"✅ Model loaded on device: {device}\n")
    
    # Set up mask output directory
    mask_output_dir = None
    if args.save_masks:
        mask_output_dir = args.output_dir
        os.makedirs(mask_output_dir, exist_ok=True)
        print(f"🎭 Masks will be saved to: {mask_output_dir}\n")
    
    # Process video based on mode
    if is_prompt_mode:
        # Custom prompt mode (segmentation)
        result = process_custom_prompt(
            video_path=args.video,
            prompt=args.prompt,
            model=model,
            processor=processor,
            sam2_transform=sam2_transform,
            device=device,
            args=args,
            mask_output_dir=mask_output_dir
        )
    else:
        # QA mode
        result = process_single_video(
            video_path=args.video,
            question=args.question,
            options=args.options,
            model=model,
            processor=processor,
            sam2_transform=sam2_transform,
            device=device,
            args=args,
            mask_output_dir=mask_output_dir
        )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if result.get('inference_time') is not None:
        print(f"\n⏱️  Inference Time: {result['inference_time']:.4f} seconds")
    
    print(f"\n📝 Response:\n{result['response']}")
    
    # QA mode specific outputs
    if not is_prompt_mode:
        if result.get('answer'):
            answer_text = ""
            if args.options:
                option_idx = ord(result['answer']) - ord('A')
                if 0 <= option_idx < len(args.options):
                    answer_text = f" ({args.options[option_idx]})"
            print(f"\n🎯 Answer: {result['answer']}{answer_text}")
        
        if result.get('segments'):
            print(f"\n⏱️  Temporal Segments (frame indices):")
            for i, seg in enumerate(result['segments']):
                print(f"   [{i+1}] {seg[0]:.1f} - {seg[1]:.1f}")
    
    # Common outputs (masks)
    if result.get('masks') is not None:
        print(f"\n🎭 Masks: {result['masks'].shape[0]} frames, shape {result['masks'].shape[1:]}")
        if result.get('mask_path'):
            print(f"   Masks saved to: {result['mask_path']}")
        if result.get('overlay_path'):
            print(f"   Overlays saved to: {result['overlay_path']}")
        if result.get('video_path'):
            print(f"   🎬 Video saved to: {result['video_path']}")
    else:
        print("\n⚠️  No masks were generated")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
