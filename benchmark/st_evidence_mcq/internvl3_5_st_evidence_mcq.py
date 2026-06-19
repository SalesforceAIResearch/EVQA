#!/usr/bin/env python3
"""
Independent Task Video Question Answering for ST-Evidence MCQ using InternVL3.5

Three independent tasks (processed sequentially for each question):
- qa: Video QA - Given video, question, and options, model provides answer
- time_evidence: Time Evidence Selection - Given video, question, and 4 time segments, choose best one
- spatial_evidence: Spatial Evidence Selection - Given video, question, and 4 masked images, choose best one
- all: Run all three tasks sequentially (question by question)

Supported models:
- OpenGVLab/InternVL3_5-8B
- OpenGVLab/InternVL3_5-26B  
- OpenGVLab/InternVL3_5-78B

Usage:
    python internvl3_5_st_evidence_mcq.py --task qa --split val
    python internvl3_5_st_evidence_mcq.py --task all --split val
    python internvl3_5_st_evidence_mcq.py --task qa --split val --model OpenGVLab/InternVL3_5-26B

Output Format:
    - Individual tasks: JSON with answer and gt_answer for each entry_id
    - All tasks: Single unified JSON with keys per entry_id:
      * answer, gt_answer, evidence_t, gt_evidence_t, evidence_s, gt_evidence_s
"""

import os
import json
import pandas as pd
import ast
import argparse
import glob
import re
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, fps=1.0):
    """Load video and sample frames based on fps parameter."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    video_fps = float(vr.get_avg_fps())
    total_duration = len(vr) / video_fps
    num_segments = max(1, int(total_duration * fps))
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, video_fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def load_image(image_path, input_size=448, max_num=12):
    """Load and preprocess a single image."""
    img = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_video_mapping(video_dir: str) -> dict:
    """Build a mapping of video_id -> video_path for faster lookups."""
    print(f"🔍 Building video mapping from {video_dir}...")
    video_mapping = {}
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            if file.endswith('.mp4'):
                video_id = file.replace('.mp4', '')
                video_mapping[video_id] = os.path.join(video_dir, file)
    pattern = os.path.join(video_dir, '*', '*.mp4')
    for video_path in glob.glob(pattern):
        video_id = os.path.basename(video_path).replace('.mp4', '')
        video_mapping[video_id] = video_path
    print(f"✓ Found {len(video_mapping)} videos")
    return video_mapping


def find_video_path(video_id: str, video_mapping: dict) -> str:
    """Find video file path using prebuilt mapping."""
    return video_mapping.get(video_id, None)


def create_qa_prompt(question, options):
    """Create prompt for Video QA task."""
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except:
            options = [options]
    option_letters = ['A', 'B', 'C', 'D', 'E']
    formatted_options = []
    for i, option in enumerate(options):
        if i < len(option_letters):
            formatted_options.append(f"({option_letters[i]}) {option}")
    options_str = "\n".join(formatted_options)
    prompt = f"""I will provide you with a video and a multiple choice question about the video.

Question: {question}

Options:
{options_str}

Task: Please answer this question based on the video by selecting the correct option (A, B, C, D, or E).

Output format: Only provide the letter of your answer (A, B, C, D, or E), no additional text or explanation."""
    return prompt


def create_time_evidence_prompt(question, time_segments):
    """Create prompt for Time Evidence Selection task."""
    option_letters = ['A', 'B', 'C', 'D']
    formatted_options = []
    for i, segments in enumerate(time_segments):
        if i < len(option_letters):
            segments_str = ', '.join([f"[{seg[0]}s, {seg[1]}s]" for seg in segments])
            formatted_options.append(f"({option_letters[i]}) {segments_str}")
    options_str = "\n".join(formatted_options)
    prompt = f"""I have provided you with a video and a question about it. Below are four different time segment options (in seconds) from the video.

Question: {question}

Time Segment Options:
{options_str}

Task: Please select the time segment that best serves as evidence to answer the question, which should be essential and significant to understanding and answering the question.

Output format: Only provide the letter of your choice (A, B, C, or D), no additional text or explanation."""
    return prompt


def create_spatial_evidence_prompt(question):
    """Create prompt for Spatial Evidence Selection task."""
    prompt = f"""I have provided you with a video and a question about it. Below, I will show you 4 images extracted from the video. Each image has a red mask boundary overlaid on it, highlighting a different object or region.

Question: {question}

Task: Please select the image where the highlighted region best serves as spatial evidence to answer this question. The highlighted object or region should be directly relevant to understanding the question or answering the question.

Note: The images are presented in order as options A, B, C, and D (first image = A, second = B, third = C, fourth = D).

Output format: Only provide the letter of your choice (A, B, C, or D), no additional text or explanation."""
    return prompt


def parse_answer(response_text: str):
    """Parse answer letter from model response."""
    response_text = response_text.strip()
    answer_match = re.search(r'\b([A-E])\b', response_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    if len(response_text) == 1 and response_text.upper() in ['A', 'B', 'C', 'D', 'E']:
        return response_text.upper()
    return None


def get_answer_letter(answer_text: str, options: list):
    """Convert answer text to letter format based on position in options."""
    answer_text_lower = answer_text.strip().lower()
    for i, option in enumerate(options):
        if option.strip().lower() == answer_text_lower:
            return chr(65 + i)
    return None


def process_qa_task(model, tokenizer, generation_config, video_path: str, question: str, 
                     options: list, entry_id: str, fps: float, max_num: int, input_size: int):
    """Process Video QA task."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    try:
        pixel_values, num_patches_list = load_video(video_path, bound=None, input_size=input_size, max_num=max_num, fps=fps)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        prompt = video_prefix + create_qa_prompt(question, options)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=False)
        answer = parse_answer(response)
        print(f"✅ Response for {entry_id}: {response}")
        return {'answer': answer}, None
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_time_evidence_task(model, tokenizer, generation_config, video_path: str, question: str,
                                 time_segments: list, entry_id: str, fps: float, max_num: int, input_size: int):
    """Process Time Evidence Selection task."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    try:
        pixel_values, num_patches_list = load_video(video_path, bound=None, input_size=input_size, max_num=max_num, fps=fps)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        prompt = video_prefix + create_time_evidence_prompt(question, time_segments)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config, num_patches_list=num_patches_list, history=None, return_history=False)
        answer = parse_answer(response)
        print(f"✅ Response for {entry_id}: {response}")
        return {'answer': answer}, None
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_spatial_evidence_task(model, tokenizer, generation_config, video_path: str, 
                                   mask_images: list, question: str, entry_id: str, fps: float, max_num: int, input_size: int):
    """Process Spatial Evidence Selection task with video and images."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    try:
        # Load video
        video_pixel_values, video_num_patches = load_video(video_path, bound=None, input_size=input_size, max_num=max_num, fps=fps)
        
        # Load masked images
        image_pixel_values_list = []
        image_num_patches_list = []
        for img_path in mask_images:
            if not os.path.exists(img_path):
                print(f"⚠️  Warning: Masked image not found: {img_path}")
                continue
            try:
                img_pixel_values = load_image(img_path, input_size=input_size, max_num=max_num)
                image_pixel_values_list.append(img_pixel_values)
                image_num_patches_list.append(img_pixel_values.shape[0])
            except Exception as e:
                print(f"⚠️  Warning: Failed to load image {img_path}: {e}")
        
        if len(image_pixel_values_list) == 0:
            print(f"✗ No valid masked images found")
            return None, "No valid masked images found"
        
        # Combine video and images: video first, then images
        all_pixel_values = [video_pixel_values] + image_pixel_values_list
        all_pixel_values = torch.cat(all_pixel_values, dim=0).to(torch.bfloat16).cuda()
        all_num_patches = video_num_patches + image_num_patches_list
        
        # Create prefix for video frames and images
        video_image_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(all_num_patches))])
        prompt = video_image_prefix + create_spatial_evidence_prompt(question)
        response = model.chat(tokenizer, all_pixel_values, prompt, generation_config, num_patches_list=all_num_patches, history=None, return_history=False)
        answer = parse_answer(response)
        print(f"✅ Response for {entry_id}: {response}")
        return {'answer': answer}, None
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Independent Task Video Question Answering with InternVL3.5")
    parser.add_argument('--task', type=str, required=True, choices=['qa', 'time_evidence', 'spatial_evidence', 'all'],
                        help='Task type')
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3_5-8B', help='InternVL3.5 model to use')
    parser.add_argument('--data_file', type=str, default='data/st_evidence_mcq.csv', help='CSV file with video data')
    parser.add_argument('--distractors_file', type=str, default='data/temp_options.json', 
                        help='JSON file with time segment distractors')
    parser.add_argument('--mask_file', type=str, default='data/mask_options.json',
                        help='JSON file with masked image paths')
    parser.add_argument('--save_file', type=str, default=None, help='Save file')
    parser.add_argument('--video_dir', type=str, default='data/NextQA-Video', help='Video directory')
    parser.add_argument('--fps', type=float, default=1.0, help='Frame rate')
    parser.add_argument('--max_num', type=int, default=1, help='Maximum number of patches per frame')
    parser.add_argument('--input_size', type=int, default=448, help='Input size for image preprocessing')
    args = parser.parse_args()    
    # Initialize model
    print(f"\n🔄 Loading {args.model}...")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=2048, do_sample=False, temperature=0.0)
    print(f"✓ Model loaded successfully")
    
    # Load data
    print(f"📊 Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    test_df = df
    
    # Determine tasks
    tasks_to_run = ['qa', 'time_evidence', 'spatial_evidence'] if args.task == 'all' else [args.task]
    unified_results = args.task == 'all'
    
    # Load additional data
    distractors_dict, mask_dict = {}, {}
    if 'time_evidence' in tasks_to_run:
        distractors_file = args.distractors_file
        if os.path.exists(distractors_file):
            with open(distractors_file, 'r') as f:
                distractors_dict = json.load(f)
            print(f"✓ Loaded {len(distractors_dict)} distractor entries")
    
    if 'spatial_evidence' in tasks_to_run:
        if os.path.exists(args.mask_file):
            with open(args.mask_file, 'r') as f:
                mask_dict = json.load(f)
            print(f"✓ Loaded {len(mask_dict)} mask entries")
    
    # Build video mapping
    video_mapping = build_video_mapping(args.video_dir)
    
    # Save file
    results_base = 'result'
    if args.save_file:
        save_file = args.save_file
    else:
        model_name = args.model.split('/')[-1]
        task_name = 'all' if unified_results else args.task
        save_file = f"{results_base}/internvl/{model_name}_{task_name}_{args.fps}fps.json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # Load existing results
    result_dict = {}
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            result_dict = json.load(f)
        print(f"📂 Loaded {len(result_dict)} existing entries")
        if unified_results:
            qa_valid = sum(1 for e in result_dict.values() if e.get('answer') is not None)
            time_valid = sum(1 for e in result_dict.values() if e.get('evidence_t') is not None)
            spatial_valid = sum(1 for e in result_dict.values() if e.get('evidence_s') is not None)
            print(f"📊 Current: QA={qa_valid}, Time={time_valid}, Spatial={spatial_valid}")
    
    # Process entries
    pbar = tqdm(total=len(test_df) * len(tasks_to_run), desc="Processing", unit="task")
    
    for index, row in test_df.iterrows():
        if 'entry_id' not in row or pd.isna(row['entry_id']):
            continue
        entry_id = str(row['entry_id'])
        
        for task in tasks_to_run:
            try:
                # Check if done
                if unified_results:
                    task_key = 'answer' if task == 'qa' else ('evidence_t' if task == 'time_evidence' else 'evidence_s')
                    if entry_id in result_dict and result_dict[entry_id].get(task_key) is not None:
                        pbar.update(1)
                        continue
                else:
                    if entry_id in result_dict and result_dict[entry_id].get('answer') is not None:
                        pbar.update(1)
                        continue
                
                # Get video
                if 'video' not in row or pd.isna(row['video']):
                    pbar.update(1)
                    continue
                video_id = str(row['video'])
                video_path = find_video_path(video_id, video_mapping)
                if not video_path:
                    pbar.update(1)
                    continue
                
                # Process task
                if task == 'qa':
                    question = row.get('question', '')
                    options_str = row.get('options', '')
                    options = ast.literal_eval(options_str) if isinstance(options_str, str) else options_str
                    gt_answer_text = row.get('answer', '')
                    gt_answer = get_answer_letter(gt_answer_text, options) if gt_answer_text else ''
                    
                    print(f"\n📹 [QA] Processing {entry_id}: {video_id}.mp4")
                    result, error = process_qa_task(model, tokenizer, generation_config, video_path, question, options, entry_id, args.fps, args.max_num, args.input_size)
                    
                    if unified_results:
                        if entry_id not in result_dict:
                            result_dict[entry_id] = {}
                        result_dict[entry_id]['answer'] = result['answer'] if result else None
                        result_dict[entry_id]['gt_answer'] = gt_answer
                    else:
                        result_dict[entry_id] = {
                            'answer': result['answer'] if result else None,
                            'gt_answer': gt_answer
                        }
                
                elif task == 'time_evidence':
                    if entry_id not in distractors_dict:
                        pbar.update(1)
                        continue
                    distractor_data = distractors_dict[entry_id]
                    question = distractor_data.get('question', row.get('question', ''))
                    time_segments = distractor_data.get('time_evidence_options', [])
                    if not time_segments or len(time_segments) < 4:
                        pbar.update(1)
                        continue
                    gt_time_evidence_idx = distractor_data.get('correct_answer', '')
                    
                    print(f"\n📹 [TIME] Processing {entry_id}: {video_id}.mp4")
                    result, error = process_time_evidence_task(model, tokenizer, generation_config, video_path, question, time_segments, entry_id, args.fps, args.max_num, args.input_size)
                    
                    if unified_results:
                        if entry_id not in result_dict:
                            result_dict[entry_id] = {}
                        result_dict[entry_id]['evidence_t'] = result['answer'] if result else None
                        result_dict[entry_id]['gt_evidence_t'] = gt_time_evidence_idx
                    else:
                        result_dict[entry_id] = {
                            'answer': result['answer'] if result else None,
                            'gt_answer': gt_time_evidence_idx
                        }
                
                elif task == 'spatial_evidence':
                    mask_data = None
                    if entry_id in mask_dict:
                        mask_data = mask_dict[entry_id]
                    else:
                        for key, data in mask_dict.items():
                            if data.get('entry_id') == entry_id:
                                mask_data = data
                                break
                    if not mask_data:
                        pbar.update(1)
                        continue
                    
                    question = mask_data.get('question', row.get('question', ''))
                    if 'options' in mask_data and isinstance(mask_data['options'], list):
                        image_paths_dict = {
                            'gt': mask_data.get('gt', ''),
                            'd1': mask_data.get('d1', ''),
                            'd2': mask_data.get('d2', ''),
                            'd3': mask_data.get('d3', '')
                        }
                        mask_images = [image_paths_dict.get(opt_key, '') for opt_key in mask_data['options']]
                    else:
                        mask_images = mask_data.get('mcq_options', [])
                    if not mask_images or len(mask_images) < 4:
                        pbar.update(1)
                        continue
                    gt_index = mask_data.get('answer', mask_data.get('correct_answer', 0))
                    gt_spatial_evidence_idx = chr(65 + gt_index)
                    
                    print(f"\n🖼️  [SPATIAL] Processing {entry_id}: {video_id}.mp4")
                    result, error = process_spatial_evidence_task(model, tokenizer, generation_config, video_path, mask_images, question, entry_id, args.fps, args.max_num, args.input_size)
                    
                    if unified_results:
                        if entry_id not in result_dict:
                            result_dict[entry_id] = {}
                        result_dict[entry_id]['evidence_s'] = result['answer'] if result else None
                        result_dict[entry_id]['gt_evidence_s'] = gt_spatial_evidence_idx
                    else:
                        result_dict[entry_id] = {
                            'answer': result['answer'] if result else None,
                            'gt_answer': gt_spatial_evidence_idx
                        }
                
                # Save after each task
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                pbar.update(1)
                
            except Exception as e:
                print(f"❌ Error processing {entry_id} [{task}]: {e}")
                import traceback
                traceback.print_exc()
                pbar.update(1)
                continue
    
    pbar.close()
    print(f"\n💾 Results saved to: {save_file}")


if __name__ == "__main__":
    main()
