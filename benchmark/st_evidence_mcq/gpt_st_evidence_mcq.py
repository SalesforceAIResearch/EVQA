#!/usr/bin/env python3
"""
Independent Task Video Question Answering for ST-Evidence MCQ using OpenAI Models

Three independent tasks (processed sequentially for each question):
- qa: Video QA - Given video, question, and options, model provides answer
- time_evidence: Time Evidence Selection - Given video, question, and 4 time segments, choose best one
- spatial_evidence: Spatial Evidence Selection - Given video, question, and 4 masked images, choose best one
- all: Run all three tasks sequentially (question by question)

Supported models: gpt-4o, gpt-5, gpt-5-mini, o3

Usage:
    # Run video QA task
    python gpt_st_evidence_mcq.py --task qa --split val --model gpt-4o
    
    # Run time evidence selection task
    python gpt_st_evidence_mcq.py --task time_evidence --split val --model gpt-4o
    
    # Run spatial evidence selection task
    python gpt_st_evidence_mcq.py --task spatial_evidence --split val --model gpt-4o
    
    # Run all tasks sequentially (saves to single unified file)
    python gpt_st_evidence_mcq.py --task all --split val --model gpt-4o
    # This creates: result/gpt/{model}_all_{split}_{fps}fps.json
    # with format: {"entry_id": {"answer": "A", "evidence_t": "B", "evidence_s": "C", ...}}
    
    # Custom model and FPS
    python gpt_st_evidence_mcq.py --task qa --split train --model gpt-5 --fps 2

Output Format:
    - Individual tasks: JSON with answer and gt_answer for each entry_id
    - All tasks: Single unified JSON with keys per entry_id:
      * answer: QA answer (A/B/C/D/E) or null if error
      * gt_answer: Ground truth QA answer
      * evidence_t: Time evidence answer (A/B/C/D) or null if error
      * gt_evidence_t: Ground truth time evidence answer
      * evidence_s: Spatial evidence answer (A/B/C/D) or null if error
      * gt_evidence_s: Ground truth spatial evidence answer
"""

import os
import json
import pandas as pd
import ast
import argparse
import re
import base64
import time
import glob
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import openai
from decord import VideoReader, cpu
from PIL import Image
import io


# OpenAI API configuration (using Salesforce gateway)
openai.api_key = "dummy"
openai.default_headers = {"X-Api-Key": os.environ.get("SALESFORCE_API_KEY", "")}
openai.base_url = "https://gateway.salesforceresearch.ai/openai/process/v1/"


def encode_image_to_base64(image):
    """Encode PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_video_frames(video_path, fps=1.0, max_frames=32, target_size=(512, 512)):
    """
    Load video and sample frames based on fps parameter.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to sample (default: 1.0)
        max_frames: Maximum number of frames to extract
        target_size: Target size for resizing frames (default: 512x512)
    
    Returns:
        List of PIL Images resized to target_size
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    video_fps = float(vr.get_avg_fps())
    
    # Calculate total video duration
    total_duration = len(vr) / video_fps
    
    # Calculate number of frames to sample based on fps
    num_frames = min(max_frames, max(1, int(total_duration * fps)))
    
    # Calculate frame indices
    frame_indices = []
    for i in range(num_frames):
        frame_idx = int((i * len(vr)) / num_frames)
        frame_indices.append(min(frame_idx, max_frame))
    
    # Extract and resize frames
    frames = []
    for frame_idx in frame_indices:
        img = Image.fromarray(vr[frame_idx].asnumpy()).convert('RGB')
        # Resize to target size to reduce API costs
        img = img.resize(target_size, Image.LANCZOS)
        frames.append(img)
    
    return frames


def load_image(image_path, target_size=(512, 512)):
    """Load and resize a single image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    return img


def create_image_content_list(frames):
    """Create OpenAI API content list from frames"""
    content_list = []
    for frame in frames:
        base64_image = encode_image_to_base64(frame)
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    return content_list


def call_openai_model(messages, model="gpt-4o", temperature=0.0, max_tokens=2048):
    """Call OpenAI model API with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # GPT-5 and o3 use max_completion_tokens and don't support custom temperature
            if model in ['gpt-5', 'o3']:
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    # GPT-5/o3 only support default temperature=1, so we omit it
                    max_completion_tokens=max_tokens,
                )
            else:
                response = openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise


def build_video_mapping(video_dir: str) -> dict:
    """Build a mapping of video_id -> video_path for faster lookups."""
    print(f"🔍 Building video mapping from {video_dir}...")
    video_mapping = {}
    
    # First check for direct videos in video_dir
    if os.path.exists(video_dir):
        for file in os.listdir(video_dir):
            if file.endswith('.mp4'):
                video_id = file.replace('.mp4', '')
                video_mapping[video_id] = os.path.join(video_dir, file)
    
    # Then search in subfolders
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
    # Parse options if it's a string
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except:
            options = [options]
    
    # Format options dynamically
    option_letters = ['A', 'B', 'C', 'D', 'E']
    formatted_options = []
    
    for i, option in enumerate(options):
        if i < len(option_letters):
            formatted_options.append(f"({option_letters[i]}) {option}")
    
    options_str = "\n".join(formatted_options)
    
    prompt = f"""I will provide you with a video (as frames) and a multiple choice question about the video.

Question: {question}

Options:
{options_str}

Task: Please answer this question based on the video by selecting the correct option (A, B, C, D, or E).

Output format: Only provide the letter of your answer (A, B, C, D, or E), no additional text or explanation."""
    
    return prompt


def create_time_evidence_prompt(question, time_segments):
    """Create prompt for Time Evidence Selection task.
    time_segments should be a list of 4 time segment options, each being a list of [start, end] pairs.
    """
    # Format time segments as options
    option_letters = ['A', 'B', 'C', 'D']
    formatted_options = []
    
    for i, segments in enumerate(time_segments):
        if i < len(option_letters):
            # Format segments list as string
            segments_str = ', '.join([f"[{seg[0]}s, {seg[1]}s]" for seg in segments])
            formatted_options.append(f"({option_letters[i]}) {segments_str}")
    
    options_str = "\n".join(formatted_options)
    
    prompt = f"""I have provided you with a video (as frames) and a question about it. Below are four different time segment options (in seconds) from the video.

Question: {question}

Time Segment Options:
{options_str}

Task: Please select the time segment that best serves as evidence to answer the question, which should be essential and significant to understanding and answering the question.

Output format: Only provide the letter of your choice (A, B, C, or D), no additional text or explanation."""
    
    return prompt


def create_spatial_evidence_prompt(question):
    """Create prompt for Spatial Evidence Selection task.
    Note: The 4 masked images are provided as separate images after video frames and this prompt.
    """
    prompt = f"""I have provided you with a video (as frames) and a question about it. Below, I will show you 4 images extracted from the video. Each image has a red mask boundary overlaid on it, highlighting a different object or region.

Question: {question}

Task: Please select the image where the highlighted region best serves as spatial evidence to answer this question. The highlighted object or region should be directly relevant to understanding the question or answering the question.

Note: The images are presented in order as options A, B, C, and D (first image = A, second = B, third = C, fourth = D).

Output format: Only provide the letter of your choice (A, B, C, or D), no additional text or explanation."""
    
    return prompt


def parse_answer(response_text: str):
    """Parse answer letter from model response."""
    # Clean up response text
    response_text = response_text.strip()
    
    # Try to extract single letter answer (A, B, C, D, E)
    answer_match = re.search(r'\b([A-E])\b', response_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # If just a single letter
    if len(response_text) == 1 and response_text.upper() in ['A', 'B', 'C', 'D', 'E']:
        return response_text.upper()
    
    return None


def get_answer_letter(answer_text: str, options: list):
    """Convert answer text to letter format (A, B, C, D, etc.) based on position in options."""
    answer_text_lower = answer_text.strip().lower()
    for i, option in enumerate(options):
        if option.strip().lower() == answer_text_lower:
            return chr(65 + i)  # 65 is ASCII for 'A'
    return None


def process_qa_task(model: str, video_path: str, question: str, 
                     options: list, entry_id: str, fps: float, max_frames: int):
    """Process Video QA task: Given video, question, and options, return answer."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Load video frames
        frames = load_video_frames(video_path, fps=fps, max_frames=max_frames)
        image_content = create_image_content_list(frames)
        
        # Create prompt
        prompt = create_qa_prompt(question, options)
        
        # Create messages
        messages = [{
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt}]
        }]
        
        # Generate response
        response_text = call_openai_model(messages, model=model)
        
        answer = parse_answer(response_text)
        print(f"✅ Response for {entry_id}: {response_text}")
        
        return {'answer': answer, 'raw_response': response_text}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_time_evidence_task(model: str, video_path: str, question: str,
                                 time_segments: list, entry_id: str, fps: float, max_frames: int):
    """Process Time Evidence Selection task: Given video, question, and 4 time segment options, choose best one."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Load video frames
        frames = load_video_frames(video_path, fps=fps, max_frames=max_frames)
        image_content = create_image_content_list(frames)
        
        # Create prompt
        prompt = create_time_evidence_prompt(question, time_segments)
        
        # Create messages
        messages = [{
            "role": "user",
            "content": image_content + [{"type": "text", "text": prompt}]
        }]
        
        # Generate response
        response_text = call_openai_model(messages, model=model)
        
        answer = parse_answer(response_text)
        print(f"✅ Response for {entry_id}: {response_text}")
        
        return {'answer': answer, 'raw_response': response_text}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_spatial_evidence_task(model: str, video_path: str, 
                                   mask_images: list, question: str, entry_id: str, fps: float, max_frames: int):
    """Process Spatial Evidence Selection task: Given video, 4 masked images and question, choose best one."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Load video frames
        frames = load_video_frames(video_path, fps=fps, max_frames=max_frames)
        video_content = create_image_content_list(frames)
        
        # Create prompt
        prompt = create_spatial_evidence_prompt(question)
        
        # Load masked images
        image_content = []
        for img_path in mask_images:
            if not os.path.exists(img_path):
                print(f"⚠️  Warning: Masked image not found: {img_path}")
                continue
            try:
                img = load_image(img_path)
                base64_image = encode_image_to_base64(img)
                image_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"⚠️  Warning: Failed to load image {img_path}: {e}")
        
        if len(image_content) == 0:
            print(f"✗ No valid masked images found")
            return None, "No valid masked images found"
        
        # Create messages: video frames, then prompt, then masked images
        messages = [{
            "role": "user",
            "content": video_content + [{"type": "text", "text": prompt}] + image_content
        }]
        
        # Generate response
        response_text = call_openai_model(messages, model=model)
        
        answer = parse_answer(response_text)
        print(f"✅ Response for {entry_id}: {response_text}")
        
        return {'answer': answer, 'raw_response': response_text}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Independent Task Video Question Answering with OpenAI Models")
    parser.add_argument('--task', type=str, required=True, choices=['qa', 'time_evidence', 'spatial_evidence', 'all'],
                        help='Task type: qa (video QA), time_evidence (time segment selection), spatial_evidence (masked image selection), all (run all tasks)')
    parser.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model to use (gpt-4o, gpt-5, gpt-5-mini, o3)')
    parser.add_argument('--data_file', type=str, default='data/st_evidence_mcq.csv', help='CSV file with video data')
    parser.add_argument('--distractors_file', type=str, default='data/temp_options.json', 
                        help='JSON file with time segment distractors (use {split} placeholder for split name, needed for time_evidence task)')
    parser.add_argument('--mask_file', type=str, default='data/mask_options.json',
                        help='JSON file with masked image paths (needed for spatial_evidence task)')
    parser.add_argument('--save_file', type=str, default=None, help='Save file (default: result/gpt/{model}_{task}_{split}_{fps}fps.json)')
    parser.add_argument('--video_dir', type=str, default='data/NextQA-Video', help='Video directory')
    parser.add_argument('--fps', type=int, default=1, help='Frame rate')
    parser.add_argument('--max_frames', type=int, default=32, help='Maximum number of frames to extract from video')
    args = parser.parse_args()    
    print(f"✓ Using OpenAI model: {args.model}")
    
    # Load CSV data
    print(f"📊 Loading data from {args.data_file}...")
    test_df = pd.read_csv(args.data_file)
    print(f"✓ Using all {len(test_df)} entries")
    
    # Determine tasks to run
    if args.task == 'all':
        tasks_to_run = ['qa', 'time_evidence', 'spatial_evidence']
        unified_results = True
    else:
        tasks_to_run = [args.task]
        unified_results = False
    
    # Load additional data files
    distractors_dict = {}
    mask_dict = {}
    
    if 'time_evidence' in tasks_to_run:
        distractors_file = args.distractors_file
        if os.path.exists(distractors_file):
            print(f"📋 Loading time segment distractors from {distractors_file}...")
            with open(distractors_file, 'r') as f:
                distractors_dict = json.load(f)
            print(f"✓ Loaded {len(distractors_dict)} distractor entries")
        else:
            print(f"⚠️  Warning: Distractors file not found: {distractors_file}")
            if args.task == 'time_evidence':
                print("Cannot proceed with time_evidence task without distractors file")
                return
    
    if 'spatial_evidence' in tasks_to_run:
        if os.path.exists(args.mask_file):
            print(f"🖼️  Loading mask data from {args.mask_file}...")
            with open(args.mask_file, 'r') as f:
                mask_dict = json.load(f)
            print(f"✓ Loaded {len(mask_dict)} mask entries")
        else:
            print(f"⚠️  Warning: Mask file not found: {args.mask_file}")
            if args.task == 'spatial_evidence':
                print("Cannot proceed with spatial_evidence task without mask file")
                return
    
    # Build video mapping
    video_mapping = build_video_mapping(args.video_dir)
    
    # Determine save file path
    results_base = 'result'
    if args.save_file:
        save_file = args.save_file
    else:
        task_name = 'all' if unified_results else args.task
        save_file = f"{results_base}/gpt/{args.model}_{task_name}_{args.fps}fps.json"
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # Load existing results if file exists
    result_dict = {}
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            result_dict = json.load(f)
        print(f"📂 Loaded existing results from {save_file}")
        print(f"✓ Found {len(result_dict)} existing entries")
        
        if unified_results:
            qa_valid = sum(1 for entry in result_dict.values() if entry.get('answer') is not None)
            time_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_t') is not None)
            spatial_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_s') is not None)
            print(f"📊 Current status: QA={qa_valid}, Time={time_valid}, Spatial={spatial_valid}")
    
    # Process entries: outer loop over questions, inner loop over tasks
    processed_count = 0
    
    # Create progress bar
    pbar = tqdm(total=len(test_df) * len(tasks_to_run), desc="Processing", unit="task")
    
    for index, row in test_df.iterrows():
        # Get entry_id
        if 'entry_id' not in row or pd.isna(row['entry_id']):
            print(f"⚠️  Skipping row {index}: missing entry_id")
            continue
            
        entry_id = str(row['entry_id'])
        
        # Process all tasks for this entry
        for task in tasks_to_run:
            try:
                # Check if already done
                if unified_results:
                    task_key = 'answer' if task == 'qa' else ('evidence_t' if task == 'time_evidence' else 'evidence_s')
                    if entry_id in result_dict and result_dict[entry_id].get(task_key) is not None:
                        pbar.update(1)
                        continue
                else:
                    if entry_id in result_dict and result_dict[entry_id].get('answer') is not None:
                        pbar.update(1)
                        continue
                
                # Get video info
                if 'video' not in row or pd.isna(row['video']):
                    pbar.update(1)
                    continue
                
                video_id = str(row['video'])
                video_path = find_video_path(video_id, video_mapping)
                
                if not video_path:
                    pbar.update(1)
                    continue
                
                # Process based on task type
                if task == 'qa':
                    question = row.get('question', '')
                    options_str = row.get('options', '')
                    options = ast.literal_eval(options_str) if isinstance(options_str, str) else options_str
                    
                    gt_answer_text = row.get('answer', '')
                    gt_answer = get_answer_letter(gt_answer_text, options) if gt_answer_text else ''
                    
                    print(f"\n📹 [QA] Processing {entry_id}: {video_id}.mp4")
                    
                    result, error = process_qa_task(args.model, video_path, question, options, entry_id, args.fps, args.max_frames)
                    
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
                    gt_time_segments = distractor_data.get('ground_truth_segments', [])
                    
                    print(f"\n📹 [TIME] Processing {entry_id}: {video_id}.mp4")
                    
                    result, error = process_time_evidence_task(args.model, video_path, question, time_segments, entry_id, args.fps, args.max_frames)
                    
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
                    
                    result, error = process_spatial_evidence_task(args.model, video_path, mask_images, question, entry_id, args.fps, args.max_frames)
                    
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
                
                processed_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"❌ Error processing {entry_id} [{task}]: {e}")
                import traceback
                traceback.print_exc()
                pbar.update(1)
                continue
    
    pbar.close()
    
    # Final statistics
    print("\n" + "="*80)
    print("🎉 PROCESSING COMPLETED!")
    print("="*80)
    
    if unified_results:
        # Unified results statistics
        qa_valid = sum(1 for entry in result_dict.values() if entry.get('answer') is not None)
        qa_correct = sum(1 for entry in result_dict.values() if entry.get('answer') == entry.get('gt_answer'))
        
        time_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_t') is not None)
        time_correct = sum(1 for entry in result_dict.values() if entry.get('evidence_t') == entry.get('gt_evidence_t'))
        
        spatial_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_s') is not None)
        spatial_correct = sum(1 for entry in result_dict.values() if entry.get('evidence_s') == entry.get('gt_evidence_s'))
        
        print(f"\n📊 QA Task:")
        print(f"  Valid responses: {qa_valid}/{len(test_df)}")
        if qa_valid > 0:
            print(f"  Accuracy: {qa_correct}/{qa_valid} = {qa_correct/qa_valid*100:.2f}%")
        
        print(f"\n📊 Time Evidence Task:")
        print(f"  Valid responses: {time_valid}/{len(test_df)}")
        if time_valid > 0:
            print(f"  Accuracy: {time_correct}/{time_valid} = {time_correct/time_valid*100:.2f}%")
        
        print(f"\n📊 Spatial Evidence Task:")
        print(f"  Valid responses: {spatial_valid}/{len(test_df)}")
        if spatial_valid > 0:
            print(f"  Accuracy: {spatial_correct}/{spatial_valid} = {spatial_correct/spatial_valid*100:.2f}%")
    else:
        # Single task statistics
        valid_responses = sum(1 for entry in result_dict.values() if entry.get('answer') is not None)
        correct_responses = sum(1 for entry in result_dict.values() if entry.get('answer') == entry.get('gt_answer'))
        
        print(f"\n📊 {args.task.upper()} Task:")
        print(f"  Total entries: {len(test_df)}")
        print(f"  Valid responses: {valid_responses}")
        if valid_responses > 0:
            print(f"  Accuracy: {correct_responses}/{valid_responses} = {correct_responses/valid_responses*100:.2f}%")
    
    print(f"\n💾 Results saved to: {save_file}")
    print("="*80)


if __name__ == "__main__":
    main()

