#!/usr/bin/env python3
"""
Independent Task Video Question Answering for ST-Evidence MCQ using Qwen3VL

Three independent tasks (processed sequentially for each question):
- qa: Video QA - Given video, question, and options, model provides answer
- time_evidence: Time Evidence Selection - Given video, question, and 4 time segments, choose best one
- spatial_evidence: Spatial Evidence Selection - Given video, question, and 4 masked images, choose best one
- all: Run all three tasks sequentially (question by question)

Usage:
    # Run video QA task
    python qwen3vl_st_evidence_mcq.py --task qa --split val --model Qwen/Qwen3-VL-8B-Instruct
    
    # Run time evidence selection task
    python qwen3vl_st_evidence_mcq.py --task time_evidence --split val --model Qwen/Qwen3-VL-8B-Instruct
    
    # Run spatial evidence selection task
    python qwen3vl_st_evidence_mcq.py --task spatial_evidence --split val --model Qwen/Qwen3-VL-8B-Instruct
    
    # Run all tasks sequentially (saves to single unified file)
    python qwen3vl_st_evidence_mcq.py --task all --split val --model Qwen/Qwen3-VL-8B-Instruct
    # This creates: result/qwen/{model_name}_all_{split}_{fps}fps.json
    # with format: {"entry_id": {"answer": "A", "evidence_t": "B", "evidence_s": "C", ...}}
    
    # Custom model and FPS
    python qwen3vl_st_evidence_mcq.py --task qa --split train --model Qwen/Qwen3-VL-30B-A3B-Instruct --fps 2

Output Format:
    - Individual tasks: Separate JSON files per task
    - All tasks: Single unified JSON with keys:
      * answer: QA answer (A/B/C/D/E) or null if error
      * evidence_t: Time evidence answer (A/B/C/D) or null if error
      * evidence_s: Spatial evidence answer (A/B/C/D) or null if error
      * gt_answer, gt_evidence_t, gt_evidence_s: Ground truth answers
"""

import os
import json
import pandas as pd
import ast
import argparse
import glob
import re
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from vllm import LLM, SamplingParams


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
    
    prompt = f"""I will provide you with a video and a multiple choice question about the video.

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
    
    prompt = f"""I have provided you with a video and a question about it. Below are four different time segment options (in seconds) from the video.

Question: {question}

Time Segment Options:
{options_str}

Task: Please select the time segment that best serves as evidence to answer the question, which should be essential and significant to understanding and answering the question.

Output format: Only provide the letter of your choice (A, B, C, or D), no additional text or explanation."""
    
    return prompt


def create_spatial_evidence_prompt(question):
    """Create prompt for Spatial Evidence Selection task.
    Note: The 4 masked images are provided as separate images after this prompt.
    """
    prompt = f"""I have provided you with a video and a question about it. Below, I will show you 4 images extracted from the video. Each image has a red mask boundary overlaid on it, highlighting a different object or region.

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


def process_qa_task(llm, processor, sampling_params, video_path: str, question: str, 
                     options: list, entry_id: str, fps: float, max_frames: int):
    """Process Video QA task: Given video, question, and options, return answer."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Create prompt
        prompt = create_qa_prompt(question, options)
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "max_frames": max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Generate prompt text
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info
        _, video_inputs, video_kwargs = process_vision_info(
            conversations=[conversation], 
            return_video_kwargs=True, 
            return_video_metadata=True
        )
        
        # Create input
        llm_input = {
            "prompt": text, 
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs
        }
        
        # Generate response
        output = llm.generate([llm_input], sampling_params=sampling_params)
        response_text = output[0].outputs[0].text.strip()
        
        answer = parse_answer(response_text)
        print(f"✅ Response for {entry_id}: {response_text}")
        
        return {'answer': answer, 'raw_response': response_text}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_qa_task_batch(llm, processor, sampling_params, batch_data: list, fps: float, max_frames: int):
    """Process batch of Video QA tasks."""
    llm_inputs = []
    
    for item in batch_data:
        video_path = item['video_path']
        question = item['question']
        options = item['options']
        
        # Create prompt
        prompt = create_qa_prompt(question, options)
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "max_frames": max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Generate prompt text
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info
        _, video_inputs, video_kwargs = process_vision_info(
            conversations=[conversation], 
            return_video_kwargs=True, 
            return_video_metadata=True
        )
        
        # Add to batch inputs
        llm_inputs.append({
            "prompt": text, 
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs
        })
    
    # Generate responses for entire batch
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    
    # Parse results
    results = []
    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text.strip()
        answer = parse_answer(response_text)
        entry_id = batch_data[i]['entry_id']
        print(f"✅ Response for {entry_id}: {response_text}")
        results.append({'answer': answer, 'raw_response': response_text})
    
    return results


def process_time_evidence_task(llm, processor, sampling_params, video_path: str, question: str,
                                 time_segments: list, entry_id: str, fps: float, max_frames: int):
    """Process Time Evidence Selection task: Given video, question, and 4 time segment options, choose best one."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Create prompt
        prompt = create_time_evidence_prompt(question, time_segments)
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "max_frames": max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Generate prompt text
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info
        _, video_inputs, video_kwargs = process_vision_info(
            conversations=[conversation], 
            return_video_kwargs=True, 
            return_video_metadata=True
        )
        
        # Create input
        llm_input = {
            "prompt": text, 
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs
        }
        
        # Generate response
        output = llm.generate([llm_input], sampling_params=sampling_params)
        response_text = output[0].outputs[0].text.strip()
        
        answer = parse_answer(response_text)
        print(f"✅ Response for {entry_id}: {response_text}")
        
        return {'answer': answer, 'raw_response': response_text}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_time_evidence_task_batch(llm, processor, sampling_params, batch_data: list, fps: float, max_frames: int):
    """Process batch of Time Evidence Selection tasks."""
    llm_inputs = []
    
    for item in batch_data:
        video_path = item['video_path']
        question = item['question']
        time_segments = item['time_segments']
        
        # Create prompt
        prompt = create_time_evidence_prompt(question, time_segments)
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "max_frames": max_frames,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Generate prompt text
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info
        _, video_inputs, video_kwargs = process_vision_info(
            conversations=[conversation], 
            return_video_kwargs=True, 
            return_video_metadata=True
        )
        
        # Add to batch inputs
        llm_inputs.append({
            "prompt": text, 
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs
        })
    
    # Generate responses for entire batch
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    
    # Parse results
    results = []
    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text.strip()
        answer = parse_answer(response_text)
        entry_id = batch_data[i]['entry_id']
        print(f"✅ Response for {entry_id}: {response_text}")
        results.append({'answer': answer, 'raw_response': response_text})
    
    return results


def process_spatial_evidence_task(llm, processor, sampling_params, video_path: str, 
                                   mask_images: list, question: str, entry_id: str, fps: float, max_frames: int):
    """Process Spatial Evidence Selection task: Given video, 4 masked images and question, choose best one."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Create prompt
        prompt = create_spatial_evidence_prompt(question)
        
        # Build content list: video, then prompt, then images
        content = [
            {
                "type": "video",
                "video": video_path,
                "fps": fps,
                "max_frames": max_frames,
            },
            {"type": "text", "text": prompt}
        ]
        
        # Add masked images
        for img_path in mask_images:
            if not os.path.exists(img_path):
                print(f"⚠️  Warning: Masked image not found: {img_path}")
                continue
            content.append({
                "type": "image",
                "image": img_path,
            })
        
        if len(content) < 3:  # video + prompt + at least 1 image
            print(f"✗ No valid masked images found")
            return None, "No valid masked images found"
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        # Generate prompt text
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations=[conversation], 
            return_video_kwargs=True, 
            return_video_metadata=True
        )
        
        # Create input with both video and images
        multi_modal_data = {}
        if video_inputs is not None:
            multi_modal_data["video"] = video_inputs
        if image_inputs is not None:
            multi_modal_data["image"] = image_inputs
        
        llm_input = {
            "prompt": text, 
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": video_kwargs
        }
        
        # Generate response
        output = llm.generate([llm_input], sampling_params=sampling_params)
        response_text = output[0].outputs[0].text.strip()
        
        answer = parse_answer(response_text)
        print(f"✅ Response for {entry_id}: {response_text}")
        
        return {'answer': answer, 'raw_response': response_text}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_spatial_evidence_task_batch(llm, processor, sampling_params, batch_data: list, fps: float, max_frames: int):
    """Process batch of Spatial Evidence Selection tasks."""
    llm_inputs = []
    
    for item in batch_data:
        video_path = item['video_path']
        question = item['question']
        mask_images = item['mask_images']
        
        # Create prompt
        prompt = create_spatial_evidence_prompt(question)
        
        # Build content list: video, then prompt, then images
        content = [
            {
                "type": "video",
                "video": video_path,
                "fps": fps,
                "max_frames": max_frames,
            },
            {"type": "text", "text": prompt}
        ]
        
        # Add masked images
        for img_path in mask_images:
            if os.path.exists(img_path):
                content.append({
                    "type": "image",
                    "image": img_path,
                })
        
        if len(content) < 3:  # video + prompt + at least 1 image
            continue
        
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        # Generate prompt text
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Process vision info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations=[conversation], 
            return_video_kwargs=True, 
            return_video_metadata=True
        )
        
        # Create input with both video and images
        multi_modal_data = {}
        if video_inputs is not None:
            multi_modal_data["video"] = video_inputs
        if image_inputs is not None:
            multi_modal_data["image"] = image_inputs
        
        llm_inputs.append({
            "prompt": text, 
            "multi_modal_data": multi_modal_data,
            "mm_processor_kwargs": video_kwargs
        })
    
    # Generate responses for entire batch
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    
    # Parse results
    results = []
    for i, output in enumerate(outputs):
        response_text = output.outputs[0].text.strip()
        answer = parse_answer(response_text)
        entry_id = batch_data[i]['entry_id']
        print(f"✅ Response for {entry_id}: {response_text}")
        results.append({'answer': answer, 'raw_response': response_text})
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Independent Task Video Question Answering with Qwen3VL")
    parser.add_argument('--task', type=str, default='all', choices=['qa', 'time_evidence', 'spatial_evidence', 'all'],
                        help='Task type: qa (video QA), time_evidence (time segment selection), spatial_evidence (masked image selection), all (run all tasks)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='Qwen3VL model to use')
    parser.add_argument('--data_file', type=str, default='data/st_evidence_mcq.csv', help='CSV file with video data')
    parser.add_argument('--distractors_file', type=str, default='data/temp_options.json', 
                        help='JSON file with time segment distractors (use {split} placeholder for split name, needed for time_evidence task)')
    parser.add_argument('--mask_file', type=str, default='data/mask_options.json',
                        help='JSON file with masked image paths (needed for spatial_evidence task)')
    parser.add_argument('--save_file', type=str, default=None, help='Save file (default: result/qwen/{model_name}_{task}_{split}_{fps}fps.json)')
    parser.add_argument('--video_dir', type=str, default='data/NextQA-Video', help='Video directory')
    parser.add_argument('--fps', type=int, default=1, help='Frame rate')
    parser.add_argument('--max_frames', type=int, default=768, help='Maximum number of frames to process')
    parser.add_argument('--gpu_num', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference (default: 1 for sequential processing)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=512, help='Maximum output tokens')
    
    args = parser.parse_args()    
    # Initialize VLLM
    print(f"🔄 Initializing VLLM with model {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.gpu_num,
        gpu_memory_utilization=0.95,
        limit_mm_per_prompt={"image": 10, "video": 1}
    )
    
    # Initialize processor
    processor = AutoProcessor.from_pretrained(args.model)
    
    # Create sampling params
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    print(f"✓ VLLM initialized successfully")
    
    # Load CSV data
    print(f"📊 Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)

    test_df = df
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
        # Extract model name from path
        model_name = args.model.split('/')[-1]
        task_name = 'all' if unified_results else args.task
        save_file = f"{results_base}/qwen/{model_name}_{task_name}_{args.fps}fps.json"
    
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
    
    # Process entries: outer loop over tasks (enables batching), inner loop over questions
    processed_count = 0
    
    # Create progress bar
    pbar = tqdm(total=len(test_df) * len(tasks_to_run), desc="Processing", unit="task")
    
    # Process each task separately (enables batching)
    for current_task in tasks_to_run:
        print(f"\n{'='*80}")
        print(f"🔄 Processing {current_task.upper()} task")
        print(f"{'='*80}\n")
        
        # Collect all items to process for this task
        items_to_process = []
        item_metadata = []
        
        for index, row in test_df.iterrows():
            # Get entry_id
            if 'entry_id' not in row or pd.isna(row['entry_id']):
                pbar.update(1)
                continue
                
            entry_id = str(row['entry_id'])
            
            # Check if already done
            if unified_results:
                task_key = 'answer' if current_task == 'qa' else ('evidence_t' if current_task == 'time_evidence' else 'evidence_s')
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
            
            # Collect data based on task type
            try:
                if current_task == 'qa':
                    question = row.get('question', '')
                    options_str = row.get('options', '')
                    options = ast.literal_eval(options_str) if isinstance(options_str, str) else options_str
                    
                    gt_answer_text = row.get('answer', '')
                    gt_answer = get_answer_letter(gt_answer_text, options) if gt_answer_text else ''
                    
                    items_to_process.append({
                        'entry_id': entry_id,
                        'video_path': video_path,
                        'question': question,
                        'options': options
                    })
                    item_metadata.append({
                        'entry_id': entry_id,
                        'video_id': video_id,
                        'gt_answer': gt_answer,
                        'gt_answer_text': gt_answer_text
                    })
                
                elif current_task == 'time_evidence':
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
                    
                    items_to_process.append({
                        'entry_id': entry_id,
                        'video_path': video_path,
                        'question': question,
                        'time_segments': time_segments
                    })
                    item_metadata.append({
                        'entry_id': entry_id,
                        'video_id': video_id,
                        'gt_time_evidence_idx': gt_time_evidence_idx,
                        'gt_time_segments': gt_time_segments
                    })
                
                elif current_task == 'spatial_evidence':
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
                    
                    items_to_process.append({
                        'entry_id': entry_id,
                        'video_path': video_path,
                        'question': question,
                        'mask_images': mask_images
                    })
                    item_metadata.append({
                        'entry_id': entry_id,
                        'video_id': video_id,
                        'gt_spatial_evidence_idx': gt_spatial_evidence_idx,
                        'gt_index': gt_index
                    })
            
            except Exception as e:
                print(f"❌ Error collecting data for {entry_id}: {e}")
                pbar.update(1)
                continue
        
        # Process collected items in batches
        print(f"📦 Collected {len(items_to_process)} items to process for {current_task}")
        
        for i in range(0, len(items_to_process), args.batch_size):
            batch = items_to_process[i:i+args.batch_size]
            batch_meta = item_metadata[i:i+args.batch_size]
            
            try:
                if current_task == 'qa':
                    print(f"\n📹 [QA] Processing batch {i//args.batch_size + 1}/{(len(items_to_process)-1)//args.batch_size + 1}")
                    results = process_qa_task_batch(llm, processor, sampling_params, batch, args.fps, args.max_frames)
                    
                    for j, result in enumerate(results):
                        entry_id = batch[j]['entry_id']
                        meta = batch_meta[j]
                        
                        if unified_results:
                            if entry_id not in result_dict:
                                result_dict[entry_id] = {}
                            result_dict[entry_id]['answer'] = result['answer'] if result else None
                            result_dict[entry_id]['gt_answer'] = meta['gt_answer']
                        else:
                            if result:
                                result_dict[entry_id] = {
                                    'answer': result['answer'],
                                    'raw_response': result['raw_response'],
                                    'gt_answer': meta['gt_answer'],
                                    'gt_answer_text': meta['gt_answer_text']
                                }
                            else:
                                result_dict[entry_id] = {
                                    'answer': None,
                                    'gt_answer': meta['gt_answer'],
                                    'gt_answer_text': meta['gt_answer_text']
                                }
                        pbar.update(1)
                
                elif current_task == 'time_evidence':
                    print(f"\n📹 [TIME] Processing batch {i//args.batch_size + 1}/{(len(items_to_process)-1)//args.batch_size + 1}")
                    results = process_time_evidence_task_batch(llm, processor, sampling_params, batch, args.fps, args.max_frames)
                    
                    for j, result in enumerate(results):
                        entry_id = batch[j]['entry_id']
                        meta = batch_meta[j]
                        
                        if unified_results:
                            if entry_id not in result_dict:
                                result_dict[entry_id] = {}
                            result_dict[entry_id]['evidence_t'] = result['answer'] if result else None
                            result_dict[entry_id]['gt_evidence_t'] = meta['gt_time_evidence_idx']
                        else:
                            if result:
                                result_dict[entry_id] = {
                                    'answer': result['answer'],
                                    'raw_response': result['raw_response'],
                                    'gt_answer': meta['gt_time_evidence_idx'],
                                    'gt_time_segments': meta['gt_time_segments']
                                }
                            else:
                                result_dict[entry_id] = {
                                    'answer': None,
                                    'gt_answer': meta['gt_time_evidence_idx'],
                                    'gt_time_segments': meta['gt_time_segments']
                                }
                        pbar.update(1)
                
                elif current_task == 'spatial_evidence':
                    print(f"\n🖼️  [SPATIAL] Processing batch {i//args.batch_size + 1}/{(len(items_to_process)-1)//args.batch_size + 1}")
                    results = process_spatial_evidence_task_batch(llm, processor, sampling_params, batch, args.fps, args.max_frames)
                    
                    for j, result in enumerate(results):
                        entry_id = batch[j]['entry_id']
                        meta = batch_meta[j]
                        
                        if unified_results:
                            if entry_id not in result_dict:
                                result_dict[entry_id] = {}
                            result_dict[entry_id]['evidence_s'] = result['answer'] if result else None
                            result_dict[entry_id]['gt_evidence_s'] = meta['gt_spatial_evidence_idx']
                        else:
                            if result:
                                result_dict[entry_id] = {
                                    'answer': result['answer'],
                                    'raw_response': result['raw_response'],
                                    'gt_answer': meta['gt_spatial_evidence_idx'],
                                    'gt_index': meta['gt_index']
                                }
                            else:
                                result_dict[entry_id] = {
                                    'answer': None,
                                    'gt_answer': meta['gt_spatial_evidence_idx'],
                                    'gt_index': meta['gt_index']
                                }
                        pbar.update(1)
                
                # Save after each batch
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                processed_count += len(batch)
                
            except Exception as e:
                print(f"❌ Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                # Update progress for failed batch items
                for _ in range(len(batch)):
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

