#!/usr/bin/env python3
"""
Independent Task Video Question Answering for ST-Evidence MCQ using LLaVA-OneVision-1.5

Three independent tasks (processed sequentially for each question):
- qa: Video QA - Given video, question, and options, model provides answer
- time_evidence: Time Evidence Selection - Given video, question, and 4 time segments, choose best one
- spatial_evidence: Spatial Evidence Selection - Given video, question, and 4 masked images, choose best one
- all: Run all three tasks sequentially (question by question)

Supported models:
- lmms-lab/LLaVA-OneVision-1.5-8B-Instruct
- lmms-lab/LLaVA-OneVision-1.5-13B-Instruct

Usage:
    python llava_ov1_5_st_evidence_mcq.py --task qa --split val
    python llava_ov1_5_st_evidence_mcq.py --task all --split val
    python llava_ov1_5_st_evidence_mcq.py --task qa --split val --model lmms-lab/LLaVA-OneVision-1.5-13B-Instruct

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
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu
from PIL import Image


def extract_video_frames(video_path, fps=1.0, max_frames=64, max_size=None):
    """Extract frames from video as PIL Images."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    video_fps = float(vr.get_avg_fps())
    total_duration = len(vr) / video_fps
    num_frames = min(max_frames, max(1, int(total_duration * fps)))
    
    frame_indices = []
    for i in range(num_frames):
        frame_idx = int((i * len(vr)) / num_frames)
        frame_indices.append(min(frame_idx, max_frame))
    
    frames = []
    for frame_idx in frame_indices:
        img = Image.fromarray(vr[frame_idx].asnumpy()).convert('RGB')
        if max_size is not None:
            w, h = img.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        frames.append(img)
    return frames


def load_image(image_path, max_size=None):
    """Load and optionally resize a single image."""
    img = Image.open(image_path).convert('RGB')
    if max_size is not None:
        w, h = img.size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


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


def process_qa_task(model, processor, video_path: str, question: str, 
                     options: list, entry_id: str, fps: float, max_frames: int, max_size: int):
    """Process Video QA task."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    try:
        frames = extract_video_frames(video_path, fps=fps, max_frames=max_frames, max_size=max_size)
        prompt = create_qa_prompt(question, options)
        
        content = []
        for frame in frames:
            image_dict = {"type": "image", "image": frame}
            if max_size is not None:
                image_dict["max_pixels"] = max_size * max_size
            content.append(image_dict)
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        answer = parse_answer(response)
        print(f"✅ Response for {entry_id}: {response}")
        return {'answer': answer}, None
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_time_evidence_task(model, processor, video_path: str, question: str,
                                 time_segments: list, entry_id: str, fps: float, max_frames: int, max_size: int):
    """Process Time Evidence Selection task."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    try:
        frames = extract_video_frames(video_path, fps=fps, max_frames=max_frames, max_size=max_size)
        prompt = create_time_evidence_prompt(question, time_segments)
        
        content = []
        for frame in frames:
            image_dict = {"type": "image", "image": frame}
            if max_size is not None:
                image_dict["max_pixels"] = max_size * max_size
            content.append(image_dict)
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        answer = parse_answer(response)
        print(f"✅ Response for {entry_id}: {response}")
        return {'answer': answer}, None
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_spatial_evidence_task(model, processor, video_path: str, 
                                   mask_images: list, question: str, entry_id: str, fps: float, max_frames: int, max_size: int):
    """Process Spatial Evidence Selection task with video and images."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    try:
        # Load video frames
        frames = extract_video_frames(video_path, fps=fps, max_frames=max_frames, max_size=max_size)
        prompt = create_spatial_evidence_prompt(question)
        
        # Build content: video frames first, then prompt, then masked images
        content = []
        for frame in frames:
            image_dict = {"type": "image", "image": frame}
            if max_size is not None:
                image_dict["max_pixels"] = max_size * max_size
            content.append(image_dict)
        content.append({"type": "text", "text": prompt})
        
        # Add masked images
        for img_path in mask_images:
            if not os.path.exists(img_path):
                print(f"⚠️  Warning: Masked image not found: {img_path}")
                continue
            try:
                img = load_image(img_path, max_size=max_size)
                image_dict = {"type": "image", "image": img}
                if max_size is not None:
                    image_dict["max_pixels"] = max_size * max_size
                content.append(image_dict)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load image {img_path}: {e}")
        
        if len(content) <= len(frames) + 1:  # Only frames and prompt, no masked images
            print(f"✗ No valid masked images found")
            return None, "No valid masked images found"
        
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        
        answer = parse_answer(response)
        print(f"✅ Response for {entry_id}: {response}")
        return {'answer': answer}, None
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Independent Task Video Question Answering with LLaVA-OneVision-1.5")
    parser.add_argument('--task', type=str, required=True, choices=['qa', 'time_evidence', 'spatial_evidence', 'all'],
                        help='Task type')
    parser.add_argument('--model', type=str, default='lmms-lab/LLaVA-OneVision-1.5-8B-Instruct', help='LLaVA-OV1.5 model to use')
    parser.add_argument('--data_file', type=str, default='data/st_evidence_mcq.csv', help='CSV file with video data')
    parser.add_argument('--distractors_file', type=str, default='data/temp_options.json', 
                        help='JSON file with time segment distractors')
    parser.add_argument('--mask_file', type=str, default='data/mask_options.json',
                        help='JSON file with masked image paths')
    parser.add_argument('--save_file', type=str, default=None, help='Save file')
    parser.add_argument('--video_dir', type=str, default='data/NextQA-Video', help='Video directory')
    parser.add_argument('--fps', type=float, default=1.0, help='Frame rate')
    parser.add_argument('--max_frames', type=int, default=64, help='Maximum number of frames to extract')
    parser.add_argument('--max_size', type=int, default=512, help='Maximum size for frame/image resizing (for memory efficiency)')
    args = parser.parse_args()    
    # Initialize model
    print(f"\n🔄 Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
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
        model_name = args.model.split('/')[-1].replace('-', '_').lower()
        task_name = 'all' if unified_results else args.task
        save_file = f"{results_base}/llava_ov/{model_name}_{task_name}_{args.fps}fps.json"
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
                    result, error = process_qa_task(model, processor, video_path, question, options, entry_id, args.fps, args.max_frames, args.max_size)
                    
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
                    result, error = process_time_evidence_task(model, processor, video_path, question, time_segments, entry_id, args.fps, args.max_frames, args.max_size)
                    
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
                    result, error = process_spatial_evidence_task(model, processor, video_path, mask_images, question, entry_id, args.fps, args.max_frames, args.max_size)
                    
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

