#!/usr/bin/env python3
"""
Independent Task Video Question Answering for ST-Evidence MCQ using Vertex AI

Three independent tasks (processed sequentially for each question):
- qa: Video QA - Given video, question, and options, model provides answer
- time_evidence: Time Evidence Selection - Given video, question, and 4 time segments, choose best one
- spatial_evidence: Spatial Evidence Selection - Given video, question, and 4 masked images, choose best one
- all: Run all three tasks sequentially (question by question)
- multiturn: Run all three tasks in one chat session with context carried forward

Usage:
    # Run video QA task
    python gemini_st_evidence_mcq.py --task qa --split val
    
    # Run time evidence selection task
    python gemini_st_evidence_mcq.py --task time_evidence --split val
    
    # Run spatial evidence selection task
    python gemini_st_evidence_mcq.py --task spatial_evidence --split val
    
    # Run all tasks sequentially (saves to single unified file)
    python gemini_st_evidence_mcq.py --task all --split val
    # This creates: result/gemini/{model}_all_{split}_{fps}fps.json
    # with format: {"entry_id": {"answer": "A", "evidence_t": "B", "evidence_s": "C", ...}}
    
    # Run all tasks in multi-turn dialog (saves to single unified file)
    python gemini_st_evidence_mcq.py --task multiturn --split val
    # This creates: result/gemini/{model}_multiturn_{split}_{fps}fps.json
    # Previous task responses serve as context for later tasks
    
    # Custom model and FPS
    python gemini_st_evidence_mcq.py --task qa --split train --model gemini-2.5-flash --fps 2

Output Format:
    - Individual tasks: Separate JSON files per task
    - All tasks: Single unified JSON with keys:
      * answer: QA answer (A/B/C/D/E) or null if error
      * evidence_t: Time evidence answer (A/B/C/D) or null if error
      * evidence_s: Spatial evidence answer (A/B/C/D) or null if error
      * gt_answer, gt_evidence_t, gt_evidence_s: Ground truth answers
    - Multiturn: Same format as 'all' but with contextual responses
"""

from google import genai
from google.genai import types
from google.genai.types import Part, Blob, GenerateContentConfig, VideoMetadata
import os
import time
import json
import pandas as pd
import ast
import argparse
import glob
from pathlib import Path
from tqdm import tqdm


def initialize_gemini(project_id: str = "salesforce-research-internal", location: str = "us-central1", model: str = None):
    """Initialize Gemini client with Vertex AI."""
    try:
        # Initialize genai client with Vertex AI
        client = genai.Client(vertexai=True, project=project_id, location=location)
        
        # gemini-2.5-pro: no thinking_config; flash: thinking_budget=0
        if model and 'pro' in model.lower() and 'flash' not in model.lower():
            config = GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
            )
        else:
            config = GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        
        print(f"✓ Successfully initialized Gemini client for project: {project_id}")
        return client, config
    except Exception as e:
        print(f"✗ Failed to initialize Gemini client: {e}")
        return None


def get_mime_type(video_path: str) -> str:
    """Get MIME type based on file extension."""
    file_ext = Path(video_path).suffix.lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska'
    }
    return mime_types.get(file_ext, 'video/mp4')


def print_thinking_summary(response, entry_id: str = "", prefix: str = ""):
    """
    Extract and print thinking summary from Gemini response.
    
    Args:
        response: Gemini API response object
        entry_id: Optional entry identifier for context
        prefix: Optional prefix for indentation (e.g., "  " or "    ")
    
    Returns:
        tuple: (thought_text, answer_text) or (None, response.text)
    """
    thought_text = None
    answer_text = None
    
    if response and response.candidates and len(response.candidates) > 0:
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought_text = part.text
                print(f"{prefix}💭 THINKING SUMMARY {f'({entry_id})' if entry_id else ''}:")
                print(f"{prefix}   {part.text}")
            else:
                answer_text = part.text
                print(f"{prefix}✅ FINAL ANSWER {f'({entry_id})' if entry_id else ''}: {part.text}")
        
        if thought_text or answer_text:
            return thought_text, answer_text
    
    # Fallback if no structured parts found
    return None, (response.text if response else None)


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
            # Handle nested list format
            if isinstance(segments, list) and len(segments) > 0:
                # Format each segment
                segment_strs = []
                for seg in segments:
                    if isinstance(seg, list) and len(seg) == 2:
                        segment_strs.append(f"[{seg[0]}, {seg[1]}]")
                    else:
                        segment_strs.append(str(seg))
                formatted_options.append(f"({option_letters[i]}) {', '.join(segment_strs)}")
            else:
                formatted_options.append(f"({option_letters[i]}) {segments}")
    
    options_str = "\n".join(formatted_options)
    
    prompt = f"""I have provided you with a video and a question about it. Below are four different time segment options (in seconds) from the video.

Question: {question}

Time Segment Options:
{options_str}

Task: Please select the time segment that best serves as evidence to answer the question, which should be essential and significant to understanding and answering the question.

Output format: Only provide the letter of your choice (A, B, C, or D), no additional text or explanation."""
    
    return prompt


def create_spatial_evidence_prompt(question):
    """Create prompt for Spatial Evidence Selection task with Video and Masked Images."""
    prompt = f"""I have provided you with a video and a question about it. Below, I will show you 4 images extracted from the video. Each image has a red mask boundary overlaid on it, highlighting a different object or region.

Question: {question}

Task: Please select the image where the highlighted region best serves as spatial evidence to answer this question. The highlighted object or region should be directly relevant to understanding the question or answering the question.

Note: The images are presented in order as options A, B, C, and D (first image = A, second = B, third = C, fourth = D).

Output format: Only provide the letter of your choice (A, B, C, or D), no additional text or explanation."""
    
    return prompt


def parse_answer(response_text: str):
    """Parse answer letter from model response."""
    import re
    
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


def load_image_as_blob(image_path: str):
    """Load an image file and return as Blob."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Determine MIME type based on extension
    ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(ext, 'image/jpeg')
    
    # Read image data
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    return Blob(mime_type=mime_type, data=image_data)


def process_qa_task(client, config, model_name: str, video_path: str, question: str, 
                     options: list, entry_id: str, fps: float):
    """Process Video QA task: Given video, question, and options, return answer."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Create video part
        mime_type = get_mime_type(video_path)
        with open(video_path, 'rb') as f:
            video_data = f.read()
        blob = Blob(mime_type=mime_type, data=video_data)
        video_part = Part(inlineData=blob, videoMetadata=VideoMetadata(fps=fps))
        
        # Create prompt
        prompt = create_qa_prompt(question, options)
        
        # Generate response
        response = client.models.generate_content(
            model=model_name,
            contents=[video_part, prompt],
            config=config
        )
        
        answer = parse_answer(response.text) if response else None
        print(f"✅ Response for {entry_id}: {response.text if response else 'None'}")
        
        return {'answer': answer, 'raw_response': response.text if response else None}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_time_evidence_task(client, config, model_name: str, video_path: str, question: str, 
                                 time_segments: list, entry_id: str, fps: float):
    """Process Time Evidence Selection task: Given video, question, and 4 time segments, choose best one."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Create video part
        mime_type = get_mime_type(video_path)
        with open(video_path, 'rb') as f:
            video_data = f.read()
        blob = Blob(mime_type=mime_type, data=video_data)
        video_part = Part(inlineData=blob, videoMetadata=VideoMetadata(fps=fps))
        
        # Create prompt
        prompt = create_time_evidence_prompt(question, time_segments)
        
        # Generate response
        response = client.models.generate_content(
            model=model_name,
            contents=[video_part, prompt],
            config=config
        )
        
        answer = parse_answer(response.text) if response else None
        print(f"✅ Response for {entry_id}: {response.text if response else 'None'}")
        
        return {'answer': answer, 'raw_response': response.text if response else None}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_spatial_evidence_task(client, config, model_name: str, video_path: str, 
                                   mask_images: list, question: str, entry_id: str, fps: float,
                                   print_thinking: bool = False):
    """Process Spatial Evidence Selection task: Given video, 4 masked images and question, choose best one."""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Create video part
        mime_type = get_mime_type(video_path)
        with open(video_path, 'rb') as f:
            video_data = f.read()
        blob = Blob(mime_type=mime_type, data=video_data)
        video_part = Part(inlineData=blob, videoMetadata=VideoMetadata(fps=fps))
        
        # Load masked images
        image_parts = []
        for i, img_path in enumerate(mask_images):
            if not os.path.exists(img_path):
                print(f"⚠️  Warning: Masked image not found: {img_path}")
                continue
            try:
                img_blob = load_image_as_blob(img_path)
                image_parts.append(Part(inlineData=img_blob))
            except Exception as e:
                print(f"⚠️  Warning: Failed to load image {img_path}: {e}")
        
        if len(image_parts) == 0:
            print(f"✗ No valid masked images found")
            return None, "No valid masked images found"
        
        # Create prompt
        prompt = create_spatial_evidence_prompt(question)
        
        # Generate response with video first, then question, then images
        contents = [video_part] + [prompt] + image_parts
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )
        
        # Extract thinking summary and answer
        if print_thinking:
            thought_text, answer_text = print_thinking_summary(response, entry_id, prefix="  ")
        else:
            thought_text = None
            answer_text = None
            if response and response.candidates and len(response.candidates) > 0:
                for part in response.candidates[0].content.parts:
                    if part.text and not part.thought:
                        answer_text = part.text
                        break
        
        # Use answer_text if available, otherwise fallback to response.text
        final_text = answer_text if answer_text else (response.text if response else None)
        
        # Parse answer
        answer = parse_answer(final_text) if final_text else None
        if not print_thinking:
            print(f"✅ Response for {entry_id}: {response.text if response else 'None'}")
        
        return {'answer': answer, 'raw_response': response.text if response else None}, None
        
    except Exception as e:
        print(f"✗ Error processing {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def process_multiturn_task(client, config, model_name: str, video_path: str, 
                            question: str, options: list, 
                            time_segments: list, mask_images: list, 
                            entry_id: str, fps: float,
                            print_thinking: bool = False):
    """
    Process all three tasks in a multi-turn dialog session.
    The context from previous turns is preserved for later tasks.
    
    Returns: dict with keys 'answer', 'evidence_t', 'evidence_s', or None if error
    """
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None, f"Video file not found: {video_path}"
    
    try:
        # Create video part
        mime_type = get_mime_type(video_path)
        with open(video_path, 'rb') as f:
            video_data = f.read()
        blob = Blob(mime_type=mime_type, data=video_data)
        video_part = Part(inlineData=blob, videoMetadata=VideoMetadata(fps=fps))
        
        # Start a chat session
        chat = client.chats.create(
            model=model_name,
            config=config
        )
        
        # Turn 1: Video QA
        print(f"  🔹 Turn 1: QA task")
        qa_prompt = create_qa_prompt(question, options)
        qa_response = chat.send_message([video_part, qa_prompt])
        qa_answer = parse_answer(qa_response.text) if qa_response else None
        print(f"  ✅ QA Response: {qa_response.text if qa_response else 'None'}")
        
        # Turn 2: Time Evidence Selection (with context from Turn 1)
        print(f"  🔹 Turn 2: Time evidence task (with QA context)")
        time_prompt = create_time_evidence_prompt(question, time_segments)
        time_response = chat.send_message(time_prompt)
        time_answer = parse_answer(time_response.text) if time_response else None
        print(f"  ✅ Time Evidence Response: {time_response.text if time_response else 'None'}")
        
        # Turn 3: Spatial Evidence Selection (with context from Turn 1 & 2)
        print(f"  🔹 Turn 3: Spatial evidence task (with QA + Time context)")
        
        # Load masked images
        image_parts = []
        for i, img_path in enumerate(mask_images):
            if not os.path.exists(img_path):
                print(f"    ⚠️  Warning: Masked image not found: {img_path}")
                continue
            try:
                img_blob = load_image_as_blob(img_path)
                image_parts.append(Part(inlineData=img_blob))
            except Exception as e:
                print(f"    ⚠️  Warning: Failed to load image {img_path}: {e}")
        
        if len(image_parts) == 0:
            print(f"  ✗ No valid masked images found for spatial evidence")
            spatial_answer = None
        else:
            spatial_prompt = create_spatial_evidence_prompt(question)
            spatial_response = chat.send_message([spatial_prompt] + image_parts)
            
            # Extract thinking summary and answer
            if print_thinking:
                thought_text, answer_text = print_thinking_summary(spatial_response, f"{entry_id} Turn 3", prefix="    ")
            else:
                thought_text = None
                answer_text = None
                if spatial_response and spatial_response.candidates and len(spatial_response.candidates) > 0:
                    for part in spatial_response.candidates[0].content.parts:
                        if part.text and not part.thought:
                            answer_text = part.text
                            break
            
            # Use answer_text if available, otherwise fallback to response.text
            final_text = answer_text if answer_text else (spatial_response.text if spatial_response else None)
            
            spatial_answer = parse_answer(final_text) if final_text else None
            if not print_thinking:
                print(f"  ✅ Spatial Evidence Response: {spatial_response.text if spatial_response else 'None'}")
        
        return {
            'answer': qa_answer,
            'evidence_t': time_answer,
            'evidence_s': spatial_answer
        }, None
        
    except Exception as e:
        print(f"✗ Error in multi-turn processing for {entry_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description="Independent Task Video Question Answering with VertexAI API")
    parser.add_argument('--task', type=str, required=True, choices=['qa', 'time_evidence', 'spatial_evidence', 'all', 'multiturn'],
                        help='Task type: qa (video QA), time_evidence (time segment selection), spatial_evidence (masked image selection), all (run all tasks independently), multiturn (run all tasks in one chat with context)')
    parser.add_argument('--model', type=str, default='gemini-2.5-pro', help='Gemini model to use')
    parser.add_argument('--project', type=str, default='salesforce-research-internal', help='Google Cloud project ID')
    parser.add_argument('--location', type=str, default='us-central1', help='Vertex AI location')
    parser.add_argument('--data_file', type=str, default='data/st_evidence_mcq.csv', help='CSV file with video data')
    parser.add_argument('--distractors_file', type=str, default='data/temp_options.json', 
                        help='JSON file with time segment distractors (use {split} placeholder for split name, needed for time_evidence task)')
    parser.add_argument('--mask_file', type=str, default='data/mask_options.json',
                        help='JSON file with masked image paths (needed for spatial_evidence task)')
    parser.add_argument('--save_file', type=str, default=None, help='Save file (default: result/gemini/{model}_{task}_{split}_{fps}fps.json)')
    parser.add_argument('--video_dir', type=str, default='data/NextQA-Video', help='Video directory')
    parser.add_argument('--fps', type=int, default=1, help='Frame rate')
    parser.add_argument('--print_thinking', action='store_true',
                        help='Print thinking summary from model responses (shows reasoning process)')
    args = parser.parse_args()    
    # Initialize Gemini client
    result = initialize_gemini(args.project, args.location, model=args.model)
    if not result:
        return
    
    client, config = result
    
    # Determine tasks to run
    if args.task == 'all':
        tasks_to_run = ['qa', 'time_evidence', 'spatial_evidence']
        unified_results = True
        multiturn_mode = False
    elif args.task == 'multiturn':
        tasks_to_run = ['qa', 'time_evidence', 'spatial_evidence']  # All tasks in one session
        unified_results = True
        multiturn_mode = True
    else:
        tasks_to_run = [args.task]
        unified_results = False
        multiturn_mode = False
    
    if multiturn_mode:
        print(f"\n🎯 Mode: Multi-turn dialog (all tasks in one chat session)")
    else:
        print(f"\n🎯 Tasks to run: {', '.join(tasks_to_run)}")
    
    # Load CSV data
    if not os.path.exists(args.data_file):
        print(f"❌ Data file not found: {args.data_file}")
        return
        
    test_df = pd.read_csv(args.data_file).astype(str)
    print(f"📊 Using all {len(test_df)} entries")
    
    # Load task-specific data
    distractors_dict = None
    mask_dict = None
    
    if 'time_evidence' in tasks_to_run:
        distractors_file = args.distractors_file
        if not os.path.exists(distractors_file):
            print(f"❌ Distractors file not found: {distractors_file}")
            if args.task == 'time_evidence':
                return
            tasks_to_run = [t for t in tasks_to_run if t != 'time_evidence']
        else:
            with open(distractors_file, 'r') as f:
                distractors_dict = json.load(f)
            print(f"📊 Loaded {len(distractors_dict)} entries from distractors file")
    
    if 'spatial_evidence' in tasks_to_run:
        if not os.path.exists(args.mask_file):
            print(f"❌ Mask file not found: {args.mask_file}")
            if args.task == 'spatial_evidence':
                return
            tasks_to_run = [t for t in tasks_to_run if t != 'spatial_evidence']
        else:
            with open(args.mask_file, 'r') as f:
                mask_dict = json.load(f)
            print(f"📊 Loaded {len(mask_dict)} entries from mask file")
    
    # Build video mapping
    video_mapping = build_video_mapping(args.video_dir)
    
    # Setup results file
    results_base = 'result'
    if unified_results:
        if args.save_file is None:
            task_name = 'multiturn' if multiturn_mode else 'all'
            save_file = f'{results_base}/gemini/{args.model}_{task_name}_{args.fps}fps.json'
        else:
            save_file = args.save_file
    else:
        task = tasks_to_run[0]
        if args.save_file is None:
            save_file = f'{results_base}/gemini/{args.model}_{task}_{args.fps}fps.json'
        else:
            save_file = args.save_file
    
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # Load existing results
    result_dict = {}
    if os.path.exists(save_file):
        result_dict = json.load(open(save_file, 'r'))
        print(f"📂 Loaded existing results: {len(result_dict)} entries")
    
    print(f"💾 Results will be saved to: {save_file}\n")
    
    # Print statistics
    if unified_results:
        qa_valid = sum(1 for entry in result_dict.values() if entry.get('answer') is not None)
        time_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_t') is not None)
        spatial_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_s') is not None)
        print(f"📊 Current status: QA={qa_valid}, Time={time_valid}, Spatial={spatial_valid}")
    
    # ============================================================
    # MULTI-TURN PROCESSING MODE
    # ============================================================
    if multiturn_mode:
        print("\n🔄 Starting multi-turn dialog processing...\n")
        
        # Create progress bar for multiturn (one entry per question)
        pbar = tqdm(total=len(test_df), desc="Processing", unit="question")
        
        for index, row in test_df.iterrows():
            # Get entry_id
            if 'entry_id' not in row or pd.isna(row['entry_id']):
                print(f"⚠️  Skipping row {index}: missing entry_id")
                pbar.update(1)
                continue
                
            entry_id = str(row['entry_id'])
            
            # Check if already fully processed
            if (entry_id in result_dict and 
                result_dict[entry_id].get('answer') is not None and
                result_dict[entry_id].get('evidence_t') is not None and
                result_dict[entry_id].get('evidence_s') is not None):
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
            
            # Gather all required data
            # 1. QA data
            question = row.get('question', '')
            options_str = row.get('options', '')
            try:
                options = ast.literal_eval(options_str)
            except:
                options = [options_str]
            
            gt_answer_text = row.get('answer', '')
            gt_answer = get_answer_letter(gt_answer_text, options) if gt_answer_text else ''
            
            # 2. Time evidence data
            if entry_id not in distractors_dict:
                print(f"⚠️  Skipping {entry_id}: missing time evidence data")
                pbar.update(1)
                continue
            
            distractor_data = distractors_dict[entry_id]
            time_segments = distractor_data.get('time_evidence_options', [])
            
            if not time_segments or len(time_segments) < 4:
                print(f"⚠️  Skipping {entry_id}: invalid time segments")
                pbar.update(1)
                continue
            
            gt_time_evidence_idx = distractor_data.get('correct_answer', '')
            
            # 3. Spatial evidence data
            mask_data = None
            if entry_id in mask_dict:
                mask_data = mask_dict[entry_id]
            else:
                for key, data in mask_dict.items():
                    if data.get('entry_id') == entry_id:
                        mask_data = data
                        break
            
            if not mask_data:
                print(f"⚠️  Skipping {entry_id}: missing spatial evidence data")
                pbar.update(1)
                continue
            
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
                print(f"⚠️  Skipping {entry_id}: invalid mask images")
                pbar.update(1)
                continue
            
            gt_index = mask_data.get('answer', mask_data.get('correct_answer', 0))
            gt_spatial_evidence_idx = chr(65 + gt_index)
            
            # Process all three tasks in multi-turn dialog
            print(f"\n🔄 [MULTITURN] Processing {entry_id}: {video_id}.mp4")
            
            try:
                result, error = process_multiturn_task(
                    client, config, args.model, video_path,
                    question, options, time_segments, mask_images,
                    entry_id, args.fps, args.print_thinking
                )
                
                if result:
                    result_dict[entry_id] = {
                        'answer': result.get('answer'),
                        'evidence_t': result.get('evidence_t'),
                        'evidence_s': result.get('evidence_s'),
                        'gt_answer': gt_answer,
                        'gt_evidence_t': gt_time_evidence_idx,
                        'gt_evidence_s': gt_spatial_evidence_idx
                    }
                else:
                    # Partial results on error
                    if entry_id not in result_dict:
                        result_dict[entry_id] = {}
                    result_dict[entry_id].update({
                        'answer': None,
                        'evidence_t': None,
                        'evidence_s': None,
                        'gt_answer': gt_answer,
                        'gt_evidence_t': gt_time_evidence_idx,
                        'gt_evidence_s': gt_spatial_evidence_idx
                    })
                
                # Save after each entry
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error processing {entry_id}: {e}")
                import traceback
                traceback.print_exc()
            
            pbar.update(1)
        
        pbar.close()
    
    # ============================================================
    # INDEPENDENT TASK PROCESSING MODE (original)
    # ============================================================
    else:
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
                        try:
                            options = ast.literal_eval(options_str)
                        except:
                            options = [options_str]
                        
                        gt_answer_text = row.get('answer', '')
                        gt_answer = get_answer_letter(gt_answer_text, options) if gt_answer_text else ''
                        
                        print(f"\n📹 [QA] Processing {entry_id}: {video_id}.mp4")
                        
                        result, error = process_qa_task(client, config, args.model, video_path, question, options, entry_id, args.fps)
                        
                        if unified_results:
                            if entry_id not in result_dict:
                                result_dict[entry_id] = {}
                            result_dict[entry_id]['answer'] = result['answer'] if result else None
                            result_dict[entry_id]['gt_answer'] = gt_answer
                        else:
                            if result:
                                result_dict[entry_id] = {
                                    'answer': result['answer'],
                                    'raw_response': result['raw_response'],
                                    'gt_answer': gt_answer,
                                    'gt_answer_text': gt_answer_text
                                }
                            else:
                                result_dict[entry_id] = {
                                    'answer': None,
                                    'gt_answer': gt_answer,
                                    'gt_answer_text': gt_answer_text,
                                    'error': error
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
                        
                        result, error = process_time_evidence_task(client, config, args.model, video_path, question, time_segments, entry_id, args.fps)
                        
                        if unified_results:
                            if entry_id not in result_dict:
                                result_dict[entry_id] = {}
                            result_dict[entry_id]['evidence_t'] = result['answer'] if result else None
                            result_dict[entry_id]['gt_evidence_t'] = gt_time_evidence_idx
                        else:
                            if result:
                                result_dict[entry_id] = {
                                    'answer': result['answer'],
                                    'raw_response': result['raw_response'],
                                    'gt_answer': gt_time_evidence_idx,
                                    'gt_time_segments': gt_time_segments
                                }
                            else:
                                result_dict[entry_id] = {
                                    'answer': None,
                                    'gt_answer': gt_time_evidence_idx,
                                    'gt_time_segments': gt_time_segments,
                                    'error': error
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
                        
                        result, error = process_spatial_evidence_task(client, config, args.model, video_path, mask_images, question, entry_id, args.fps, args.print_thinking)
                        
                        if unified_results:
                            if entry_id not in result_dict:
                                result_dict[entry_id] = {}
                            result_dict[entry_id]['evidence_s'] = result['answer'] if result else None
                            result_dict[entry_id]['gt_evidence_s'] = gt_spatial_evidence_idx
                        else:
                            if result:
                                result_dict[entry_id] = {
                                    'answer': result['answer'],
                                    'raw_response': result['raw_response'],
                                    'gt_answer': gt_spatial_evidence_idx,
                                    'gt_index': gt_index
                                }
                            else:
                                result_dict[entry_id] = {
                                    'answer': None,
                                    'gt_answer': gt_spatial_evidence_idx,
                                    'gt_index': gt_index,
                                    'error': error
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
        total_entries = len(result_dict)
        qa_valid = sum(1 for entry in result_dict.values() if entry.get('answer') is not None)
        time_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_t') is not None)
        spatial_valid = sum(1 for entry in result_dict.values() if entry.get('evidence_s') is not None)
        all_complete = sum(1 for entry in result_dict.values() 
                          if entry.get('answer') is not None 
                          and entry.get('evidence_t') is not None 
                          and entry.get('evidence_s') is not None)
        
        print(f"\n📊 UNIFIED RESULTS SUMMARY:")
        print(f"📋 Total entries: {total_entries}")
        print(f"✅ QA (answer): {qa_valid}/{total_entries} ({(qa_valid/total_entries)*100:.1f}%)" if total_entries > 0 else "✅ QA: 0/0")
        print(f"✅ Time (evidence_t): {time_valid}/{total_entries} ({(time_valid/total_entries)*100:.1f}%)" if total_entries > 0 else "✅ Time: 0/0")
        print(f"✅ Spatial (evidence_s): {spatial_valid}/{total_entries} ({(spatial_valid/total_entries)*100:.1f}%)" if total_entries > 0 else "✅ Spatial: 0/0")
        print(f"✅ All three complete: {all_complete}/{total_entries} ({(all_complete/total_entries)*100:.1f}%)" if total_entries > 0 else "✅ All complete: 0/0")
    else:
        total_entries = len(result_dict)
        valid = sum(1 for entry in result_dict.values() if entry.get('answer') is not None and not entry.get('error', False))
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"📋 Total entries: {total_entries}")
        print(f"✅ Valid responses: {valid}/{total_entries} ({(valid/total_entries)*100:.1f}%)" if total_entries > 0 else "✅ Valid: 0/0")
    
    print(f"💾 Results saved to: {save_file}")


if __name__ == "__main__":
    main()

