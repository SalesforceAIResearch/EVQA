import os
import json
import argparse
import ast
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math
import numpy as np
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

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

# Video processing functions
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
    """
    Load video and sample frames based on fps parameter.
    
    Args:
        video_path: Path to video file
        bound: Optional time bounds [start, end]
        input_size: Size for image preprocessing
        max_num: Maximum number of patches per frame
        fps: Frames per second to sample (default: 1.0)
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    video_fps = float(vr.get_avg_fps())
    
    # Calculate total video duration
    total_duration = len(vr) / video_fps
    
    # Calculate number of frames to sample based on fps
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

def create_answer_prompt(question, options):
    """Create prompt for answering the question"""
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

def create_temporal_evidence_prompt(question, options, answer):
    """Create prompt for generating temporal evidence (uses conversation history)"""
    prompt = f"""Now, please identify the essential time segments (in seconds) in the video that are highly related and required to answer this question. These should be the specific moments where key events, interactions, or changes occur that directly support your answer.

Output format: Provide ONLY a nested list of time segments in the format:
[[start1, end1], [start2, end2], ...]

For example:
[[2.5, 5.8], [7.1, 9.4]]

Only output the nested list, no additional text or explanation."""
    
    return prompt

def create_ref_expression_prompt(question, options, answer, evidence_segments):
    """Create prompt for generating referring expressions (uses conversation history)"""
    prompt = f"""Now, identify all the objects or regions in the video that serve as evidence for answering the question.

Task: List all objects or regions that are essential evidence. These should be:
1. Objects related to the QUESTION (objects that help understand what is being asked)
2. Objects related to the ANSWER (objects that directly support or constitute the answer)

REFERRING EXPRESSION GUIDELINES:
- Keep expressions simple but discriminative (e.g., "red cube" or "red cube on left")
- Use basic descriptors: color, shape, size, materials, actions, simple position (left/right/center/top/bottom), relative position (left of, right of, behind, in front of, etc.)
- Avoid complex relationships or temporal descriptions
- Make them clear enough for grounding models to locate objects
- Focus only on objects/regions essential for answering the question

Output format: Provide ONLY a list of referring expressions in the format:
["expression 1", "expression 2", "expression 3", ...]

For example:
["person wearing a blue jacket", "red cube on the left", "yellow sphere in the center"]

Only output the list, no additional text or explanation."""
    
    return prompt


def create_simultaneous_prompt(question, options):
    """Create prompt for single-turn mode - answer all three questions at once (JSON format)"""
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

Task: Please complete the following three tasks:

1. ANSWER: Select the correct option (A, B, C, D, or E) based on the video.

2. TEMPORAL EVIDENCE: Identify the essential time segments (in seconds) in the video that are highly related and required to answer this question. These should be the specific moments where key events, interactions, or changes occur that directly support your answer.

3. REFERRING EXPRESSIONS: Identify all the objects or regions in the video that serve as evidence for answering the question. These should be:
   - Objects related to the QUESTION (objects that help understand what is being asked)
   - Objects related to the ANSWER (objects that directly support or constitute the answer)

REFERRING EXPRESSION GUIDELINES:
- Keep expressions simple but discriminative (e.g., "red cube" or "red cube on left")
- Use basic descriptors: color, shape, size, materials, actions, simple position (left/right/center/top/bottom), relative position (left of, right of, behind, in front of, etc.)
- Avoid complex relationships or temporal descriptions
- Make them clear enough for grounding models to locate objects
- Focus only on objects/regions essential for answering the question

Output format: Please provide your response in JSON format with the following structure:

{{
  "answer": "A",
  "segments": [[2.5, 5.8], [7.1, 9.4]],
  "referring_expressions": ["person wearing a blue jacket", "red cube on the left", "yellow sphere in the center"]
}}

Please respond with ONLY the JSON object, no additional text."""
    
    return prompt

def load_data(csv_file, video_dir):
    """Load CSV data and prepare for processing."""
    print(f"📊 Loading {csv_file}...")
    df = pd.read_csv(csv_file).astype(str)
    
    data = []
    for _, row in df.iterrows():
        try:
            video_id = row['video_id']
            
            # Get entry_id if available
            entry_id = row.get('entry_id', None) if 'entry_id' in row else None
            if entry_id is None or entry_id == 'None' or str(entry_id).strip() == '':
                entry_id = f"video_{video_id}"
            
            video_path = os.path.join(video_dir, row['video_path'])
            
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"✗ Video not found: {video_path}")
                continue
            
            options = ast.literal_eval(row['candidates'])
            question = row['question']
            answer = row['answer']
            
            data.append({
                'entry_id': entry_id,
                'video_id': video_id,
                'video_path': video_path,
                'question': question,
                'options': options,
                'answer': answer
            })
        except Exception as e:
            print(f"❌ Error processing row {row.get('entry_id', 'unknown')}: {e}")
            continue
    
    print(f"📊 Loaded {len(data)} QA pairs")
    return data

def parse_list_response(response_text):
    """Parse list from response text (handles both nested lists and simple lists)"""
    try:
        # Remove any markdown code blocks
        text = response_text.strip()
        if "```python" in text:
            code_start = text.find("```python") + 9
            code_end = text.find("```", code_start)
            text = text[code_start:code_end].strip()
        elif "```" in text:
            code_start = text.find("```") + 3
            code_end = text.find("```", code_start)
            text = text[code_start:code_end].strip()
        
        # Try to find list pattern
        # Look for [[...]] or [...]
        import re
        # Find the first [ and last ]
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            list_text = text[start_idx:end_idx+1]
            # Use ast.literal_eval for safe evaluation
            import ast
            return ast.literal_eval(list_text)
        else:
            print(f"❌ No list found in response")
            return None
            
    except Exception as e:
        print(f"❌ List parsing error: {e}")
        print(f"Response text: {response_text[:500]}")
        return None


def parse_simultaneous_response(response_text):
    """Parse single-turn mode response in JSON format (with fallback to structured text)"""
    import json
    
    try:
        # First try to parse as JSON
        # Extract JSON from response if wrapped in markdown
        json_text = response_text.strip()
        if "```json" in json_text:
            json_start = json_text.find("```json") + 7
            json_end = json_text.find("```", json_start)
            json_text = json_text[json_start:json_end].strip()
        elif "```" in json_text:
            json_start = json_text.find("```") + 3
            json_end = json_text.find("```", json_start)
            json_text = json_text[json_start:json_end].strip()
        
        # Try to parse as JSON
        try:
            parsed_json = json.loads(json_text)
            
            # Handle array-wrapped JSON: [{"answer": ..., "segments": ..., ...}]
            if isinstance(parsed_json, list) and len(parsed_json) > 0:
                result = parsed_json[0]
            else:
                result = parsed_json
            
            # Validate required fields
            if isinstance(result, dict) and 'answer' in result and 'segments' in result and 'referring_expressions' in result:
                print(f"✅ Successfully parsed JSON response")
                return result
            else:
                print(f"⚠️ JSON missing required fields, trying fallback parsing")
        except json.JSONDecodeError:
            print(f"⚠️ JSON parsing failed, trying fallback parsing")
        
        # Fallback to structured text parsing
        result = {}
        
        # Extract answer
        answer_match = re.search(r'Answer:\s*([A-E])', response_text, re.IGNORECASE)
        if answer_match:
            result['answer'] = answer_match.group(1).upper()
        else:
            # Try to find just a letter at the beginning
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in response_text.upper():
                    result['answer'] = letter
                    break
            if 'answer' not in result:
                result['answer'] = None
        
        # Extract segments
        segments_match = re.search(r'Segments?:\s*(\[\[.*?\]\])', response_text, re.DOTALL | re.IGNORECASE)
        if segments_match:
            segments_text = segments_match.group(1)
            result['segments'] = ast.literal_eval(segments_text)
        else:
            # Try to find any nested list in the response
            nested_list_match = re.search(r'\[\[.*?\]\]', response_text, re.DOTALL)
            if nested_list_match:
                result['segments'] = ast.literal_eval(nested_list_match.group(0))
            else:
                result['segments'] = None
        
        # Extract referring expressions
        ref_expr_match = re.search(r'Referring Expressions?:\s*(\[.*?\])', response_text, re.DOTALL | re.IGNORECASE)
        if ref_expr_match:
            ref_expr_text = ref_expr_match.group(1)
            # Make sure we get the complete list
            bracket_count = 0
            end_pos = 0
            for i, char in enumerate(ref_expr_text):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break
            ref_expr_text = ref_expr_text[:end_pos]
            result['referring_expressions'] = ast.literal_eval(ref_expr_text)
        else:
            # Try to find a simple list (not nested) after segments
            # Look for the last list in the response
            lists = re.findall(r'\[[^\[\]]*\]', response_text)
            if lists:
                try:
                    # Try the last list that contains strings
                    for lst in reversed(lists):
                        parsed = ast.literal_eval(lst)
                        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                            result['referring_expressions'] = parsed
                            break
                    if 'referring_expressions' not in result:
                        result['referring_expressions'] = None
                except:
                    result['referring_expressions'] = None
            else:
                result['referring_expressions'] = None
        
        print(f"⚠️ Used fallback parsing")
        return result
        
    except Exception as e:
        print(f"❌ Single-turn response parsing error: {e}")
        print(f"Response text: {response_text[:500]}")
        return {'answer': None, 'segments': None, 'referring_expressions': None}

def process_video_three_stage(video_path, question, options, model, tokenizer, generation_config, fps=1.0, max_num=1):
    """
    Process video in three stages using multi-round conversation with history:
    1. Answer the question
    2. Generate temporal evidence (as nested list) - uses history from stage 1
    3. Generate referring expressions (as simple list) - uses history from stage 2
    
    Returns partial results if any stage fails (doesn't throw away successful stages)
    """
    # Initialize result dictionary
    result = {}
    
    # Load video
    pixel_values, num_patches_list = load_video(video_path, fps=fps, max_num=max_num)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Create video prefix for prompt (only needed for first round)
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    
    # Stage 1: Answer the question
    print("\n📝 Stage 1: Answering the question...")
    try:
        answer_prompt = create_answer_prompt(question, options)
        full_prompt = video_prefix + answer_prompt
        
        answer_response, history = model.chat(
            tokenizer, pixel_values, full_prompt, generation_config,
            num_patches_list=num_patches_list, history=None, return_history=True
        )
        print(f"Answer: {answer_response}")
        
        # Extract answer letter
        answer = answer_response.strip().upper()
        if len(answer) > 1:
            # Try to extract just the letter
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in answer:
                    answer = letter
                    break
        
        result['answer'] = answer
        print(f"✅ Stage 1 successful: {answer}")
    except Exception as e:
        print(f"❌ Stage 1 failed: {e}")
        result['answer'] = None
        # If answer fails, we can't continue meaningfully
        return result if result['answer'] else None
    
    # Stage 2: Generate temporal evidence (nested list format)
    # Use history from stage 1, so model remembers the video and question
    print("\n⏱️  Stage 2: Generating temporal evidence...")
    try:
        temporal_prompt = create_temporal_evidence_prompt(question, options, answer)
        
        temporal_response, history = model.chat(
            tokenizer, pixel_values, temporal_prompt, generation_config,
            num_patches_list=num_patches_list, history=history, return_history=True
        )
        print(f"Temporal evidence response: {temporal_response[:200]}...")
        
        # Parse temporal evidence as nested list
        evidence_segments = parse_list_response(temporal_response)
        if evidence_segments is None or not isinstance(evidence_segments, list):
            raise ValueError("Failed to parse temporal evidence as nested list")
        
        # Validate that it's a nested list of [start, end] pairs
        if not all(isinstance(seg, list) and len(seg) == 2 for seg in evidence_segments):
            raise ValueError(f"Invalid temporal evidence format: {evidence_segments}")
        
        print(f"Parsed temporal evidence: {evidence_segments}")
        result['segments'] = evidence_segments
        print(f"✅ Stage 2 successful")
    except Exception as e:
        print(f"❌ Stage 2 failed: {e}")
        result['segments'] = None
        # Continue to stage 3 even if stage 2 fails
    
    # Stage 3: Generate referring expressions (simple list format)
    # Use history from stage 2, so model remembers everything
    print("\n🎯 Stage 3: Generating referring expressions...")
    try:
        # Only proceed if we have segments, otherwise use empty list for prompt
        evidence_segments_for_prompt = result.get('segments', [])
        ref_expr_prompt = create_ref_expression_prompt(question, options, answer, evidence_segments_for_prompt)
        
        ref_expr_response, history = model.chat(
            tokenizer, pixel_values, ref_expr_prompt, generation_config,
            num_patches_list=num_patches_list, history=history, return_history=True
        )
        print(f"Referring expressions response: {ref_expr_response[:200]}...")
        
        # Parse referring expressions as simple list
        ref_expressions = parse_list_response(ref_expr_response)
        if ref_expressions is None or not isinstance(ref_expressions, list):
            raise ValueError("Failed to parse referring expressions as list")
        
        # Validate that it's a list of strings
        if not all(isinstance(expr, str) for expr in ref_expressions):
            raise ValueError(f"Invalid referring expressions format: {ref_expressions}")
        
        print(f"Parsed referring expressions: {ref_expressions}")
        result['referring_expressions'] = ref_expressions
        print(f"✅ Stage 3 successful")
    except Exception as e:
        print(f"❌ Stage 3 failed: {e}")
        result['referring_expressions'] = None
    
    # Return result with whatever we successfully got
    print(f"\n📊 Final result status: answer={result.get('answer') is not None}, "
          f"segments={result.get('segments') is not None}, "
          f"referring_expressions={result.get('referring_expressions') is not None}")
    
    return result


def process_video_simultaneous(video_path, question, options, model, tokenizer, generation_config, fps=1.0, max_num=1):
    """
    Process video in single-turn mode - answer all three questions at once.
    
    Returns result with all three fields or None for fields that failed to parse.
    """
    # Load video
    pixel_values, num_patches_list = load_video(video_path, fps=fps, max_num=max_num)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Create video prefix for prompt
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    
    print("\n📝 Single-turn mode: Asking all three questions at once (JSON format)...")
    try:
        # Create simultaneous prompt
        simultaneous_prompt = create_simultaneous_prompt(question, options)
        full_prompt = video_prefix + simultaneous_prompt
        
        # Get response
        response, _ = model.chat(
            tokenizer, pixel_values, full_prompt, generation_config,
            num_patches_list=num_patches_list, history=None, return_history=True
        )
        print(f"Response: {response[:500]}...")
        
        # Parse simultaneous response
        result = parse_simultaneous_response(response)
        
        # Report what was successfully parsed
        successful_fields = []
        if result.get('answer') is not None:
            successful_fields.append('answer')
        if result.get('segments') is not None:
            successful_fields.append('segments')
        if result.get('referring_expressions') is not None:
            successful_fields.append('referring_expressions')
        
        if len(successful_fields) == 3:
            print(f"✅ All fields successfully parsed")
        elif len(successful_fields) > 0:
            print(f"⚠️  Partial success (parsed: {', '.join(successful_fields)})")
        else:
            print(f"❌ Failed to parse any fields")
        
        print(f"\n📊 Result: answer={result.get('answer')}, "
              f"segments={result.get('segments') is not None}, "
              f"referring_expressions={result.get('referring_expressions') is not None}")
        
        return result
        
    except Exception as e:
        print(f"❌ Single-turn mode failed: {e}")
        return {'answer': None, 'segments': None, 'referring_expressions': None}


def process_batch_simultaneous(batch_items, model, tokenizer, generation_config, fps=1.0, max_num=1):
    """
    Process a batch of videos in single-turn mode.
    
    Args:
        batch_items: List of items to process in batch
        
    Returns:
        List of results corresponding to batch_items
    """
    # Initialize results list with None for all items
    results = [None] * len(batch_items)
    batch_pixel_values = []
    batch_num_patches_lists = []
    batch_prompts = []
    batch_indices = []  # Track which items are valid
    
    print(f"\n📦 Batch inference: Loading {len(batch_items)} videos...")
    
    # Load all videos in the batch
    for i, item in enumerate(batch_items):
        try:
            pixel_values, num_patches_list = load_video(item['video_path'], fps=fps, max_num=max_num)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # Create video prefix and prompt
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            simultaneous_prompt = create_simultaneous_prompt(item['question'], item['options'])
            full_prompt = video_prefix + simultaneous_prompt
            
            batch_pixel_values.append(pixel_values)
            batch_num_patches_lists.append(num_patches_list)
            batch_prompts.append(full_prompt)
            batch_indices.append(i)
            
        except Exception as e:
            print(f"❌ Error loading video {item['entry_id']}: {e}")
            results[i] = {'answer': None, 'segments': None, 'referring_expressions': None}
    
    # Process batch
    print(f"🔄 Batch inference: Processing {len(batch_indices)} valid videos...")
    
    for batch_idx, (pixel_values, num_patches_list, prompt) in enumerate(zip(batch_pixel_values, batch_num_patches_lists, batch_prompts)):
        i = batch_indices[batch_idx]
        item = batch_items[i]
        
        try:
            # Get response for this item
            response, _ = model.chat(
                tokenizer, pixel_values, prompt, generation_config,
                num_patches_list=num_patches_list, history=None, return_history=True
            )
            
            # Parse simultaneous response
            result = parse_simultaneous_response(response)
            results[i] = result
            
            # Report what was successfully parsed
            successful_fields = []
            if result.get('answer') is not None:
                successful_fields.append('answer')
            if result.get('segments') is not None:
                successful_fields.append('segments')
            if result.get('referring_expressions') is not None:
                successful_fields.append('referring_expressions')
            
            if len(successful_fields) == 3:
                print(f"✅ [{i+1}/{len(batch_items)}] {item['entry_id']}: All fields parsed")
            elif len(successful_fields) > 0:
                print(f"⚠️  [{i+1}/{len(batch_items)}] {item['entry_id']}: Partial ({', '.join(successful_fields)})")
            else:
                print(f"❌ [{i+1}/{len(batch_items)}] {item['entry_id']}: Failed to parse")
            
        except Exception as e:
            print(f"❌ Error processing {item['entry_id']}: {e}")
            results[i] = {'answer': None, 'segments': None, 'referring_expressions': None}
    
    return results


def main():
    parser = argparse.ArgumentParser(description="InternVL3.5 Spatial-Temporal Evidence Analysis")
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3_5-8B',
                        help='Model name or path')
    parser.add_argument('--data-file', type=str,
                        default='data/st_evidence_gen.csv',
                        help='CSV file with video data')
    parser.add_argument('--video-dir', type=str,
                        default='data/videos_6fps',
                        help='Video directory')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second to sample (default: 1.0)')
    parser.add_argument('--max-num', type=int, default=1,
                        help='Maximum number of patches per frame (default: 1)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start index for processing (default: 0)')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='End index for processing (default: None, process all)')
    parser.add_argument('--mode', type=str, choices=['single-turn', 'multi-turn'], default='single-turn',
                        help='Processing mode: single-turn (all at once) or multi-turn (3 separate rounds)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for inference (default: 16, only works for single-turn mode)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.data_file, args.video_dir)
    if not data:
        print("❌ No data to process")
        return
    
    # Filter data by indices
    if args.end_idx is not None:
        data = data[args.start_idx:args.end_idx]
    else:
        data = data[args.start_idx:]
    
    print(f"📊 Processing {len(data)} videos (indices {args.start_idx} to {args.start_idx + len(data)})")
    print(f"🔧 Mode: {args.mode}")
    if args.mode == 'single-turn' and args.batch_size > 1:
        print(f"📦 Batch size: {args.batch_size}")
    
    # Setup results file
    model_short_name = args.model.split('/')[-1].replace('-', '_').lower()
    mode_suffix = 'single' if args.mode == 'single-turn' else 'multi'
    save_file = f'results/internvl/{model_short_name}_st_evidence_{mode_suffix}_{args.fps}fps.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    print(f"💾 Results will be saved to: {save_file}")
    
    # Load existing results if available
    result_dict = {}
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            result_dict = json.load(f)
        print(f"📂 Loaded existing results: {len(result_dict)} entries")
    
    # Model initialization
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
    
    # Check if batch processing is enabled
    use_batch = args.mode == 'single-turn' and args.batch_size > 1
    
    if use_batch:
        print(f"\n⚡ Using batch inference with batch size {args.batch_size}")
    
    # Filter out already processed items (only skip complete results)
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
                    parsed_response.get('referring_expressions') is not None):
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
    
    print(f"📊 {len(items_to_process)} items to process (skipping {skipped_count} complete, retrying {retry_partial_count} partial)")
    
    # Process videos
    processed_count = 0
    
    if use_batch:
        # Batch processing for single-turn mode
        num_batches = (len(items_to_process) + args.batch_size - 1) // args.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, len(items_to_process))
            batch_items = items_to_process[batch_start:batch_end]
            
            print(f"\n{'='*80}")
            print(f"Batch {batch_idx+1}/{num_batches}: Processing items {batch_start} to {batch_end-1}")
            
            try:
                # Process batch
                batch_results = process_batch_simultaneous(
                    batch_items,
                    model,
                    tokenizer,
                    generation_config,
                    fps=args.fps,
                    max_num=args.max_num
                )
                
                # Save results for each item in batch
                for item, result in zip(batch_items, batch_results):
                    entry_id = item['entry_id']
                    
                    if result and result.get('answer') is not None:
                        result_dict[entry_id] = {
                            'video_id': item['video_id'],
                            'parsed_response': result
                        }
                        
                        successful_stages = []
                        if result.get('answer') is not None:
                            successful_stages.append('answer')
                        if result.get('segments') is not None:
                            successful_stages.append('segments')
                        if result.get('referring_expressions') is not None:
                            successful_stages.append('referring_expressions')
                        
                        if len(successful_stages) == 3:
                            print(f"✅ Successfully processed {entry_id} (all stages)")
                        else:
                            print(f"⚠️  Partially processed {entry_id} (successful: {', '.join(successful_stages)})")
                    else:
                        result_dict[entry_id] = {
                            'video_id': item['video_id'],
                            'parsed_response': None,
                            'parse_error': True
                        }
                        print(f"❌ Failed to process {entry_id} (no answer)")
                    
                    processed_count += 1
                
                # Save after each batch
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx+1}: {e}")
                for item in batch_items:
                    result_dict[item['entry_id']] = {
                        'video_id': item['video_id'],
                        'parsed_response': None,
                        'error': str(e)
                    }
                continue
    else:
        # Sequential processing (for multi-turn mode or batch_size=1)
        for item in tqdm(items_to_process):
            entry_id = item['entry_id']
            
            print(f"\n{'='*80}")
            print(f"Processing: {entry_id}")
            print(f"Video: {Path(item['video_path']).name}")
            print(f"Question: {item['question']}")
            print(f"Options: {item['options']}")
            
            try:
                # Select processing function based on mode
                if args.mode == 'multi-turn':
                    result = process_video_three_stage(
                        item['video_path'],
                        item['question'],
                        item['options'],
                        model,
                        tokenizer,
                        generation_config,
                        fps=args.fps,
                        max_num=args.max_num
                    )
                else:  # single-turn
                    result = process_video_simultaneous(
                        item['video_path'],
                        item['question'],
                        item['options'],
                        model,
                        tokenizer,
                        generation_config,
                        fps=args.fps,
                        max_num=args.max_num
                    )
                
                if result and result.get('answer') is not None:
                    # Save result with partial data (some fields may be None)
                    result_dict[entry_id] = {
                        'video_id': item['video_id'],
                        'parsed_response': result
                    }
                    
                    # Report which stages succeeded
                    successful_stages = []
                    if result.get('answer') is not None:
                        successful_stages.append('answer')
                    if result.get('segments') is not None:
                        successful_stages.append('segments')
                    if result.get('referring_expressions') is not None:
                        successful_stages.append('referring_expressions')
                    
                    if len(successful_stages) == 3:
                        print(f"✅ Successfully processed {entry_id} (all stages)")
                    else:
                        print(f"⚠️  Partially processed {entry_id} (successful: {', '.join(successful_stages)})")
                else:
                    result_dict[entry_id] = {
                        'video_id': item['video_id'],
                        'parsed_response': None,
                        'parse_error': True
                    }
                    print(f"❌ Failed to process {entry_id} (no answer)")
                
                processed_count += 1
                
                # Save after each video
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
                
                if processed_count % 10 == 0:
                    valid_count = sum(1 for v in result_dict.values() 
                                    if v.get('parsed_response') is not None)
                    complete_count = sum(1 for v in result_dict.values() 
                                       if v.get('parsed_response') and 
                                       all(v['parsed_response'].get(k) is not None 
                                           for k in ['answer', 'segments', 'referring_expressions']))
                    print(f"\n📊 Progress: {processed_count} processed, {valid_count} valid responses ({complete_count} complete)")
            
            except Exception as e:
                print(f"❌ Error processing {entry_id}: {e}")
                result_dict[entry_id] = {
                    'video_id': item['video_id'],
                    'parsed_response': None,
                    'error': str(e)
                }
                continue
    
    # Final statistics
    print(f"\n{'='*80}")
    print("🎉 Processing completed!")
    
    total_entries = len(result_dict)
    valid_entries = sum(1 for v in result_dict.values() if v.get('parsed_response') is not None)
    failed_entries = total_entries - valid_entries
    
    # Count complete vs partial results
    complete_entries = sum(1 for v in result_dict.values() 
                          if v.get('parsed_response') and 
                          all(v['parsed_response'].get(k) is not None 
                              for k in ['answer', 'segments', 'referring_expressions']))
    partial_entries = valid_entries - complete_entries
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"📊 Total entries: {total_entries}")
    print(f"✅ Valid responses: {valid_entries}")
    print(f"   - Complete (all 3 stages): {complete_entries}")
    print(f"   - Partial (some stages): {partial_entries}")
    print(f"❌ Failed responses: {failed_entries}")
    print(f"📍 Success rate: {(valid_entries/total_entries)*100:.1f}%" if total_entries > 0 else "📍 Success rate: 0%")
    print(f"📍 Complete rate: {(complete_entries/total_entries)*100:.1f}%" if total_entries > 0 else "📍 Complete rate: 0%")
    print(f"💾 Results saved to: {save_file}")

if __name__ == '__main__':
    main()
