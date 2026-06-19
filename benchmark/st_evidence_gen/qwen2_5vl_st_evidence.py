#!/usr/bin/env python3
"""
Spatial-Temporal Evidence Analysis using Qwen2.5-VL with VLLM
Three-stage questioning: answer, temporal segments, referring expressions
Supports multi-turn and single-turn modes

Official Qwen2.5-VL parameters:
- Sampling: temperature=0.1, top_p=0.001, repetition_penalty=1.05
- Video: max_frames=768, fps configurable

Supported models:
- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-72B-Instruct

Usage:
python qwen2_5vl_st_evidence.py --model Qwen/Qwen2.5-VL-7B-Instruct --mode multi-turn --fps 1 --batch-size 8
python qwen2_5vl_st_evidence.py --model Qwen/Qwen2.5-VL-7B-Instruct --mode single-turn --fps 1 --batch-size 8
"""

import os
import json
import pandas as pd
import ast
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from vllm import LLM, SamplingParams


def create_answer_prompt(question, options):
    """Create prompt for stage 1: Answer the question"""
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

Task: Select the correct option (A, B, C, D, or E) based on the video. Provide ONLY the single letter answer, no additional text or explanation."""
    
    return prompt


def create_temporal_evidence_prompt(question, options, answer):
    """Create prompt for stage 2: Identify temporal evidence"""
    prompt = f"""Based on your previous answer ({answer}), identify the essential time segments (in seconds) in the video that are highly related and required to answer this question.

These should be the specific moments where key events, interactions, or changes occur that directly support your answer.

Output format: Provide a nested list in the format [[start1, end1], [start2, end2], ...]

For example:
[[2.5, 5.8], [7.1, 9.4]]

Only output the nested list, no additional text or explanation."""
    
    return prompt


def create_ref_expression_prompt(question, options, answer, evidence_segments):
    """Create prompt for stage 3: Generate referring expressions"""
    prompt = f"""Based on your previous answer ({answer}) and the identified time segments {evidence_segments}, identify all the objects or regions in the video that serve as evidence for answering the question.

These should be:
- Objects related to the QUESTION (objects that help understand what is being asked)
- Objects related to the ANSWER (objects that directly support or constitute the answer)

REFERRING EXPRESSION GUIDELINES:
- Keep expressions simple but discriminative (e.g., "red cube" or "red cube on left")
- Use basic descriptors: color, shape, size, materials, actions, simple position (left/right/center/top/bottom), relative position (left of, right of, behind, in front of, etc.)
- Avoid complex relationships or temporal descriptions
- Make them clear enough for grounding models to locate objects
- Focus only on objects/regions essential for answering the question

Output format: Provide a list of referring expressions in the format ["expression 1", "expression 2", ...]

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


def parse_list_response(response_text):
    """Parse list response (for segments or referring expressions)"""
    try:
        # Extract list from response if wrapped in markdown
        text = response_text.strip()
        if "```" in text:
            # Extract content between code blocks
            start = text.find("```")
            end = text.rfind("```")
            if start != end:
                text = text[start+3:end].strip()
                # Remove language identifier if present
                if text.startswith("python") or text.startswith("json"):
                    text = text.split('\n', 1)[1].strip()
        
        # Find list pattern
        list_match = re.search(r'\[.*\]', text, re.DOTALL)
        if list_match:
            list_text = list_match.group(0)
            parsed = ast.literal_eval(list_text)
            return parsed
        
        print(f"⚠️ Could not find list pattern in response")
        return None
        
    except Exception as e:
        print(f"❌ List parsing error: {e}")
        print(f"Response text: {response_text[:500]}")
        return None


def parse_simultaneous_response(response_text):
    """Parse single-turn mode response in JSON format (with fallback to structured text)"""
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


def load_data(csv_file, video_dir):
    """Load CSV data and prepare for processing."""
    print(f"📊 Loading {csv_file}...")
    df = pd.read_csv(csv_file).astype(str)
    
    data = []
    for _, row in df.iterrows():
        try:
            video_id = row['video_id']
            
            # Get entry_id if available, otherwise use video_id as fallback
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
            
            data.append({
                'entry_id': entry_id,
                'video_id': video_id,
                'video_path': video_path,
                'question': question,
                'options': options
            })
        except Exception as e:
            print(f"❌ Error processing row {row.get('entry_id', 'unknown')}: {e}")
            continue
    
    print(f"📊 Loaded {len(data)} QA pairs")
    return data


def process_single_turn_batch(videos, prompts, llm, processor, sampling_params, fps, max_frames):
    """Process batch in single-turn mode"""
    messages = []
    
    for video_path, prompt in zip(videos, prompts):
        # Create conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": fps, "max_frames": max_frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        messages.append(conversation)
    
    # Process each message individually to get correct prompts
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    _, video_inputs, _ = process_vision_info(conversations=messages, return_video_kwargs=True)
    
    llm_inputs = []
    for i in range(len(video_inputs)):
        llm_inputs.append({
            "prompt": texts[i],
            "multi_modal_data": {"video": video_inputs[i]},
        })
    
    # Generate responses
    outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
    
    return [output.outputs[0].text for output in outputs]


def process_multi_turn_item(video_path, question, options, llm, processor, sampling_params, fps, max_frames):
    """Process single item in multi-turn mode"""
    result = {'answer': None, 'segments': None, 'referring_expressions': None}
    
    try:
        # Stage 1: Answer the question
        print("  📝 Stage 1: Answering question...")
        answer_prompt = create_answer_prompt(question, options)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": fps, "max_frames": max_frames},
                    {"type": "text", "text": answer_prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        _, video_inputs, _ = process_vision_info(conversations=[conversation], return_video_kwargs=True)
        
        llm_input = {
            "prompt": text,
            "multi_modal_data": {"video": video_inputs},
        }
        
        output = llm.generate([llm_input], sampling_params=sampling_params)
        answer_response = output[0].outputs[0].text.strip()
        
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
            print(f"  ⚠️ Failed to extract answer from: {answer_response[:100]}")
            return result
        
        # Stage 2: Get temporal evidence
        print("  📝 Stage 2: Identifying temporal evidence...")
        temporal_prompt = create_temporal_evidence_prompt(question, options, answer)
        
        # Add assistant's answer and new user question
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": answer_response}]})
        conversation.append({"role": "user", "content": [{"type": "text", "text": temporal_prompt}]})
        
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        llm_input = {
            "prompt": text,
            "multi_modal_data": {"video": video_inputs},
        }
        
        output = llm.generate([llm_input], sampling_params=sampling_params)
        segments_response = output[0].outputs[0].text
        
        # Parse segments
        segments = parse_list_response(segments_response)
        if segments:
            result['segments'] = segments
            print(f"  ✅ Segments: {segments}")
        else:
            print(f"  ⚠️ Failed to parse segments from: {segments_response[:100]}")
            return result
        
        # Stage 3: Get referring expressions
        print("  📝 Stage 3: Generating referring expressions...")
        ref_expr_prompt = create_ref_expression_prompt(question, options, answer, segments)
        
        # Add assistant's segments and new user question
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": segments_response}]})
        conversation.append({"role": "user", "content": [{"type": "text", "text": ref_expr_prompt}]})
        
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        llm_input = {
            "prompt": text,
            "multi_modal_data": {"video": video_inputs},
        }
        
        output = llm.generate([llm_input], sampling_params=sampling_params)
        ref_expr_response = output[0].outputs[0].text
        
        # Parse referring expressions
        ref_expressions = parse_list_response(ref_expr_response)
        if ref_expressions:
            result['referring_expressions'] = ref_expressions
            print(f"  ✅ Referring expressions: {ref_expressions}")
        else:
            print(f"  ⚠️ Failed to parse referring expressions from: {ref_expr_response[:100]}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Multi-turn processing error: {e}")
        return result


def main():
    # Fix CUDA multiprocessing issue
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    parser = argparse.ArgumentParser(description="Three-stage spatial-temporal evidence analysis with Qwen2.5-VL")
    parser.add_argument('--model', type=str, 
                        choices=['Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'],
                        default='Qwen/Qwen2.5-VL-7B-Instruct',
                        help='Model name')
    parser.add_argument('--data-file', type=str,
                        default='data/st_evidence_gen.csv',
                        help='CSV file with video data')
    parser.add_argument('--video-dir', type=str, 
                        default='data/videos_6fps',
                        help='Video directory')
    parser.add_argument('--fps', type=int, default=1, 
                        help='Frames per second (default: 1)')
    parser.add_argument('--max-frames', type=int, default=768, 
                        help='Max frames for video sampling (default: 768)')
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='Batch size (only for single-turn mode)')
    parser.add_argument('--gpu-num', type=int, default=1, 
                        help='Number of GPUs for tensor parallelism (default: 1)')
    parser.add_argument('--mode', type=str, choices=['single-turn', 'multi-turn'], 
                        default='single-turn',
                        help='Processing mode: single-turn (all at once) or multi-turn (sequential)')
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(args.data_file, args.video_dir)
    if not data:
        print("❌ No data to process")
        return
    
    print(f"📊 Total videos to process: {len(data)}")
    print(f"🔧 Mode: {args.mode}")
    
    # Setup results file
    model_short_name = args.model.split('/')[-1].replace('-', '_').lower()
    save_file = f'results/qwen2_5vl/{model_short_name}_st_evidence_{args.mode.replace("-", "_")}_{args.fps}fps.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    print(f"💾 Results will be saved to: {save_file}")
    
    # Load existing results if available
    result_dict = {}
    if os.path.exists(save_file):
        result_dict = json.load(open(save_file, 'r'))
        print(f"📂 Loaded existing results: {len(result_dict)} entries")
    
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
    
    if not items_to_process:
        print("✅ All done!")
        return
    
    # Load model - Official Qwen2.5-VL configuration
    MODEL_PATH = args.model
    print(f"🔄 Loading {MODEL_PATH}...")
    
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 1, "video": 1},
        tensor_parallel_size=args.gpu_num,
        trust_remote_code=True,
    )
    
    # Official Qwen2.5-VL sampling parameters with higher max_tokens for our task
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=4096,  # Higher for structured output
        stop_token_ids=[],
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Process based on mode
    processed_count = 0
    
    if args.mode == 'single-turn':
        # Batch processing for single-turn mode
        print(f"\n⚡ Single-turn mode: Processing in batches of {args.batch_size}")
        
        for i in tqdm(range(0, len(items_to_process), args.batch_size)):
            batch = items_to_process[i:i + args.batch_size]
            videos = [item['video_path'] for item in batch]
            prompts = [create_simultaneous_prompt(item['question'], item['options']) for item in batch]
            
            try:
                # Generate responses
                responses = process_single_turn_batch(videos, prompts, llm, processor, 
                                                     sampling_params, args.fps, args.max_frames)
                
                # Process each response
                for item, response_text in zip(batch, responses):
                    entry_id = item['entry_id']
                    video_id = item['video_id']
                    
                    print(f"\n{'='*80}")
                    print(f"Processing: {entry_id}")
                    print(f"Video: {Path(item['video_path']).name}")
                    print(f"Question: {item['question']}")
                    
                    # Parse response
                    parsed_result = parse_simultaneous_response(response_text)
                    
                    # Store result
                    result_dict[entry_id] = {
                        'video_id': video_id,
                        'parsed_response': parsed_result
                    }
                    
                    # Report status
                    successful = [k for k, v in parsed_result.items() if v is not None]
                    if len(successful) == 3:
                        print(f"✅ Successfully processed {entry_id}")
                    else:
                        print(f"⚠️ Partially processed {entry_id} (successful: {', '.join(successful)})")
                    
                    processed_count += 1
                
                # Save after each batch
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
            
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                continue
    
    else:
        # Sequential processing for multi-turn mode
        print(f"\n🔄 Multi-turn mode: Processing sequentially")
        
        for item in tqdm(items_to_process):
            entry_id = item['entry_id']
            video_id = item['video_id']
            
            print(f"\n{'='*80}")
            print(f"Processing: {entry_id}")
            print(f"Video: {Path(item['video_path']).name}")
            print(f"Question: {item['question']}")
            
            try:
                # Process in multi-turn mode
                parsed_result = process_multi_turn_item(
                    item['video_path'],
                    item['question'],
                    item['options'],
                    llm,
                    processor,
                    sampling_params,
                    args.fps,
                    args.max_frames
                )
                
                # Store result
                result_dict[entry_id] = {
                    'video_id': video_id,
                    'parsed_response': parsed_result
                }
                
                # Report status
                successful = [k for k, v in parsed_result.items() if v is not None]
                if len(successful) == 3:
                    print(f"✅ Successfully processed {entry_id}")
                else:
                    print(f"⚠️ Partially processed {entry_id} (successful: {', '.join(successful)})")
                
                processed_count += 1
                
                # Save after each item
                with open(save_file, 'w') as f:
                    json.dump(result_dict, f, indent=2)
            
            except Exception as e:
                print(f"❌ Error processing {entry_id}: {e}")
                continue
    
    print("\n🎉 Processing completed!")
    
    # Calculate final statistics
    final_total = len(result_dict)
    complete_count = sum(1 for entry in result_dict.values() 
                        if entry.get('parsed_response') and
                        all(entry['parsed_response'].get(k) is not None 
                            for k in ['answer', 'segments', 'referring_expressions']))
    partial_count = sum(1 for entry in result_dict.values()
                       if entry.get('parsed_response') and
                       any(entry['parsed_response'].get(k) is None 
                           for k in ['answer', 'segments', 'referring_expressions']))
    failed_count = final_total - complete_count - partial_count
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"📊 Total entries: {final_total}")
    print(f"✅ Complete responses: {complete_count}")
    print(f"⚠️  Partial responses: {partial_count}")
    print(f"❌ Failed responses: {failed_count}")
    print(f"📍 Success rate: {(complete_count/final_total)*100:.1f}%" if final_total > 0 else "📍 Success rate: 0%")
    print(f"💾 Results saved to: {save_file}")


if __name__ == "__main__":
    main()

