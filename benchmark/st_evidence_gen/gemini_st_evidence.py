#!/usr/bin/env python3
"""
Spatial-Temporal Evidence Analysis using Gemini 2.5 Pro
Three-stage multi-round conversation approach:
1. Answer the question
2. Generate temporal evidence (as nested list)
3. Generate referring expressions (as simple list)
"""

import os
import time
import json
import pandas as pd
import ast
import re
import argparse
from pathlib import Path
from tqdm import tqdm

from google import genai
from google.genai.types import Part, Blob, GenerateContentConfig, VideoMetadata


def initialize_gemini(project_id: str = "salesforce-research-internal", location: str = "us-central1"):
    """Initialize Gemini client with Vertex AI."""
    try:
        # Initialize genai client with Vertex AI
        client = genai.Client(vertexai=True, project=project_id, location=location)

        # Create generation config
        config = GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
        )
        
        print(f"✓ Successfully initialized Gemini client for project: {project_id}")
        return client, config
    except Exception as e:
        print(f"✗ Failed to initialize Gemini client: {e}")
        return None, None


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
    """Create prompt for single-turn mode - answer all three questions at once in JSON format"""
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

Output format: Return a JSON object with the following structure:

{{
  "answer": "single letter A/B/C/D/E",
  "segments": [[start1, end1], [start2, end2], ...],
  "referring_expressions": ["expression 1", "expression 2", ...]
}}

Example output:

{{
  "answer": "A",
  "segments": [[2.5, 5.8], [7.1, 9.4]],
  "referring_expressions": ["person wearing a blue jacket", "red cube on the left", "yellow sphere in the center"]
}}

Please provide your response in JSON format only, no additional text or explanation."""
    
    return prompt


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
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            list_text = text[start_idx:end_idx+1]
            # Use ast.literal_eval for safe evaluation
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


def process_video_three_stage(client, config, model_name, video_path, question, options, fps=1):
    """
    Process video in three stages using multi-round conversation with Gemini.
    Retries with higher token limit if first attempt fails.
    """
    # First try with normal config
    result = _process_video_three_stage_attempt(client, config, model_name, video_path, question, options, fps)
    
    if result is not None and result.get('answer') is not None:
        return result
    
    # If failed, retry with higher token limit
    print(f"⚠️ Retrying multi-turn with higher tokens (9218)")
    high_config = GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=9218,
    )
    
    return _process_video_three_stage_attempt(client, high_config, model_name, video_path, question, options, fps)


def _process_video_three_stage_attempt(client, config, model_name, video_path, question, options, fps=1):
    """
    Single attempt to process video in three stages using multi-round conversation with Gemini:
    1. Answer the question
    2. Generate temporal evidence (as nested list) - uses history from stage 1
    3. Generate referring expressions (as simple list) - uses history from stage 2
    
    Returns partial results if any stage fails (doesn't throw away successful stages)
    """
    # Initialize result dictionary
    result = {}
    
    # Load video
    mime_type = get_mime_type(video_path)
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    blob = Blob(mime_type=mime_type, data=video_data)
    video_part = Part(inlineData=blob, videoMetadata=VideoMetadata(fps=fps))
    
    # Start chat session with video
    chat = client.chats.create(model=model_name, config=config)
    
    # Stage 1: Answer the question
    print("\n📝 Stage 1: Answering the question...")
    try:
        answer_prompt = create_answer_prompt(question, options)
        
        # Send video + question in first message
        response = chat.send_message([video_part, answer_prompt])
        answer_response = response.text
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
    # Continue the same chat session (has history)
    print("\n⏱️  Stage 2: Generating temporal evidence...")
    try:
        temporal_prompt = create_temporal_evidence_prompt(question, options, answer)
        
        response = chat.send_message(temporal_prompt)
        temporal_response = response.text
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
    # Continue the same chat session (has full history)
    print("\n🎯 Stage 3: Generating referring expressions...")
    try:
        evidence_segments_for_prompt = result.get('segments', [])
        ref_expr_prompt = create_ref_expression_prompt(question, options, answer, evidence_segments_for_prompt)
        
        response = chat.send_message(ref_expr_prompt)
        ref_expr_response = response.text
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


def process_video_simultaneous(client, config, model_name, video_path, question, options, fps=1):
    """
    Process video in single-turn mode - answer all three questions at once.
    Retries with higher token limit if first attempt fails.
    """
    # First try with normal config
    result = _process_video_simultaneous_attempt(client, config, model_name, video_path, question, options, fps)
    
    if result is not None and result.get('answer') is not None:
        return result
    
    # If failed, retry with higher token limit
    print(f"⚠️ Retrying single-turn with higher tokens (9218)")
    high_config = GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=9218,
    )
    
    return _process_video_simultaneous_attempt(client, high_config, model_name, video_path, question, options, fps)


def _process_video_simultaneous_attempt(client, config, model_name, video_path, question, options, fps=1):
    """
    Single attempt to process video in single-turn mode - answer all three questions at once.
    
    Returns result with all three fields or None for fields that failed to parse.
    """
    # Load video
    mime_type = get_mime_type(video_path)
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    blob = Blob(mime_type=mime_type, data=video_data)
    video_part = Part(inlineData=blob, videoMetadata=VideoMetadata(fps=fps))
    
    print("\n📝 Single-turn mode: Asking all three questions at once (JSON format)...")
    try:
        # Create simultaneous prompt
        simultaneous_prompt = create_simultaneous_prompt(question, options)
        
        # Create JSON-specific config for single-turn mode
        json_config = GenerateContentConfig(
            temperature=config.temperature if hasattr(config, 'temperature') else 0.1,
            max_output_tokens=config.max_output_tokens if hasattr(config, 'max_output_tokens') else 4096,
            response_mime_type="application/json"
        )
        
        # Generate response using model (not chat for single request)
        response = client.models.generate_content(
            model=model_name,
            contents=[video_part, simultaneous_prompt],
            config=json_config
        )
        response_text = response.text
        print(f"Response: {response_text[:500]}...")
        
        # Parse simultaneous response
        result = parse_simultaneous_response(response_text)
        
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


def main():
    parser = argparse.ArgumentParser(description="Gemini 2.5 Pro Spatial-Temporal Evidence Analysis")
    parser.add_argument('--model', type=str, default='gemini-2.5-pro',
                        help='Model name (gemini-2.5-pro or gemini-2.5-flash)')
    parser.add_argument('--project-id', type=str, default='salesforce-research-internal',
                        help='GCP project ID')
    parser.add_argument('--location', type=str, default='us-central1',
                        help='GCP location')
    parser.add_argument('--data-file', type=str,
                        default='data/st_evidence_gen.csv',
                        help='CSV file with video data')
    parser.add_argument('--video-dir', type=str,
                        default='data/videos_6fps',
                        help='Video directory')
    parser.add_argument('--fps', type=int, default=1,
                        help='Frames per second for video processing')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Start index for processing (default: 0)')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='End index for processing (default: None, process all)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--mode', type=str, choices=['single-turn', 'multi-turn'], default='single-turn',
                        help='Processing mode: single-turn (all at once) or multi-turn (3 separate rounds)')
    
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
    
    # Setup results file
    model_short_name = args.model.replace('-', '_').replace('.', '_').lower()
    mode_suffix = 'single' if args.mode == 'single-turn' else 'multi'
    save_file = f'results/gemini/{model_short_name}_st_evidence_{mode_suffix}_{args.fps}fps.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    print(f"💾 Results will be saved to: {save_file}")
    
    # Load existing results if available
    result_dict = {}
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            result_dict = json.load(f)
        print(f"📂 Loaded existing results: {len(result_dict)} entries")
    
    # Initialize Gemini client
    print(f"\n🔄 Initializing Gemini client...")
    client, config = initialize_gemini(project_id=args.project_id, location=args.location)
    if client is None:
        print("❌ Failed to initialize Gemini client")
        return
    
    # Process each video
    processed_count = 0
    for item in tqdm(data):
        entry_id = item['entry_id']
        
        # Skip if already processed successfully with complete results
        if entry_id in result_dict:
            parsed_response = result_dict[entry_id].get('parsed_response')
            if parsed_response is not None:
                # Check if all three fields are present and valid
                if (parsed_response.get('answer') is not None and 
                    parsed_response.get('segments') is not None and 
                    parsed_response.get('referring_expressions') is not None):
                    print(f"⏭️  Skipping {entry_id} (already processed with complete results)")
                    continue
                else:
                    print(f"🔄 Retrying {entry_id} (has partial results)")
        
        print(f"\n{'='*80}")
        print(f"Processing: {entry_id}")
        print(f"Video: {Path(item['video_path']).name}")
        print(f"Question: {item['question']}")
        print(f"Options: {item['options']}")
        
        try:
            # Select processing function based on mode
            if args.mode == 'multi-turn':
                result = process_video_three_stage(
                    client,
                    config,
                    args.model,
                    item['video_path'],
                    item['question'],
                    item['options'],
                    fps=args.fps
                )
            else:  # single-turn
                result = process_video_simultaneous(
                    client,
                    config,
                    args.model,
                    item['video_path'],
                    item['question'],
                    item['options'],
                    fps=args.fps
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
            
            # Rate limiting
            time.sleep(args.delay)
        
        except Exception as e:
            print(f"❌ Error processing {entry_id}: {e}")
            result_dict[entry_id] = {
                'video_id': item['video_id'],
                'parsed_response': None,
                'error': str(e)
            }
            # Save after error
            with open(save_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
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

