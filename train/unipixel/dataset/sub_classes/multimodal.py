# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy
import random

import nncore
from torch.utils.data import Dataset

from unipixel.dataset.hybrid import DATASETS

GROUNDED_QA_PROMPTS = [
    '{} \nAnswer the question and provide temporal evidence in the form of [[start1, end1], [start2, end2], ...].',
    '{} \nAnswer the question and provide temporal evidence. The temporal evidence should be in the form of [[start1, end1], [start2, end2], ...].',
    '{} \nProvide your answer along with supporting evidence: temporal segments in the format [[start1, end1], [start2, end2], ...].',
    '{} \nAnswer with evidence. Include time segments as [[start_sec, end_sec], ...].',
    '{} \nPlease answer and identify when (temporal evidence: [[start, end], ...]) the relevant information appears.',
]

class MultimodalDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args, repeat=1):
        super().__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = len(anno['conversations'][1]['value'].split(' '))
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if data_args.min_video_len >= 0 and anno.get('duration', float('inf')) < data_args.min_video_len:
                continue
            if data_args.max_video_len >= 0 and anno.get('duration', 0) > data_args.max_video_len:
                continue
            annos.append(anno)

        self.annos = annos
        self.raw_length = len(raw_annos)
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.repeat = repeat

    def __len__(self):
        return len(self.annos) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.annos)

        anno = copy.deepcopy(self.annos[idx])

        assert not ('image' in anno and 'video' in anno), anno

        assert anno['conversations'][0]['from'] == 'human'
        init_prompt = anno['conversations'][0]['value']

        for key in ('<image>', '<video>'):
            if init_prompt.startswith(f'{key}\n'):
                init_prompt = init_prompt[len(f'{key}\n'):]
            if init_prompt.endswith(f'\n{key}'):
                init_prompt = init_prompt[:-len(f'\n{key}')]

        if 'image' in anno:
            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'image',
                    'image': nncore.join(self.IMAGE_ROOT, anno['image']),
                    'min_pixels': 128 * 28 * 28,
                    'max_pixels': 2048 * 28 * 28
                }, {
                    'type': 'text',
                    'text': init_prompt
                }]
            }]
        elif 'video' in anno:
            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'video',
                    'video': nncore.join(self.VIDEO_ROOT, anno['video']),
                    'num_threads': self.data_args.num_threads,
                    'min_pixels': 128 * 28 * 28,
                    'max_pixels': 256 * 28 * 28,
                    'max_frames': int(self.data_args.sample_frames.split(',')[-1]),
                    'fps': 2
                }, {
                    'type': 'text',
                    'text': init_prompt
                }]
            }]
        else:
            messages = [{'role': 'user', 'content': init_prompt}]

        for conv in anno['conversations'][1:]:
            assert conv['from'] in ('human', 'gpt')
            role = 'user' if conv['from'] == 'human' else 'assistant'
            messages.append({'role': role, 'content': conv['value']})

        meta = dict(messages=messages)
        return meta


@DATASETS.register(name='llava_instruct_665k_videogpt_plus_576k')
class LlavaVideoGPTPlusDataset(MultimodalDataset):

    ANNO_PATH = 'data/general/llava_v1_5_mix665k_with_videogpt_plus_576k.json'

    IMAGE_ROOT = 'data/llava_instruct'
    VIDEO_ROOT = 'data/videogpt_plus'

    SOURCE = 'llava_instruct_665k_videogpt_plus_576k'

    @classmethod
    def load_annos(self):
        annos = nncore.load(self.ANNO_PATH)

        for anno in annos:
            assert not ('image' in anno and 'video' in anno), anno
            if 'image' in anno:
                anno['source'] = 'llava_instruct_665k'
                anno['data_type'] = 'multimodal'
            elif 'video' in anno:
                anno['source'] = 'videogpt_plus_576k'
                anno['data_type'] = 'multimodal'
            else:
                anno['source'] = 'llava_instruct_665k'
                anno['data_type'] = 'text'

        return annos


class GroundedQADataset(MultimodalDataset):
    """
    Base class for Grounded Video QA datasets with temporal evidence.
    
    Child classes should define:
    - ANNO_PATH or get_anno_path(split): Path to annotation CSV file
    - VIDEO_ROOT: Root directory for videos
    - SOURCE: Dataset source name
    - supports_splits (optional): Whether the dataset has train/val splits
    """
    
    # These should be overridden by child classes
    ANNO_PATH = None
    VIDEO_ROOT = None
    SOURCE = None
    supports_splits = False
    
    def __init__(self, processor, model_args, data_args, training_args, repeat=1, split='train'):
        self.split = split if self.supports_splits else 'train'
        super().__init__(processor, model_args, data_args, training_args, repeat)
    
    @classmethod
    def get_anno_path(cls, split='train'):
        """
        Get annotation path for the given split.
        Override this method if the dataset has multiple splits.
        """
        return cls.ANNO_PATH
    
    @classmethod
    def load_annos(cls, split='train'):
        """Load annotations from CSV file."""
        import pandas as pd
        import ast
        
        anno_path = cls.get_anno_path(split)
        if anno_path is None:
            raise ValueError(f"{cls.__name__} must define ANNO_PATH or override get_anno_path()")
        
        df = pd.read_csv(anno_path)
        
        annos = []
        for _, row in df.iterrows():
            entry_id = row['entry_id']
            video_path = row['video_path']
            question = row['question']
            answer = row['answer']
            
            # Parse candidates (multiple choice options)
            candidates = ast.literal_eval(row['candidates'])      
                  
            # Parse temporal evidence
            try:
                temp_evidence = ast.literal_eval(row['temp_evidence'])
            except:
                temp_evidence = []

            # Get answer letter (A, B, C, D)
            ans_idx = candidates.index(answer)
            ans_letter = chr(ord('A') + ans_idx)
            
            # Format question with options - ensure all options end with "." to avoid information leakage
            normalized_options = []
            for opt in candidates:
                opt = opt.strip()
                if not opt.endswith('.'):
                    opt = opt + '.'
                normalized_options.append(opt)
            
            options_text = '\n'.join([f'{chr(ord("A") + i)}. {opt}' for i, opt in enumerate(normalized_options)])
            question_with_options = f"{question}\n{options_text}"
            
            # Randomly select a prompt template for diversity
            prompt_template = random.choice(GROUNDED_QA_PROMPTS)
            question_with_prompt = prompt_template.format(question_with_options)
            
            # Format answer with temporal evidence
            if temp_evidence:
                # Convert temporal segments to list format [[start, end], ...]
                answer_with_evidence = f"Answer: {ans_letter}. Temporal evidence: {temp_evidence}"
            else:
                answer_with_evidence = f"Answer: {ans_letter}."
            
            anno = {
                'video': video_path,
                'conversations': [
                    {
                        'from': 'human',
                        'value': f'<video>\n{question_with_prompt}'
                    },
                    {
                        'from': 'gpt',
                        'value': answer_with_evidence
                    }
                ],
                'source': cls.SOURCE,
                'data_type': 'multimodal',
                'entry_id': entry_id,
            }
            
            annos.append(anno)
        
        split_info = f" ({split} split)" if cls.supports_splits else ""
        print(f"Loaded {len(annos)} samples from {cls.SOURCE}{split_info}")
        return annos


@DATASETS.register(name='et_instruct')
class ETInstructDataset(GroundedQADataset):
    """ET-Instruct: Grounded Video QA with temporal evidence from ego-centric videos."""

    ANNO_PATH = 'data/et_instruct/et_instruct_gvq_qa_with_options.csv'
    VIDEO_ROOT = 'data/et_instruct'
    SOURCE = 'et_instruct'
    supports_splits = False


@DATASETS.register(name='rextime')
class RexTimeDataset(GroundedQADataset):
    """RexTime: Grounded Video QA with temporal reasoning and evidence."""

    VIDEO_ROOT = 'data/rextime'
    SOURCE = 'rextime'
    supports_splits = True
    
    @classmethod
    def get_anno_path(cls, split='train'):
        """Get annotation path for the given split."""
        if split == 'train':
            return 'data/rextime/rextime_train.csv'
        elif split in ['val', 'valid']:
            return 'data/rextime/rextime_val.csv'
        else:
            raise ValueError(f"Unknown split: {split}. Must be 'train' or 'val'.")
