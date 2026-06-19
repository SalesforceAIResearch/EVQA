# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random
from collections import defaultdict
from itertools import accumulate

import nncore
import torch.nn.functional as F
import torchvision.transforms.functional as T
from tabulate import tabulate
from torch.utils.data import Dataset

from unipixel.dataset.utils import preprocess, process_vision_info
from unipixel.utils.transforms import get_sam2_transform

DATASETS = nncore.Registry('datasets')


class HybridDataset(Dataset):

    @staticmethod
    def _get_vision_modality(anno):
        """Infer vision modality (image/video/none) from annotation structure."""
        # Check for explicit image/video keys (multimodal dataset)
        if 'image' in anno:
            return 'image'
        elif 'video' in anno:
            return 'video'
        
        # Check for video_path (indicates video)
        if 'video_path' in anno:
            return 'video'
        
        # Check frames field
        if 'frames' in anno:
            frames = anno['frames']
            if isinstance(frames, list):
                if len(frames) > 1:
                    return 'video'
                elif len(frames) == 1:
                    return 'image'
        
        # Default: no vision modality (text-only)
        return None

    def __init__(self, processor, model_args, data_args, training_args):
        super().__init__()

        self.debug_print_count = 0  # Counter for debug printing
        self.debug_print_limit = getattr(data_args, 'debug_print_samples', 0)

        datasets = []
        for key in data_args.datasets.split(','):
            key, repeat = key.split(':') if ':' in key else (key, 1)
            datasets.append(DATASETS.get(key)(processor, model_args, data_args, training_args, int(repeat)))

        # Refine data_types to include vision modality for homogeneous batching
        data_types = []
        for d in datasets:
            for i in range(len(d.annos) * d.repeat):
                anno = d.annos[i % len(d.annos)]
                base_type = anno['data_type']
                
                # Determine vision modality from annotation structure
                vision_modality = self._get_vision_modality(anno)
                
                # Append modality suffix to data_type for grouping
                # Only refine if base_type doesn't already specify image/video
                if vision_modality and not ('image' in base_type or 'video' in base_type):
                    refined_type = f"{base_type}@{vision_modality}"
                else:
                    refined_type = base_type
                
                data_types.append(refined_type)

        cum_length = [0] + list(accumulate([len(d) for d in datasets]))
        idx_ranges = [[cum_length[i], cum_length[i + 1]] for i in range(len(cum_length) - 1)]

        if training_args.local_rank in (0, -1):
            raw_length = sum(d.raw_length * d.repeat for d in datasets)
            cur_length = idx_ranges[-1][-1]

            ratio = round(cur_length / raw_length * 100, 2)
            print(f'Number of samples: {raw_length} (original) -> {cur_length} (filtered) {ratio}%')

            data_type_cnt = ' '.join([f'{data_types.count(t)} ({t})' for t in list(set(data_types))])
            print(f'Data types: {data_type_cnt}')

            cnts, reps = defaultdict(int), dict()
            for dataset in datasets:
                for anno in dataset.annos:
                    source = anno.get('source', 'unknown')
                    cnts[source] += dataset.repeat
                    if source not in reps:
                        reps[source] = dataset.repeat

            tab = [[k, reps[k], v, round(v / cur_length, 5)] for k, v in cnts.items()]
            print(tabulate(tab, headers=['Source', 'Repeat', '#Samples', 'Ratio'], tablefmt='pretty', stralign='left'))

        self.sam2_transform = get_sam2_transform(model_args.sam2_image_size)

        self.datasets = datasets
        self.data_types = data_types
        self.idx_ranges = idx_ranges
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def _print_debug_sample(self, idx, meta, formatted_text, input_ids, labels, source):
        """Print debug information for a training sample."""
        print(f"\n{'='*100}")
        print(f"DEBUG SAMPLE #{self.debug_print_count} (Global idx: {idx}, Dataset: {source})")
        print(f"{'='*100}")
        
        # Print formatted prompt (what goes into the model)
        print(f"\n📥 FORMATTED INPUT TEXT (after chat template):")
        print(f"{'─'*100}")
        print(formatted_text[:1000])  # Truncate if too long
        if len(formatted_text) > 1000:
            print(f"... (truncated, total length: {len(formatted_text)} chars)")
        
        # Print input IDs info
        print(f"\n🔢 INPUT IDS:")
        print(f"{'─'*100}")
        print(f"  Shape: {input_ids.shape}")
        print(f"  First 50 tokens: {input_ids[:50].tolist()}")
        
        # Decode input for readability (skip special tokens)
        try:
            decoded_input_clean = self.processor.tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"\n📄 DECODED INPUT (text only, first 500 chars):")
            print(f"{'─'*100}")
            print(decoded_input_clean[:500])
            if len(decoded_input_clean) > 500:
                print(f"... (total {len(decoded_input_clean)} chars)")
        except Exception as e:
            print(f"  ⚠️ Could not decode input: {e}")
        
        # Print labels info
        print(f"\n🎯 LABELS:")
        print(f"{'─'*100}")
        print(f"  Shape: {labels.shape}")
        
        # Find valid labels (not -100)
        valid_mask = labels != -100
        valid_labels = labels[valid_mask]
        ignore_count = (labels == -100).sum().item()
        
        print(f"  Ignore tokens (-100): {ignore_count}/{len(labels)} ({100*ignore_count/len(labels):.1f}%)")
        print(f"  Valid tokens: {len(valid_labels)}/{len(labels)} ({100*len(valid_labels)/len(labels):.1f}%)")
        
        if len(valid_labels) > 0:
            print(f"  Valid label tokens: {valid_labels.tolist()}")
            
            # Decode valid labels (skip special tokens)
            try:
                decoded_labels_clean = self.processor.tokenizer.decode(valid_labels, skip_special_tokens=True)
                print(f"\n📄 DECODED LABELS (text only - what model should generate):")
                print(f"{'─'*100}")
                print(decoded_labels_clean)
            except Exception as e:
                print(f"  ⚠️ Could not decode labels: {e}")
        
        # Print original messages structure
        if 'messages' in meta:
            print(f"\n📨 ORIGINAL MESSAGES STRUCTURE:")
            print(f"{'─'*100}")
            for i, msg in enumerate(meta['messages']):
                role = msg.get('role', msg.get('from', 'unknown'))
                content = msg.get('content', msg.get('value', ''))
                
                # Handle both list and string content
                if isinstance(content, list):
                    print(f"  Message {i} ({role}):")
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_content = item.get('text', '')
                                print(f"    [text]: {text_content[:200]}")
                                if len(text_content) > 200:
                                    print(f"    ... (truncated)")
                            elif item.get('type') in ('video', 'image'):
                                print(f"    [{item['type']}]: {item.get(item['type'], 'N/A')}")
                else:
                    print(f"  Message {i} ({role}): {str(content)[:200]}")
                    if len(str(content)) > 200:
                        print(f"  ... (truncated)")
        
        print(f"\n{'='*100}\n")

    def __len__(self):
        return self.idx_ranges[-1][-1]

    def __getitem__(self, idx):
        for retry in range(self.data_args.max_retries + 1):
            try:
                return self.fetch_data(idx)
            except Exception as err:
                for (s, e), dataset in zip(self.idx_ranges, self.datasets):
                    if s <= idx < e:
                        break
                print(f'Error in loading {idx} ({idx - s} of {dataset.SOURCE}): {type(err).__name__}({err})')
                idx = random.choice([i for i, t in enumerate(self.data_types[s:e]) if t == self.data_types[idx]]) + s

        raise RuntimeError(f'Data loading failed after {retry} retries')

    def fetch_data(self, idx):
        for (s, e), dataset in zip(self.idx_ranges, self.datasets):
            if s <= idx < e:
                meta = dataset[idx - s]
                break

        text = self.processor.apply_chat_template(meta['messages'])
        text = [text.strip()]

        images, videos, kwargs = process_vision_info(meta['messages'], return_video_kwargs=True, sanity_check=True)

        data = self.processor(text=text, images=images, videos=videos, return_tensors='pt', **kwargs)
        assert data['input_ids'].size(0) == 1

        num_tokens, max_num_tokens = data['input_ids'].size(1), self.data_args.max_num_tokens
        if max_num_tokens > 0 and num_tokens > max_num_tokens:
            raise ValueError(f'number of tokens exceeds limit: {num_tokens} > {max_num_tokens}')

        data['input_ids'] = data['input_ids'][0]
        data['labels'] = preprocess(data['input_ids'], self.processor.cache_text[0], self.processor.tokenizer,
                                    self.model_args.conv_type)

        if 'frames' in meta:
            data['frames'] = self.sam2_transform(meta['frames'])

        for key in ('frame_size', 'point_coords', 'point_labels', 'point_frames', 'label_obj_to_frame_idx',
                    'label_mask'):
            if key in meta:
                data[key] = meta[key]

        if 'refer_mask' in meta:
            # ensure refer mask has the correct size
            refer_mask = meta['refer_mask']

            # spatial patch size set to 14, spatial merge size set to 2
            refer_mask = T.resize(refer_mask, (data['video_grid_thw'][0][1] * 14, data['video_grid_thw'][0][2] * 14))
            refer_mask = F.max_pool2d(refer_mask, kernel_size=28, stride=28)
            refer_mask = refer_mask > 0

            # double check the mask after resizing
            if (refer_mask == 0).all(dim=(0, 2, 3)).any():
                raise ValueError(f'empty refer mask after resizing: {refer_mask.size()}')

            assert refer_mask.size(0) == data['video_grid_thw'][0][0]
            assert refer_mask.size(2) == data['video_grid_thw'][0][1] // 2
            assert refer_mask.size(3) == data['video_grid_thw'][0][2] // 2

            data['refer_mask'] = refer_mask

        # incorporate meta data for debugging
        data['meta'] = dict(idx=idx, dataset=dataset.SOURCE, dataset_idx=idx - s, num_tokens=num_tokens, 
                           data_type=self.data_types[idx])

        # Debug printing of prompts and labels
        if self.debug_print_limit > 0 and self.debug_print_count < self.debug_print_limit:
            self.debug_print_count += 1
            self._print_debug_sample(idx, meta, text[0], data['input_ids'], data['labels'], dataset.SOURCE)

        return data
