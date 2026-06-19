# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import warnings

import torch
from torch.nn.utils.rnn import pad_sequence

from unipixel.constants import IGNORE_INDEX


class HybridDataCollator(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [d['input_ids'] for d in batch]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = [d['labels'] for d in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        assert input_ids.size() == labels.size()

        seq_len, max_len = input_ids.size(1), self.tokenizer.model_max_length
        if seq_len > max_len:
            warnings.warn(f'The length of input sequence exceeds model max length: {seq_len} > {max_len}')
            input_ids, labels = input_ids[:, :max_len], labels[:, :max_len]

        data = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids != self.tokenizer.pad_token_id)

        # With refined data_types, all samples in a batch should have the same keys
        # Check ANY sample in batch for these keys (not just batch[0])
        
        # Concatenate vision features if present in all samples
        for key in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'second_per_grid_ts'):
            # Check if key exists in ANY sample
            key_count = sum(1 for d in batch if key in d)
            if key_count > 0:
                if key_count != len(batch):
                    raise ValueError(
                        f'Heterogeneous batch detected: key "{key}" is present in {key_count}/{len(batch)} samples. '
                        f'All samples in a batch must have identical keys. '
                        f'Sample keys: {[list(d.keys()) for d in batch[:3]]}'
                    )
                data[key] = torch.cat([d[key] for d in batch])

        # Collect task-specific keys
        for key in ('frames', 'frame_size', 'point_coords', 'point_labels', 'point_frames', 'refer_mask',
                    'label_obj_to_frame_idx', 'label_mask', 'meta'):
            # Check if key exists in ANY sample
            key_count = sum(1 for d in batch if key in d)
            if key_count > 0:
                if key_count != len(batch):
                    raise ValueError(
                        f'Heterogeneous batch detected: key "{key}" is present in {key_count}/{len(batch)} samples. '
                        f'All samples in a batch must have identical keys. '
                        f'Sample keys: {[list(d.keys()) for d in batch[:3]]}'
                    )
                data[key] = [d[key] for d in batch]

        return data
