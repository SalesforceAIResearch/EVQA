# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy
import random
import re
from collections import OrderedDict

import nncore
import numpy as np
import termplotlib as tpl
import torch
import torch.distributed as dist
import torchvision.transforms.functional as T
from torch.utils.data import Dataset

from sam2.modeling.sam2_utils import get_next_point, sample_box_points
from unipixel.constants import REF_TOKEN, SEG_TOKEN
from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import build_obj_to_frame_idx, process_masks
from unipixel.utils.env import get_auto_device
from unipixel.utils.io import load_frames, load_frames_with_inds

SEG_MEMORY_PROMPTS = [
    '{} Please localize the relevant objects with IDs.',
    '{} Segment all the relevant target(s) with ID(s) in the video.',
    '{} Can you segment the target object(s) with ID(s) in the video?',
    'Please segment the mentioned objects with IDs in the question: {}',
    'Find and track the relevant objects with IDs in the query: {}',
    "Given the query '{}', please provide segmentation masks for the relevant objects with IDs.",
]


class MemoryDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args, repeat=1):
        super().__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = len(anno.get('question', '').split(' ')) + len(anno.get('response', '').split(' '))
            min_frames, max_frames = data_args.min_video_frames, data_args.max_video_frames
            if data_args.min_num_words >= 0 and num_words < data_args.min_num_words:
                continue
            if data_args.max_num_words >= 0 and num_words > data_args.max_num_words:
                continue
            if min_frames >= 0 and len(anno['frames' if 'frames' in anno else 'all_frame_inds']) < min_frames:
                continue
            if max_frames >= 0 and len(anno['frames' if 'frames' in anno else 'all_frame_inds']) > max_frames:
                continue
            annos.append(anno)

        if training_args.local_rank in (0, -1):
            print(f'[{self.SOURCE}]')

            d = [len(a['frames' if 'frames' in anno else 'all_frame_inds']) for a in annos]
            d, _ = torch.Tensor(d).sort()
            n, r = min(d.size(0), 10), d.flip(0)
            print(f'Top-{n} max number of frames: {[round(r[i].item(), 1) for i in range(n)]}')
            print(f'Top-{n} min number of frames: {[round(d[i].item(), 1) for i in range(n)]}')
            print(f'Average number of frames ({d.size(0)} samples): {round(d.mean().item(), 1)}')

            print('Number of frames histogram:')
            counts, edges = np.histogram(d)
            labels = [f'{edges[i]:.2f} - {edges[i + 1]:.2f}' for i in range(len(edges) - 1)]
            fig = tpl.figure()
            fig.barh(counts, labels)
            fig.show()

        # sample point_inds and sync across processes
        device = get_auto_device()
        refer_inds = [i for i, a in enumerate(annos) if '{}' in a['data_type']]
        point_inds = torch.LongTensor(random.sample(refer_inds, len(refer_inds) // 2)).to(device)

        dist.broadcast(point_inds, 0)
        point_inds = point_inds.tolist()
        box_inds = list(set(refer_inds) - set(point_inds))
        assert len(point_inds) + len(box_inds) == len(refer_inds)

        for idx in point_inds:
            annos[idx]['data_type'] = annos[idx]['data_type'].format('point')

        for idx in box_inds:
            annos[idx]['data_type'] = annos[idx]['data_type'].format('box')

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

        if 'video_path' in anno:
            frames, paths, inds = load_frames_with_inds(
                anno['video_path'],
                anno['all_frame_inds'],
                single_frame_mode=False,
                sample_frames=self.data_args.sample_frames,
                sample_type=self.data_args.sample_type,
                sample_for_llm_only=self.data_args.sample_for_llm_only,
                num_threads=self.data_args.num_threads)
        elif 'inst_it' in anno['source']:
            # no sampling for inst_it datasets
            frames, paths, inds = load_frames(anno['frames'], sample_frames=-1)
        else:
            frames, paths, inds = load_frames(
                anno['frames'],
                sample_frames=self.data_args.sample_frames,
                sample_type=self.data_args.sample_type,
                sample_for_llm_only=self.data_args.sample_for_llm_only)

        frame_size = frames.shape[1:3]

        sample_frames = int(self.data_args.sample_frames.split(',')[-1])
        pixels_factor = max(1, min(int(sample_frames / len(paths)), sample_frames))

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': paths,
                'num_threads': self.data_args.num_threads,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * pixels_factor
            }]
        }]

        if len(anno['samples']) > 1:
            samples = random.sample(anno['samples'], min(self.data_args.max_conv_turns, len(anno['samples'])))
        else:
            samples = anno['samples']

        label_mask = []

        for sample in samples:
            # wrap the question with memory templates
            question = random.choice(SEG_MEMORY_PROMPTS).format(sample['question'])
            response = sample['response']

            if len(messages) == 1:
                messages[0]['content'].append({'type': 'text', 'text': question})
            else:
                messages.append({'role': 'user', 'content': question})

            messages.append({'role': 'assistant', 'content': response})

            masks = process_masks(sample, frame_size, inds)
            label_mask.append(masks)

        # label_mask: num_turns * num_turn_obj * num_sampled_frames * height * width
        label_mask = torch.stack([torch.stack(o) for t in label_mask for o in t]).transpose(0, 1)
        assert label_mask.shape[2:] == frame_size
        label_mask = T.resize(label_mask, (self.model_args.sam2_image_size, self.model_args.sam2_image_size))
        label_mask = label_mask > 0

        # label_mask: num_sampled_frames * num_objs * sam2_image_size * sam2_image_size
        if (label_mask == 0).all(dim=(0, 2, 3)).any():
            raise ValueError(f'empty mask: {label_mask.size()}')

        if self.data_args.max_num_objects > 0 and label_mask.size(1) > self.data_args.max_num_objects:
            raise ValueError(f'number of masks exceeds limit: {label_mask.size(1)} > {self.data_args.max_num_objects}')

        point_coords, point_labels, point_frames = [], [], []
        for obj_idx in range(label_mask.size(1)):
            sample_pool = label_mask.any(dim=(-1, -2))[:, obj_idx].nonzero()[:, 0].tolist()
            prompt_frame_idx = random.choice(sample_pool)
            area = label_mask[prompt_frame_idx, None, obj_idx, None]
            point_frames.append(prompt_frame_idx)

            if 'point' in anno['data_type']:
                obj_point_coords, obj_point_labels = get_next_point(area, None, 'uniform', positive_only=True)
                assert (obj_point_labels == 1).all(), obj_point_labels
            elif 'box' in anno['data_type']:
                obj_point_coords, obj_point_labels = sample_box_points(area)
                assert ((obj_point_labels == 2) + (obj_point_labels == 3)).all(), obj_point_labels
            else:
                raise ValueError(f"unknown data type: {anno['data_type']}")

            point_coords.append(obj_point_coords)
            point_labels.append(obj_point_labels)

        point_frames = [torch.LongTensor(point_frames)]

        # assuming one object has only one point or box
        # point_coords: num_objs * num_points * 2
        # point_labels: num_objs * num_points (0: pos 1: neg 2: top left 3: bottom right)
        # point_frames: num_video (1) * num_obj_per_video

        label_obj_to_frame_idx = build_obj_to_frame_idx(label_mask, self.model_args.sam2_batch_mode)

        meta = dict(
            messages=messages,
            frames=frames,
            frame_size=frame_size,
            point_coords=point_coords,
            point_labels=point_labels,
            point_frames=point_frames,
            label_obj_to_frame_idx=label_obj_to_frame_idx,
            label_mask=label_mask)

        return meta


@DATASETS.register(name='videorefer_qa_mem')
class VideoReferQAMemDataset(MemoryDataset):

    ANNO_PATH = 'data/videorefer/VideoRefer-700K/videorefer-qa-75k-wo-masks.json'

    DATA_ROOT = 'data/videorefer/VideoRefer-700K/videos'
    MASK_ROOT = 'data/videorefer/VideoRefer-700K/masks_qa'

    SOURCE = 'videorefer_qa_mem'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)

        annos = []
        for idx, raw_anno in enumerate(raw_annos):
            # skip samples with no object or more than 5 objects
            if len(raw_anno['annotation']) == 0 or len(raw_anno['annotation']) > 5:
                continue

            # skip samples with too few annotations
            if raw_anno['ratio'] < 0.3:
                continue

            # special case in videorefer_qa: raw_anno['annotation'] is a list (no need if using wo-masks)
            # if raw_anno['id'].startswith('ytb_vos'):
            #     assert len(raw_anno['annotation']) == 1 and len(raw_anno['annotation'][0]) == 1
            #     assert str(raw_anno['frame_idx']) in raw_anno['annotation'][0]
            #     raw_anno['annotation'] = [raw_anno['annotation'][0][str(raw_anno['frame_idx'])]]

            # single round conversation
            assert len(raw_anno['conversations']) == 2

            question = raw_anno['conversations'][0]['value'].strip()
            response = raw_anno['conversations'][1]['value'].strip()

            assert question.startswith('<video>\n') or question.endswith('\n<video>')

            if question.startswith('<video>\n'):
                question = question[len('<video>\n'):].strip()
            if question.endswith('\n<video>'):
                question = question[:-len('\n<video>')].strip()

            assert '<video>' not in question

            # check whether the numbers of <region> and objects are aligned
            assert question.count('<region>') == len(raw_anno['annotation'])

            # obj_frame_inds: num_objs * num_frames_per_obj
            # all_frame_inds: num_frames
            obj_frame_inds = [sorted([int(k) for k in o.keys()]) for o in raw_anno['annotation']]
            all_frame_inds = sorted(list(set(nncore.flatten(obj_frame_inds))))

            # sanity check
            assert question.count('\n') in (0, 1)
            if '\n' in question:
                assert question.startswith('There')
                assert '<object' in question
            else:
                assert question.count('<region>') == 1
                assert '<object' not in question

            # videorefer_short_caption: single_frame_mode for all samples
            # videorefer_detailed_caption and videorefer_qa: single_frame_mode for around half of the samples
            if len(all_frame_inds) == 1:
                continue

            # response may also contain objects
            question = question + f"\nOptions:\n({random.choice(['A', 'B', 'C', 'D'])}) {response}"

            if '<object' in question:
                matches = re.findall(r'<object\d+><region>', question)
                response = ''
                for match in matches:
                    # '<object0><region>' -> '0' '<object0>'
                    oid, obj = match[7:-9], match[:-8]
                    assert oid == str(int(oid)), match
                    question = question.replace(match, f'[{oid}] {REF_TOKEN}').replace(obj, f'[{oid}]')
                    response += f' [{oid}] {SEG_TOKEN}'
                response = response.strip()
                assert '<region>' not in question, question
            else:
                oid = random.randint(0, 15)
                question = question.replace('<region>', f'[{oid}] {REF_TOKEN}')
                response = f'[{oid}] {SEG_TOKEN}'

            assert question.count(REF_TOKEN) == response.count(SEG_TOKEN) > 0

            anno = dict(
                source=self.SOURCE,
                data_type='mem_{}_video',
                video_path=nncore.join(self.DATA_ROOT, raw_anno['video']),
                all_frame_inds=all_frame_inds,
                vid=nncore.pure_name(raw_anno['video']),
                samples=[
                    dict(
                        qid=raw_anno['id'],
                        question=question,
                        response=response,
                        mask_type='rle',
                        masks=nncore.join(self.MASK_ROOT, f'{idx}.json'))
                ])

            annos.append(anno)

        return annos


@DATASETS.register(name='inst_it_video_qa_raw_mem')
class InstITVideoQARawMemDataset(MemoryDataset):

    ANNO_PATH = 'data/inst_it/Inst-IT-Dataset/inst_it_dataset_video_21k_w_mask.json'

    DATA_ROOT = 'data/inst_it/Inst-IT-Dataset/videos_raw'

    SOURCE = 'inst_it_video_qa_raw_mem'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_id']

            paths = nncore.ls(
                nncore.join(self.DATA_ROOT, raw_anno['video_path'].split('/', 1)[1]),
                ext=('png', 'jpg'),
                join_path=True)
            paths.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))

            # skip samples with more than 8 frames
            if len(paths) > 8:
                continue

            anno = dict(
                source=self.SOURCE,
                data_type='mem_{}_video' if len(paths) > 1 else 'mem_{}_image',
                frames=paths,
                vid=vid,
                samples=[])

            for i, meta in enumerate(raw_anno['question_answer_pairs']):
                # both oids and tids should start from 1
                assert '[0]' not in meta['question'] and '[0]' not in meta['answer']

                # response may also contain objects
                question = meta['question'] + f"\nOptions:\n({random.choice(['A', 'B', 'C', 'D'])}) {meta['answer']}"

                # potential bugs in annotations
                question = question.replace('<0>', '<1>')
                for j in range(1, 10):
                    question = question.replace(f'[0{j}]', f'[{j}]')

                matches = list(set(re.findall(r'\[\d+\]', question)))

                # skip samples with no object or more than 5 objects
                if len(matches) == 0 or len(matches) > 5:
                    continue

                oids, masks, response = [], [], []
                for match in matches:
                    oid = match[1:-1]
                    assert oid == str(int(oid)), question
                    if oid in oids:
                        continue
                    oids.append(oid)

                    obj_masks = []
                    for path in paths:
                        frame = nncore.base_name(path)
                        if frame in raw_anno['segmentations'] and oid in raw_anno['segmentations'][frame]:
                            obj_masks.append([raw_anno['segmentations'][frame][oid]['mask']])
                        else:
                            obj_masks.append([None])
                    masks.append(obj_masks)

                    question = question.replace(match, f'{match} {REF_TOKEN}', 1)
                    response.append(f'{match} {SEG_TOKEN}')

                response = ' '.join(response)

                assert question.count(REF_TOKEN) == response.count(SEG_TOKEN) == len(masks) > 0

                sample = dict(qid=f'{vid}_{i}', question=question, response=response, mask_type='rle', masks=masks)
                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos
