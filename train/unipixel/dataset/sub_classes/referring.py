# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy
import random
from collections import OrderedDict

import nncore
import numpy as np
import termplotlib as tpl
import torch
import torch.distributed as dist
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from sam2.modeling.sam2_utils import get_next_point, sample_box_points
from unipixel.constants import REF_TOKEN
from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import process_masks
from unipixel.utils.env import get_auto_device
from unipixel.utils.io import load_frames, load_frames_with_inds

REF_IMAGE_SHORT_PROMPTS = [
    f'Give me a short description of {REF_TOKEN}.',
    f'Can you give me a short description of {REF_TOKEN}?',
    f'Can you provide me with a short description of the region in the picture marked by {REF_TOKEN}?',
    f"I'm curious about the region represented by {REF_TOKEN} in the picture. Could you describe it in a few words?",
    f'What can you tell me about the region indicated by {REF_TOKEN} in the image in a few words?',
    f"I'd like to know more about the area in the photo labeled {REF_TOKEN}. Can you give me a concise description?",
    f'Could you describe the region shown as {REF_TOKEN} in the picture concisely?',
    f'What can you give me about the region outlined by {REF_TOKEN} in the photo?',
    f'Please provide me with a brief description of the region marked with {REF_TOKEN} in the image.',
    f'Can you give me a brief introduction of the region labeled as {REF_TOKEN} in the picture?',
    f"I'm interested in knowing the region represented by {REF_TOKEN} in the photo. Can you describe it in several words?",
    f'What is the region outlined by {REF_TOKEN} in the picture like? Could you give me a streamlined description?',
    f'Can you provide me with a brief description of the region in the picture marked by {REF_TOKEN}, please?',
    f"I'm curious about the region represented by {REF_TOKEN} in the picture. Could you describe it in a few words, please?",
    f'What can you tell me about the region indicated by {REF_TOKEN} in the image?',
    f"I'd like to know more about the area in the photo labeled {REF_TOKEN}, please. Can you give me a simple description?",
    f'Could you describe the region shown as {REF_TOKEN} in the picture in several words?',
    f'Please provide me with a simple description of the region marked with {REF_TOKEN} in the image, please.',
    f"I'm interested in learning more about the region represented by {REF_TOKEN} in the photo. Can you describe it in a few words, please?",
    f'What is the region outlined by {REF_TOKEN} in the picture like, please? Could you give me a simple and clear description?',
    f'Please describe the region {REF_TOKEN} in the image concisely.',
    f'Can you offer a simple analysis of the region {REF_TOKEN} in the image?',
    f'Could tell me something about the region highlighted by {REF_TOKEN} in the picture briefly?',
    f'Can you share a simple rundown of the region denoted by {REF_TOKEN} in the presented image?',
]


class ReferringDataset(Dataset):

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

        if single_frame_mode := 'video_path' in anno:
            frames, paths, inds = load_frames_with_inds(
                anno['video_path'],
                anno['all_frame_inds'],
                single_frame_mode=True,
                sample_frames=self.data_args.sample_frames,
                sample_type=self.data_args.sample_type,
                sample_for_llm_only=self.data_args.sample_for_llm_only,
                num_threads=self.data_args.num_threads)
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
            if sample['type'] == 'qa':
                question = sample['question']
                response = sample['response']
            elif sample['type'] == 'image_short_caption':
                question = random.choice(REF_IMAGE_SHORT_PROMPTS)
                response = sample['response']
            else:
                raise KeyError(f'unknown sample type: {sample}')

            if len(messages) == 1:
                messages[0]['content'].append({'type': 'text', 'text': question})
            else:
                messages.append({'role': 'user', 'content': question})

            messages.append({'role': 'assistant', 'content': response})

            mask_type, masks = sample['mask_type'], sample['masks']

            if single_frame_mode:
                # map the only mask back to the original video
                if isinstance(masks, str):
                    masks = nncore.load(masks)
                for i in range(len(masks)):
                    assert len(masks[i]) == len(anno['all_frame_inds']) == 1
                    masks[i] = [masks[i][0] if j == anno['all_frame_inds'][0] else [None] for j in range(max(inds) + 1)]

            masks = process_masks(dict(mask_type=mask_type, masks=masks), frame_size, inds)
            label_mask.append(masks)

        # label_mask: num_turns * num_turn_obj * num_sampled_frames * height * width
        label_mask = torch.stack([torch.stack(o) for t in label_mask for o in t]).transpose(0, 1)
        assert label_mask.shape[2:] == frame_size
        label_mask = F.resize(label_mask, (self.model_args.sam2_image_size, self.model_args.sam2_image_size))
        label_mask = label_mask > 0

        # label_mask: num_sampled_frames * num_objs * sam2_image_size * sam2_image_size
        if (label_mask == 0).all(dim=(0, 2, 3)).any():
            raise ValueError(f'empty mask: {label_mask.size()}')

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

        meta = dict(messages=messages, point_coords=point_coords, point_labels=point_labels, point_frames=point_frames)
        return meta


@DATASETS.register(name='videorefer_short_caption_ref')
class VideoReferShortCaptionRefDataset(ReferringDataset):

    ANNO_PATH = 'data/videorefer/VideoRefer-700K/videorefer-short-caption-500k-wo-masks.json'

    DATA_ROOT = 'data/videorefer/VideoRefer-700K/videos'
    MASK_ROOT = 'data/videorefer/VideoRefer-700K/masks_short_caption'

    SOURCE = 'videorefer_short_caption_ref'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)

        annos = []
        for idx, raw_anno in enumerate(raw_annos):
            # single round conversation
            assert len(raw_anno['conversations']) == 2

            question = raw_anno['conversations'][0]['value'].strip()
            response = raw_anno['conversations'][1]['value'].strip()

            assert question.startswith('<video>\n') or question.endswith('\n<video>')

            if question.startswith('<video>\n'):
                question = question[len('<video>\n'):].strip()
            if question.endswith('\n<video>'):
                question = question[:-len('\n<video>')].strip()

            assert '<video>' not in question and '<object' not in question
            assert '<object' not in response

            # check whether the numbers of <region> and objects are aligned
            assert question.count('<region>') == len(raw_anno['annotation']) == 1

            # obj_frame_inds: num_objs * num_frames_per_obj
            # all_frame_inds: num_frames
            obj_frame_inds = [sorted([int(k) for k in o.keys()]) for o in raw_anno['annotation']]
            all_frame_inds = sorted(list(set(nncore.flatten(obj_frame_inds))))

            # videorefer_short_caption: single_frame_mode for all samples
            # videorefer_detailed_caption and videorefer_qa: single_frame_mode for around half of the samples

            question = question.replace('<region>', REF_TOKEN)

            anno = dict(
                source=self.SOURCE,
                data_type='ref_{}',
                video_path=nncore.join(self.DATA_ROOT, raw_anno['video']),
                all_frame_inds=all_frame_inds,
                vid=nncore.pure_name(raw_anno['video']),
                samples=[
                    dict(
                        qid=raw_anno['id'],
                        type='qa',
                        question=question,
                        response=response,
                        mask_type='rle',
                        masks=nncore.join(self.MASK_ROOT, f'{idx}.json'))
                ])

            annos.append(anno)

        return annos


@DATASETS.register(name='inst_it_image_short_caption_raw_ref')
class InstITImageShortCaptionRawRefDataset(ReferringDataset):

    ANNO_PATH = 'data/inst_it/Inst-IT-Dataset/inst_it_dataset_image_51k_w_mask.json'

    DATA_ROOT = 'data/inst_it/Inst-IT-Dataset/images_raw'

    SOURCE = 'inst_it_image_short_caption_raw_ref'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)

        key = 'short' if 'short_caption' in self.SOURCE else 'long'

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['image_id']

            anno = dict(
                source=self.SOURCE,
                data_type='ref_{}',
                frames=[nncore.join(self.DATA_ROOT, raw_anno['image_path'].split('/', 1)[1])],
                vid=vid,
                samples=[])

            for oid, caption in raw_anno['instance_level_caption'].items():
                sample = dict(
                    qid=f'{vid}_{oid}',
                    type=f'image_{key}_caption',
                    response=caption[key],
                    mask_type='rle',
                    masks=[[[raw_anno['segmentations'][oid]['mask']]]])

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos
