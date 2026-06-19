# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy
import random
import re
import warnings
from collections import OrderedDict, defaultdict

import nncore
import numpy as np
import termplotlib as tpl
import torch
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from unipixel.constants import SEG_TOKEN
from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import build_obj_to_frame_idx, process_masks
from unipixel.utils.io import load_frames, load_frames_with_stride
from .utils import REFER

SEG_ST_EVIDENCE_PROMPTS = [
    '{} \nAnswer the question and provide evidence in the form of temporal ([[start1, end1], [start2, end2], ...]) and spatial evidence (masks).',
    '{} \nAnswer the question and provide both temporal and spatial evidence. The temporal evidence should be in the form of [[start1, end1], [start2, end2], ...] and the spatial evidence should be in the form of masks.',
    '{} \nProvide your answer along with supporting evidence: temporal segments in the format [[start1, end1], [start2, end2], ...] and spatial masks.',
    '{} \nAnswer with evidence. Include time segments as [[start_sec, end_sec], ...] and highlight evidence objects with masks.',
    '{} \nPlease answer and identify when (temporal evidence: [[start, end], ...]) and where (spatial evidence: masks) the relevant information appears.',
]

SEG_EXPLANATORY_PROMPTS = [
    '{} Please provide the segmentation mask and explain why.',
    '{} Respond with the segmentation mask and explain the reason.',
    '{} Output the segmentation mask and give some explanations.',
    '{} Provide the segmentation mask while answering the question.',
    '{} Segment the relevant target and answer the question.',
]

SEG_SENTENCE_PROMPTS = [
    '{} Please provide the segmentation mask.',
    '{} Please respond with the segmentation mask.',
    '{} Output the segmentation mask.',
    '{} Give me the segmentation results directly.',
    '{} Segment the target directly.',
]

SEG_QUESTION_PROMPTS = [
    '{} Please answer the question and provide the segmentation mask.',
    '{} Please associate your answer with the segmentation mask.',
    '{} Answer the question and provide the corresponding segmentation mask.',
    '{} Give me the answer and segment the target.',
    '{} Provide a short answer while segmenting the relevant object.',
]

SEG_DESCRIPTION_PROMPTS = [
    'Please segment the object according to the description: {}',
    'Segment the target according to the description: {}',
    'Analyze the following sentences and provide the corresponding segmentation mask: {}',
    'Given the description: {} Where is the described object in this video?',
    'Find the object according to the description: {}',
]

SEG_IMAGE_QUERY_PROMPTS = [
    'Can you segment the {} in this image?',
    'Please segment the {} in this image.',
    'Can you find the {} in the image?',
    'Segment the {} in the image.',
    'Track the {} in this image.',
    'Where is the {} in this image? Please respond with the segmentation mask.',
    'What is the {}? Please output the segmentation mask.',
]

SEG_VIDEO_QUERY_PROMPTS = [
    'Can you segment the {} in this video?',
    'Please segment the {} in this video.',
    'Can you find the {} in the video?',
    'Segment the {} in the video.',
    'Track the {} in this video.',
    'Where is the {} in this video? Please respond with the segmentation mask.',
    'What is the {}? Please output the segmentation mask.',
]

SEG_RESPONSES = [
    f'The target is {SEG_TOKEN}.',
    f'The segmentation result is {SEG_TOKEN}.',
    f'Sure, the segmentation mask is {SEG_TOKEN}.',
    f'Sure, it is {SEG_TOKEN}.',
    f'{SEG_TOKEN}.',
]


class SegmentationDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args, repeat=1):
        super().__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            min_frames, max_frames = data_args.min_video_frames, data_args.max_video_frames
            if min_frames >= 0 and 'frames' in anno and len(anno['frames']) < min_frames:
                print(f"Skipping {anno['vid']} because it has less than {min_frames} frames")
                continue
            if max_frames >= 0 and 'frames' in anno and len(anno['frames']) > max_frames:
                print(f"Skipping {anno['vid']} because it has more than {max_frames} frames")
                continue
            annos.append(anno)
        print(f"after frame filter, {len(annos)} / {len(raw_annos)} are left")
        if training_args.local_rank in (0, -1):
            print(f'[{self.SOURCE}]')

            d = [len(a['frames']) for a in annos]
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

        if anno['frames'][0] is None:
            # case 1: ref_sav dataset
            frames, paths, inds = load_frames_with_stride(
                anno['video_path'],
                every_n_frames=4,
                sample_frames=self.data_args.sample_frames,
                sample_type=self.data_args.sample_type,
                sample_for_llm_only=self.data_args.sample_for_llm_only,
                num_threads=self.data_args.num_threads)
        else:
            # case 2: other datasets
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

        if isinstance(anno['samples'], dict):
            # special case for instance/semantic segmentation datasets
            masks = [np.array(Image.open(path)) for path in anno['samples']['masks']]
            oids = [oid for oid in np.unique(np.stack(masks)).tolist() if oid not in self.IGNORE_OIDS]
            assert len(oids) > 0, 'empty oids'
            oids = random.sample(oids, min(self.data_args.max_conv_turns, len(oids)))
            samples = []
            for oid in oids:
                sample = dict(type='query', query=self.CLASSES[oid], mask_type='image', masks=masks, oids=[[oid]])
                samples.append(sample)
        elif len(anno['samples']) > 1:
            samples = random.sample(anno['samples'], min(self.data_args.max_conv_turns, len(anno['samples'])))
        else:
            samples = anno['samples']

        label_mask = []

        for sample in samples:
            if sample['type'] == 'query':
                if isinstance(sample['query'], tuple):
                    obj, par = sample['query']
                    query = f'{obj} {par}' if random.random() < 0.5 else f'{par} of the {obj}'
                else:
                    query = sample['query'].strip('.')
                query = query[0].lower() + query[1:]
                for key in ('the ', 'an ', 'a '):
                    if query.startswith(key):
                        query = query[len(key):]
                template = SEG_IMAGE_QUERY_PROMPTS if 'image' in anno['data_type'] else SEG_VIDEO_QUERY_PROMPTS
                question = random.choice(template).format(query)
                response = random.choice(SEG_RESPONSES)
            elif sample['type'] == 'sentence':
                question = random.choice(SEG_SENTENCE_PROMPTS).format(sample['query'])
                response = random.choice(SEG_RESPONSES)
            elif sample['type'] == 'explanatory':
                question = random.choice(SEG_EXPLANATORY_PROMPTS).format(sample['query'])
                response = random.choice(SEG_RESPONSES) + ' ' + sample['answer']
            elif sample['type'] == 'question':
                question = random.choice(SEG_QUESTION_PROMPTS).format(sample['query'])
                response = f"{sample['answer']} {SEG_TOKEN}."
            elif sample['type'] == 'st_evidence':
                question = random.choice(SEG_ST_EVIDENCE_PROMPTS).format(sample['query'])
                response = f"{sample['answer']} {SEG_TOKEN}."
            elif sample['type'] == 'description':
                question = random.choice(SEG_DESCRIPTION_PROMPTS).format(sample['query'])
                response = random.choice(SEG_RESPONSES)
            else:
                raise KeyError(f'unknown sample type: {sample}')

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
        label_mask = F.resize(label_mask, (self.model_args.sam2_image_size, self.model_args.sam2_image_size))
        label_mask = label_mask > 0
        # print("label_mask:", label_mask.shape)

        # label_mask: num_sampled_frames * num_objs * sam2_image_size * sam2_image_size
        if (label_mask == 0).all():
            warnings.warn(f"{anno['source']} {idx} has empty mask")

        if self.data_args.max_num_objects > 0 and label_mask.size(1) > self.data_args.max_num_objects:
            raise ValueError(f'number of masks exceeds limit: {label_mask.size(1)} > {self.data_args.max_num_objects}')

        label_obj_to_frame_idx = build_obj_to_frame_idx(label_mask, self.model_args.sam2_batch_mode)

        meta = dict(
            messages=messages,
            frames=frames,
            frame_size=frame_size,
            label_obj_to_frame_idx=label_obj_to_frame_idx,
            label_mask=label_mask)

        return meta


@DATASETS.register(name='revos')
class ReVOSDataset(SegmentationDataset):

    META_DICT_TRAIN = 'data/revos/meta_expressions_train_.json'
    META_DICT_VALID = 'data/revos/meta_expressions_valid_.json'

    MASK_DICT = 'data/revos/mask_dict.json'

    DATA_ROOT = 'data/revos'

    SOURCE = 'revos'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            meta_dict = nncore.load(self.META_DICT_TRAIN, object_pairs_hook=OrderedDict)['videos']
        else:
            meta_dict = nncore.load(self.META_DICT_VALID, object_pairs_hook=OrderedDict)['videos']

        mask_dict = nncore.load(self.MASK_DICT)

        annos = []
        for vid, meta in meta_dict.items():
            cache = defaultdict(list)
            for qid, obj in meta['expressions'].items():
                assert len(obj['obj_id']) == len(obj['anno_id'])

                # skip no-object samples during training
                if split == 'train' and len(obj['obj_id']) == 0:
                    continue

                query = obj['exp'].strip()

                sample = dict(
                    qid=qid,
                    type='sentence' if query.endswith('?') else 'query',
                    query=query,
                    mask_type='rle',
                    masks=[[m for m in zip(*[mask_dict[str(a)] for a in obj['anno_id']])]])

                if split == 'train':
                    assert len(sample['masks'][0]) == len(meta['frames'])
                    assert any(any(o is not None for o in m) for m in sample['masks'][0])

                if split == 'train' and vid.split('/')[0] == 'TAO':
                    oid = '_'.join([str(o) for o in sorted(obj['obj_id'])])
                else:
                    oid = 'none'

                cache[oid].append(sample)

            for samples in cache.values():
                # TAO videos are too long, group and crop them during training
                if split == 'train' and vid.split('/')[0] == 'TAO':
                    s, e = None, None
                    for i, m in enumerate(samples[0]['masks'][0]):
                        if any(o is not None for o in m):
                            s = i
                            break
                    for i, m in enumerate(reversed(samples[0]['masks'][0])):
                        if any(o is not None for o in m):
                            e = len(samples[0]['masks'][0]) - i
                            break
                    if e - s < 2:
                        continue
                    frames = meta['frames'][s:e]
                    for i in range(len(samples)):
                        samples[i]['masks'] = [samples[i]['masks'][0][s:e]]
                else:
                    frames = meta['frames']

                anno = dict(
                    source=self.SOURCE,
                    data_type='seg_video' if len(frames) > 1 else 'seg_image',
                    frames=[nncore.join(self.DATA_ROOT, vid, f'{n}.jpg') for n in frames],
                    vid=vid,
                    samples=samples)

                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='mevis')
class MeViSDataset(SegmentationDataset):

    META_DICT_TRAIN = 'data/mevis/train/meta_expressions.json'
    META_DICT_VALID = 'data/mevis/valid/meta_expressions.json'
    META_DICT_VALID_U = 'data/mevis/valid_u/meta_expressions.json'

    MASK_DICT_TRAIN = 'data/mevis/train/mask_dict.json'
    MASK_DICT_VALID_U = 'data/mevis/valid_u/mask_dict.json'

    DATA_ROOT_TRAIN = 'data/mevis/train/JPEGImages'
    DATA_ROOT_VALID = 'data/mevis/valid/JPEGImages'
    DATA_ROOT_VALID_U = 'data/mevis/valid_u/JPEGImages'

    SOURCE = 'mevis'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            meta_dict = nncore.load(self.META_DICT_TRAIN, object_pairs_hook=OrderedDict)['videos']
            mask_dict = nncore.load(self.MASK_DICT_TRAIN)
            data_root = self.DATA_ROOT_TRAIN
        elif split == 'valid_u':
            meta_dict = nncore.load(self.META_DICT_VALID_U, object_pairs_hook=OrderedDict)['videos']
            mask_dict = nncore.load(self.MASK_DICT_VALID_U)
            data_root = self.DATA_ROOT_VALID_U
        else:
            meta_dict = nncore.load(self.META_DICT_VALID, object_pairs_hook=OrderedDict)['videos']
            data_root = self.DATA_ROOT_VALID

        annos = []
        for path, meta in meta_dict.items():
            frames = meta['frames']

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video' if len(frames) > 1 else 'seg_image',
                frames=[nncore.join(data_root, path, f'{n}.jpg') for n in frames],
                vid=path,
                samples=[])

            for qid, obj in meta['expressions'].items():
                # skip no-object samples during training
                if split == 'train' and len(obj['obj_id']) == 0:
                    continue

                sample = dict(qid=qid, type='query', query=obj['exp'].strip())

                if split in ('train', 'valid_u'):
                    sample['mask_type'] = 'rle'
                    sample['masks'] = [[m for m in zip(*[mask_dict[str(a)] for a in obj['anno_id']])]]
                    assert len(sample['masks'][0]) == len(frames)
                    assert any(any(o is not None for o in m) for m in sample['masks'][0])

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='lvvis')
class LVVISDataset(SegmentationDataset):

    META_DICT = 'data/lvvis/meta_expressions.json'

    MASK_DICT = 'data/lvvis/mask_dict.json'

    DATA_ROOT_TRAIN = 'data/lvvis/train/JPEGImages'
    DATA_ROOT_VALID = 'data/lvvis/val/JPEGImages'

    SOURCE = 'lvvis'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            data_root = self.DATA_ROOT_TRAIN
        else:
            raise KeyError('no masks for valid split')
            data_root = self.DATA_ROOT_VALID

        meta_dict = nncore.load(self.META_DICT, object_pairs_hook=OrderedDict)['videos']
        mask_dict = nncore.load(self.MASK_DICT)

        annos = []
        for path, meta in meta_dict.items():
            frames = meta['frames']

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video' if len(frames) > 1 else 'seg_image',
                frames=[nncore.join(data_root, path, f'{n}.jpg') for n in frames],
                vid=path,
                samples=[])

            for qid, obj in meta['expressions'].items():
                assert len(obj['obj_id']) == len(obj['anno_id'])

                # skip no-object samples during training
                if split == 'train' and len(obj['obj_id']) == 0:
                    continue

                sample = dict(qid=qid, type='query', query=obj['exp'].replace('_', ' ').strip())

                if split == 'train':
                    sample['mask_type'] = 'rle'
                    sample['masks'] = [[m for m in zip(*[mask_dict[str(a)] for a in obj['anno_id']])]]
                    assert len(sample['masks'][0]) == len(frames)
                    assert any(any(o is not None for o in m) for m in sample['masks'][0])

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='ref_youtube_vos')
class RefYouTubeVOSDataset(SegmentationDataset):

    META_DICT_TRAIN = 'data/ref_youtube_vos/meta_expressions/train/meta_expressions.json'
    META_DICT_VALID = 'data/ref_youtube_vos/meta_expressions/valid/meta_expressions.json'

    MASK_DICT = 'data/ref_youtube_vos/mask_dict.pkl'

    DATA_ROOT_TRAIN = 'data/ref_youtube_vos/train/JPEGImages'
    DATA_ROOT_VALID = 'data/ref_youtube_vos/valid/JPEGImages'

    SOURCE = 'ref_youtube_vos'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            meta_dict = nncore.load(self.META_DICT_TRAIN, object_pairs_hook=OrderedDict)['videos']
            mask_dict = nncore.load(self.MASK_DICT)
            assert len(mask_dict) == sum([len(v['expressions']) for v in meta_dict.values()])
            data_root = self.DATA_ROOT_TRAIN
        else:
            meta_dict = nncore.load(self.META_DICT_VALID, object_pairs_hook=OrderedDict)['videos']
            data_root = self.DATA_ROOT_VALID

        annos, idx = [], 0
        for vid, meta in meta_dict.items():
            frames = meta['frames']

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video' if len(frames) > 1 else 'seg_image',
                frames=[nncore.join(data_root, vid, f'{n}.jpg') for n in frames],
                vid=vid,
                samples=[])

            for qid, obj in meta['expressions'].items():
                sample = dict(qid=qid, type='query', query=obj['exp'].strip())

                if split == 'train':
                    sample['mask_type'] = 'rle'
                    sample['masks'] = [[[m] for m in mask_dict[str(idx)]]]
                    idx += 1
                    assert len(sample['masks'][0]) == len(frames)
                    assert any(any(o is not None for o in m) for m in sample['masks'][0])

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='ref_davis17')
class RefDAVIS17Dataset(SegmentationDataset):

    META_DICT_TRAIN = 'data/ref_davis17/meta_expressions/train/meta_expressions.json'
    META_DICT_VALID = 'data/ref_davis17/meta_expressions/valid/meta_expressions.json'

    MASK_DICT_TRAIN = 'data/ref_davis17/train/mask_dict.pkl'
    MASK_DICT_VALID = 'data/ref_davis17/valid/mask_dict.pkl'

    DATA_ROOT_TRAIN = 'data/ref_davis17/train/JPEGImages'
    DATA_ROOT_VALID = 'data/ref_davis17/valid/JPEGImages'

    SOURCE = 'ref_davis17'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            meta_dict = nncore.load(self.META_DICT_TRAIN, object_pairs_hook=OrderedDict)['videos']
            mask_dict = nncore.load(self.MASK_DICT_TRAIN)
            data_root = self.DATA_ROOT_TRAIN
        else:
            meta_dict = nncore.load(self.META_DICT_VALID, object_pairs_hook=OrderedDict)['videos']
            mask_dict = nncore.load(self.MASK_DICT_VALID)
            data_root = self.DATA_ROOT_VALID

        assert len(mask_dict) == sum([len(v['expressions']) for v in meta_dict.values()])

        annos, idx = [], 0
        for vid, meta in meta_dict.items():
            frames = meta['frames']

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video' if len(frames) > 1 else 'seg_image',
                frames=[nncore.join(data_root, vid, f'{n}.jpg') for n in frames],
                vid=vid,
                samples=[])

            for qid, obj in meta['expressions'].items():
                sample = dict(
                    qid=qid,
                    type='query',
                    query=obj['exp'].strip(),
                    mask_type='rle',
                    masks=[[[m] for m in mask_dict[str(idx)]]])

                assert len(sample['masks'][0]) == len(frames)
                # five samples do not have mask
                # assert any(any(o is not None for o in m) for m in sample['masks'][0])
                idx += 1

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='ref_sav')
class RefSAVDataset(SegmentationDataset):

    META_DICT = 'data/ref_sav/Ref-SAV-Min-Ratio-0.3.json'

    DATA_ROOT = 'data/sav'

    SOURCE = 'ref_sav'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        meta_dict = nncore.load(self.META_DICT)

        annos = []
        for vid, meta in meta_dict.items():
            video_path = meta['video_path'].split('/', 1)[1]
            assert video_path.startswith('sav_train')

            anno_path = meta['anno_path'].split('/', 1)[1]
            assert anno_path.startswith('sav_train') and 'manual' in anno_path

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video',
                video_path=nncore.join(self.DATA_ROOT, video_path),
                frames=[None],  # dummy frames
                vid=vid,
                samples=[])

            for qid, obj in meta['objects'].items():
                assert obj['video_id'] == vid and str(obj['obj_id']) == qid

                sample = dict(qid=qid, type='description', query=obj['formated'].strip())

                if split == 'train':
                    sample['mask_type'] = 'sav'
                    sample['masks'] = nncore.join(self.DATA_ROOT, anno_path)

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='groundmore')
class GroundMoReDataset(SegmentationDataset):

    META_DICT_TRAIN = 'data/groundmore/trainval_v2.json'
    META_DICT_VALID = 'data/groundmore/test_v2.json'

    ANNO_ROOT = 'data/groundmore/annotations'

    SOURCE = 'groundmore'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            meta_dict = nncore.load(self.META_DICT_TRAIN, object_pairs_hook=OrderedDict)['videos']
        else:
            meta_dict = nncore.load(self.META_DICT_VALID, object_pairs_hook=OrderedDict)['videos']

        annos = []
        for vid, meta in meta_dict.items():
            anno_dir = nncore.join(self.ANNO_ROOT, vid)
            if not nncore.is_dir(anno_dir):
                continue

            if split == 'train':
                # only use frames with masks during training
                masks = nncore.ls(nncore.join(anno_dir, 'masks'), ext='png', join_path=True)
                masks.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))

                if len(masks) == 0:
                    continue

                frames = [nncore.join(anno_dir, 'images', f'{nncore.pure_name(n)}.jpg') for n in masks]
            else:
                # uniformly sample 20 frames during evaluation
                frames = nncore.ls(nncore.join(anno_dir, 'images'), ext='jpg', join_path=True)
                frames.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))

                keep = np.linspace(0, len(frames) - 1, num=20, dtype=int)
                frames = [frames[i] for i in keep]

                masks = nncore.ls(nncore.join(anno_dir, 'masks'), ext='png')
                masks = set([nncore.pure_name(m) for m in masks])
                masks = [
                    nncore.join(anno_dir, 'masks', f'{nncore.pure_name(n)}.png')
                    if nncore.pure_name(n) in masks else None for n in frames
                ]

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video' if len(masks) > 1 else 'seg_image',
                frames=frames,
                vid=vid,
                samples=[])

            for qid, obj in meta['questions'].items():
                query = obj['question'].strip()
                if not query.endswith('?'):
                    query = query + '?'

                sample = dict(
                    qid=qid,
                    type='question',
                    query=query,
                    answer=obj['answer'],
                    task=obj['q_type'],
                    mask_type='image',
                    masks=masks,
                    oids=[[int(o) for o in obj['obj_id'].split(', ')]])

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='vicas')
class ViCaSDataset(SegmentationDataset):

    ANNO_ROOT = 'data/vicas/annotations/annotations_wo_masks_v1.0.json'

    SPLIT_TRAIN = 'data/vicas/splits/v1.0/train.json'
    SPLIT_VALID = 'data/vicas/splits/v1.0/val.json'
    SPLIT_TEST = 'data/vicas/splits/v1.0/test.json'

    DATA_ROOT = 'data/vicas/video_frames'
    MASK_ROOT = 'data/vicas/masks'

    SOURCE = 'vicas'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            anno_ids = nncore.load(self.SPLIT_TRAIN)
        elif split in ('val', 'valid'):
            anno_ids = nncore.load(self.SPLIT_VALID)
        else:
            anno_ids = nncore.load(self.SPLIT_TEST)

        preload_annos = nncore.is_file(self.ANNO_ROOT)

        if preload_annos:
            raw_annos = nncore.load(self.ANNO_ROOT)

        annos = []
        for anno_id in anno_ids:
            if preload_annos:
                raw_anno = raw_annos[str(anno_id).zfill(6)]
            else:
                raw_anno = nncore.load(nncore.join(self.ANNO_ROOT, str(anno_id).zfill(6) + '.json'))

            vid = raw_anno['filename'][:6]

            frames = [nncore.join(self.DATA_ROOT, vid, s['filename']) for s in raw_anno['segmentations']]

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video' if len(frames) > 1 else 'seg_image',
                frames=frames,
                vid=vid,
                samples=[])

            for i, meta in enumerate(raw_anno['object_referrals']):
                # skip samples with multiple instances
                # if len(meta['track_ids']) > 1:
                #     continue

                sample = dict(qid=f'{vid}_{i}', type='sentence', query=meta['prompt'].strip())

                if split == 'train':
                    if preload_annos:
                        sample['mask_type'] = 'vicas'
                        sample['masks'] = [nncore.join(self.MASK_ROOT, vid, f'{i}.json')]
                    else:
                        sample['mask_type'] = 'rle'
                        sample['masks'] = [[[s['mask_rles'][s['track_ids'].index(t)] for t in meta['track_ids']]
                                            for s in raw_anno['segmentations']]]
                        assert len(sample['masks'][0]) == len(frames)
                        assert any(any(o is not None for o in m) for m in sample['masks'][0])

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='st_evidence_gen_qa')
class STEvidenceGenQADataset(SegmentationDataset):
    """Spatial-Temporal Evidence Dataset with generated QA pairs from ViCaS videos."""

    QA_CSV_PATH = 'data/st-evidence-instruct/gen_qa/vicas/st_evidence_vicas.csv'

    DATA_ROOT = 'data/st-evidence-instruct/gen_qa/vicas/video_frames'
    ANNO_ROOT = 'data/st-evidence-instruct/gen_qa/vicas/annotations/v1.0'

    SOURCE = 'st_evidence_gen_qa'

    @classmethod
    def load_annos(self, split='train'):
        import pandas as pd
        import ast

        # Load complete CSV with temporal evidence included
        df = pd.read_csv(self.QA_CSV_PATH)

        if split == 'train':
            df = df[df['split'] == 'train']
        elif split == 'val' or split == 'valid':
            df = df[df['split'] == 'val']
        else:
            df = df[df['split'] == 'test']

        annos = []
        grouped = df.groupby('video_id')

        for video_id, group in grouped:
            vid = str(video_id).zfill(6)

            # Load video annotation to get frames and segmentations
            anno_path = nncore.join(self.ANNO_ROOT, f'{vid}.json')
            if not nncore.is_file(anno_path):
                continue

            raw_anno = nncore.load(anno_path)
            frames = [nncore.join(self.DATA_ROOT, vid, s['filename']) for s in raw_anno['segmentations']]

            if len(frames) == 0:
                continue

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video_st_evidence',
                frames=frames,
                vid=vid,
                samples=[])

            for _, row in group.iterrows():
                entry_id = row['entry_id']
                question = row['question']
                answer = row['answer']

                # Parse candidates
                try:
                    candidates = ast.literal_eval(row['candidates'])
                except:
                    continue

                # Parse spatial evidence (mask indices, 1-indexed)
                try:
                    spatial_evidence = ast.literal_eval(row['mask_evidence'])
                    if not isinstance(spatial_evidence, list):
                        spatial_evidence = []
                except:
                    spatial_evidence = []

                # Skip if no spatial evidence
                if len(spatial_evidence) == 0:
                    print(f"No spatial evidence for {entry_id}")
                    continue

                # Get temporal evidence from CSV column
                temporal_segments = row.get('temporal_evidence', '[]')
                try:
                    temporal_segments = ast.literal_eval(temporal_segments) if isinstance(temporal_segments, str) else temporal_segments
                    if not isinstance(temporal_segments, list):
                        temporal_segments = []
                except:
                    temporal_segments = []

                # Get answer letter (A, B, C, D, ...)
                ans_idx = candidates.index(answer) if answer in candidates else -1
                if ans_idx == -1:
                    continue
                ans_letter = chr(ord('A') + ans_idx)

                # Build RLE masks from segmentations for spatial evidence
                # spatial_evidence contains mask IDs (1-indexed), which correspond to track_ids
                # e.g., mask_1 -> track_id 1, mask_2 -> track_id 2
                track_ids = spatial_evidence  # mask indices are the same as track_ids

                # Extract RLE masks in ViCaS format: [[frame1_masks], [frame2_masks], ...]
                masks = [[[s['mask_rles'][s['track_ids'].index(t)] if t in s['track_ids'] else None
                          for t in track_ids]
                         for s in raw_anno['segmentations']]]

                # Verify we have valid masks
                assert len(masks[0]) == len(frames), f"Mask frames {len(masks[0])} != video frames {len(frames)}"
                assert any(any(o is not None for o in m) for m in masks[0]), "No valid masks found"

                # Format question with options
                question_with_options = question + '\nOptions:\n'
                for i, option in enumerate(candidates):
                    question_with_options += f"({chr(ord('A') + i)}) {option}\n"
                question_with_options = question_with_options.strip()

                # Format answer with only letter, temporal and spatial evidence
                answer_parts = [f"Answer: {ans_letter}."]

                if len(temporal_segments) > 0:
                    # Format as nested list: [[start1, end1], [start2, end2], ...]
                    segments_list = [f"[{s[0]:.1f}, {s[1]:.1f}]" for s in temporal_segments]
                    temporal_str = '[' + ', '.join(segments_list) + ']'
                    answer_parts.append(f"Temporal Evidence: {temporal_str}.")

                answer_parts.append("Spatial Evidence:")
                answer_with_evidence = ' '.join(answer_parts)

                sample = dict(
                    qid=entry_id,
                    type='st_evidence',
                    query=question_with_options,
                    answer=answer_with_evidence,
                    task='st_evidence',
                    mask_type='rle',
                    masks=masks)

                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        # Flatten samples following GroundMoRe pattern
        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='reason_seg')
class ReasonSegDataset(SegmentationDataset):

    DATA_ROOT_TRAIN = 'data/lisa/reason_seg/ReasonSeg/train'
    DATA_ROOT_VALID = 'data/lisa/reason_seg/ReasonSeg/val'
    DATA_ROOT_TEST = 'data/lisa/reason_seg/ReasonSeg/test'

    EXPLANATORY = 'data/lisa/reason_seg/ReasonSeg/explanatory/train.json'

    SOURCE = 'reason_seg'

    @classmethod
    def load_annos(self, split='train', add_explanatory=True):
        if split == 'train':
            images = nncore.ls(self.DATA_ROOT_TRAIN, ext='jpg', join_path=True, sort=True)
            if add_explanatory:
                exp_meta = nncore.load(self.EXPLANATORY)
                exp_meta = {nncore.pure_name(m['image']): m for m in exp_meta}
        elif split in ('val', 'valid'):
            images = nncore.ls(self.DATA_ROOT_VALID, ext='jpg', join_path=True, sort=True)
        else:
            images = nncore.ls(self.DATA_ROOT_TEST, ext='jpg', join_path=True, sort=True)

        meta_paths = [f'{p[:-4]}.json' for p in images]

        annos = []
        for image, meta_path in zip(images, meta_paths):
            meta = nncore.load(meta_path)
            vid = nncore.pure_name(image)

            shapes = []
            for i, shape in enumerate(meta['shapes']):
                assert shape['image_name'] == f'{vid}.jpg'
                assert shape['shape_type'] == 'polygon'
                assert shape['label'] in ('target', 'ignore', 'flag')
                if shape['label'] == 'flag':
                    continue
                shapes.append(shape)

            anno = dict(source=self.SOURCE, data_type='seg_image', frames=[image], vid=vid, samples=[])

            for qid, query in enumerate(meta['text']):
                sample = dict(
                    qid=qid,
                    type='sentence' if meta['is_sentence'] else 'query',
                    query=query,
                    mask_type='polygon',
                    masks=[[shapes]])
                anno['samples'].append(sample)

            if split == 'train' and add_explanatory and vid in exp_meta:
                meta = exp_meta.pop(vid)
                sample = dict(
                    qid='exp',
                    type='explanatory',
                    query=meta['query'],
                    answer=meta['outputs'],
                    mask_type='polygon',
                    masks=[[shapes]])
                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        if split == 'train' and add_explanatory:
            assert len(exp_meta) == 0, (len(exp_meta), exp_meta)

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='ade20k')
class ADE20KDataset(SegmentationDataset):

    CLASSES = [
        'unlabeled', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass',
        'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence',
        'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard',
        'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
        'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm',
        'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light',
        'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane',
        'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster',
        'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool',
        'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture',
        'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
        'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
    ]

    IGNORE_OIDS = [0, 255]

    MASK_ROOT_TRAIN = 'data/lisa/ade20k/annotations/training'
    MASK_ROOT_VALID = 'data/lisa/ade20k/annotations/validation'

    DATA_ROOT_TRAIN = 'data/lisa/ade20k/images/training'
    DATA_ROOT_VALID = 'data/lisa/ade20k/images/validation'

    SOURCE = 'ade20k'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            masks = nncore.ls(self.MASK_ROOT_TRAIN, ext='png', join_path=True, sort=True)
            data_root = self.DATA_ROOT_TRAIN
        else:
            masks = nncore.ls(self.MASK_ROOT_VALID, ext='png', join_path=True, sort=True)
            data_root = self.DATA_ROOT_VALID

        annos = []
        for mask in masks:
            vid = nncore.pure_name(mask)

            anno = dict(
                source=self.SOURCE,
                data_type='seg_image',
                frames=[nncore.join(data_root, f'{vid}.jpg')],
                vid=vid,
                samples=dict(masks=[mask]))

            annos.append(anno)

        return annos


@DATASETS.register(name='cocostuff')
class COCOStuffDataset(ADE20KDataset):

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush',
        'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet',
        'ceiling-other', 'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff',
        'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel', 'ground-other', 'hill', 'house',
        'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
        'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river',
        'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid-other',
        'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood',
        'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'
    ]

    IGNORE_OIDS = [
        0, 96, 102, 103, 110, 112, 114, 115, 116, 117, 118, 121, 123, 126, 133, 142, 160, 164, 167, 171, 172, 173, 174,
        175, 176, 177, 178, 180, 181, 255
    ]

    MASK_ROOT_TRAIN = 'data/lisa/cocostuff/train2017'
    MASK_ROOT_VALID = 'data/lisa/cocostuff/val2017'

    DATA_ROOT_TRAIN = 'data/lisa/coco/train2017'
    DATA_ROOT_VALID = 'data/lisa/coco/val2017'

    SOURCE = 'cocostuff'


@DATASETS.register(name='mapillary')
class MapillaryDataset(ADE20KDataset):

    CLASSES = [
        'bird', 'ground animal', 'ambiguous barrier', 'concrete block', 'curb', 'fence', 'guard rail', 'barrier',
        'road median', 'road side', 'lane separator', 'temporary barrier', 'wall', 'bike lane', 'crosswalk - plain',
        'curb cut', 'driveway', 'parking', 'parking aisle', 'pedestrian area', 'rail track', 'road', 'road shoulder',
        'service lane', 'sidewalk', 'traffic island', 'bridge', 'building', 'garage', 'tunnel', 'person',
        'person group', 'bicyclist', 'motorcyclist', 'other rider', 'lane marking - dashed line',
        'lane marking - straight line', 'lane marking - zigzag line', 'lane marking - ambiguous',
        'lane marking - arrow (left)', 'lane marking - arrow (other)', 'lane marking - arrow (right)',
        'lane marking - arrow (split left or straight)', 'lane marking - arrow (split right or straight)',
        'lane marking - arrow (straight)', 'lane marking - crosswalk', 'lane marking - give way (row)',
        'lane marking - give way (single)', 'lane marking - hatched (chevron)', 'lane marking - hatched (diagonal)',
        'lane marking - other', 'lane marking - stop line', 'lane marking - symbol (bicycle)',
        'lane marking - symbol (other)', 'lane marking - text', 'lane marking (only) - dashed line',
        'lane marking (only) - crosswalk', 'lane marking (only) - other', 'lane marking (only) - test', 'mountain',
        'sand', 'sky', 'snow', 'terrain', 'vegetation', 'water', 'banner', 'bench', 'bike rack', 'catch basin',
        'cctv camera', 'fire hydrant', 'junction box', 'mailbox', 'manhole', 'parking meter', 'phone booth', 'pothole',
        'signage - advertisement', 'signage - ambiguous', 'signage - back', 'signage - information', 'signage - other',
        'signage - store', 'street light', 'pole', 'pole group', 'traffic sign frame', 'utility pole', 'traffic cone',
        'traffic light - general (single)', 'traffic light - pedestrians', 'traffic light - general (upright)',
        'traffic light - general (horizontal)', 'traffic light - cyclists', 'traffic light - other',
        'traffic sign - ambiguous', 'traffic sign (back)', 'traffic sign - direction (back)',
        'traffic sign - direction (front)', 'traffic sign (front)', 'traffic sign - parking',
        'traffic sign - temporary (back)', 'traffic sign - temporary (front)', 'trash can', 'bicycle', 'boat', 'bus',
        'car', 'caravan', 'motorcycle', 'on rails', 'other vehicle', 'trailer', 'truck', 'vehicle group',
        'wheeled slow', 'water valve', 'car mount', 'dynamic', 'ego vehicle', 'ground', 'static'
    ]

    IGNORE_OIDS = [123, 255]

    MASK_ROOT_TRAIN = 'data/lisa/mapillary/training/v2.0/labels'
    MASK_ROOT_VALID = 'data/lisa/mapillary/validation/v2.0/labels'

    DATA_ROOT_TRAIN = 'data/lisa/mapillary/training/images'
    DATA_ROOT_VALID = 'data/lisa/mapillary/validation/images'

    SOURCE = 'mapillary'


@DATASETS.register(name='paco_lvis')
class PACOLVISDataset(SegmentationDataset):

    ANNO_PATH_TRAIN = 'data/lisa/vlpart/paco/annotations/paco_lvis_v1_train.json'
    ANNO_PATH_VALID = 'data/lisa/vlpart/paco/annotations/paco_lvis_v1_val.json'

    DATA_ROOT_TRAIN = 'data/lisa/coco'
    DATA_ROOT_VALID = 'data/lisa/coco'

    SOURCE = 'paco_lvis'

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            api = COCO(self.ANNO_PATH_TRAIN)
            data_root = self.DATA_ROOT_TRAIN
        else:
            api = COCO(self.ANNO_PATH_VALID)
            data_root = self.DATA_ROOT_VALID

        classes = api.loadCats(api.getCatIds())
        class_map = dict()

        for cat in classes:
            name = cat['name'].strip().split(':')
            assert len(name) in (1, 2)

            if len(name) == 1:
                name = name[0].split('_(')[0]
            else:
                name = name[0].split('_(')[0], name[1].split('_(')[0]

            class_map[cat['id']] = name

        img_ids = api.getImgIds()

        annos = []
        for img_id in img_ids:
            img = api.loadImgs([img_id])[0]

            ann_ids = api.getAnnIds(imgIds=img['id'])
            anns = api.loadAnns(ann_ids)

            anno = dict(
                source=self.SOURCE,
                data_type='seg_image',
                frames=[nncore.join(data_root, img['file_name'])],
                vid=nncore.pure_name(img['file_name']),
                samples=[])

            for ann in anns:
                sample = dict(
                    type='query',
                    query=class_map[ann['category_id']],
                    mask_type='rle',
                    masks=[[[ann['segmentation']]]],
                    height=img['height'],
                    width=img['width'])
                anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        return annos


@DATASETS.register(name='pascal_part')
class PASCALPartDataset(PACOLVISDataset):

    ANNO_PATH_TRAIN = 'data/lisa/vlpart/pascal_part/train.json'

    DATA_ROOT_TRAIN = 'data/lisa/vlpart/pascal_part/VOCdevkit/VOC2010/JPEGImages'

    SOURCE = 'pascal_part'


@DATASETS.register(name='refcoco')
class RefCOCODataset(SegmentationDataset):

    DATASET = 'refcoco'
    SPLITBY = 'unc'

    ANNO_ROOT = 'data/lisa/refer_seg'

    DATA_ROOT = 'data/lisa/refer_seg/images/mscoco/images/train2014'

    SOURCE = 'refcoco'

    @classmethod
    def load_annos(self, split='train'):
        api = REFER(self.ANNO_ROOT, dataset=self.DATASET, splitBy=self.SPLITBY)

        ref_ids = api.getRefIds(split=split)
        refs = api.loadRefs(ref_ids=ref_ids)

        img_to_refs = defaultdict(list)
        for ref in refs:
            img_to_refs[ref['image_id']].append(ref)

        img_ids = api.getImgIds(ref_ids=ref_ids)
        assert set(img_ids) == set(img_to_refs.keys()), (len(img_ids), len(img_to_refs))

        annos = []
        for img_id, refs in img_to_refs.items():
            img = api.Imgs[img_id]

            anno = dict(
                source=self.SOURCE,
                data_type='seg_image',
                frames=[nncore.join(self.DATA_ROOT, img['file_name'])],
                vid=nncore.pure_name(img['file_name']),
                samples=[])

            for ref in refs:
                ann = api.refToAnn[ref['ref_id']]
                seg = ann['segmentation']
                box = ann['bbox']

                for sent in ref['sentences']:
                    sample = dict(qid=sent['sent_id'], type='query', query=sent['sent'])
                    sample['mask_type'] = 'rle'
                    sample['masks'] = [[[seg]]] if isinstance(seg[0], list) else [[seg]]
                    sample['boxes'] = [[[box[0], box[1], box[0] + box[2], box[1] + box[3]]]]
                    sample['height'] = img['height']
                    sample['width'] = img['width']
                    assert any(any(o is not None for o in m) for m in sample['masks'][0])
                    anno['samples'].append(sample)

            if len(anno['samples']) > 0:
                annos.append(anno)

        if split != 'train':
            annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()} for s in a['samples']] for a in annos]
            annos = nncore.flatten(annos)

        return annos


@DATASETS.register(name='refcoco+')
class RefCOCOPlusDataset(RefCOCODataset):

    DATASET = 'refcoco+'
    SPLITBY = 'unc'

    SOURCE = 'refcoco+'


@DATASETS.register(name='refcocog')
class RefCOCOGDataset(RefCOCODataset):

    DATASET = 'refcocog'
    SPLITBY = 'umd'

    SOURCE = 'refcocog'


@DATASETS.register(name='refclef')
class RefClefDataset(RefCOCODataset):

    DATASET = 'refclef'
    SPLITBY = 'unc'

    DATA_ROOT = 'data/lisa/refer_seg/images/saiapr_tc-12'

    SOURCE = 'refclef'


@DATASETS.register(name='st_evidence_gen_mask')
class STEvidenceGenMaskDataset(SegmentationDataset):
    """ST Evidence (CLEVRER) with generated masks.

    Loads lightweight metadata pickle (no masks) at init time.
    Masks are loaded lazily per-sample from individual JSON files during __getitem__.

    Run `python scripts/convert_polygon_to_rle.py` to generate the metadata pickle
    and per-entry combined RLE JSON files.
    Fallback: loads raw polygon masks from per-entry JSONs (slow).
    """

    META_PKL_PATH = 'data/st-evidence-instruct/gen_mask/st_evidence_meta.pkl'
    MASK_DIR = 'data/st-evidence-instruct/gen_mask/masks'
    QA_CSV_PATH = 'data/st-evidence-instruct/gen_mask/st_evidence.csv'
    FRAME_ROOT = 'data/st-evidence-instruct/gen_mask/video_frames_6fps'

    SOURCE = 'st_evidence_gen_mask'

    @classmethod
    def load_annos(self, split='train'):
        import os

        if os.path.isfile(self.META_PKL_PATH):
            return self._load_preprocessed(self.META_PKL_PATH, split)
        print(f"[{self.SOURCE}] Metadata pickle not found: {self.META_PKL_PATH}")
        print(f"[{self.SOURCE}] Falling back to raw polygon loading (slow). "
              f"Run: python scripts/convert_polygon_to_rle.py")
        return self._load_raw(split)

    @classmethod
    def _load_preprocessed(self, pkl_path, split='train'):
        """Load lightweight metadata pickle; masks loaded lazily via path."""
        import os
        import pickle

        print(f"[{self.SOURCE}] Loading metadata from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            entries = pickle.load(f)

        annos = []
        skipped = 0
        checked_dirs = {}
        for entry in entries:
            frame_dir = os.path.join(self.FRAME_ROOT, entry['video_name'])
            frames = [os.path.join(frame_dir, f'frame_{i:04d}.jpg')
                      for i in range(entry['total_frames'])]

            if frame_dir not in checked_dirs:
                checked_dirs[frame_dir] = os.path.isdir(frame_dir)
            if not checked_dirs[frame_dir]:
                skipped += 1
                continue

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video_st_evidence',
                frames=frames,
                vid=entry['entry_id'],
                samples=[dict(
                    qid=entry['entry_id'],
                    type='st_evidence',
                    query=entry['question_with_options'],
                    answer=entry['answer_with_evidence'],
                    task='st_evidence',
                    mask_type='rle',
                    masks=entry['mask_rle_path'])])

            annos.append(anno)

        print(f"[{self.SOURCE}] Loaded {len(annos)} entries (lazy RLE), "
              f"skipped {skipped} (no frames)")

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()}
                  for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)
        return annos

    @classmethod
    def _load_raw(self, split='train'):
        """Fallback: load raw polygon masks from individual JSON files."""
        import ast
        import json
        import os

        import pandas as pd

        mask_dir = self.MASK_DIR
        df = pd.read_csv(self.QA_CSV_PATH)

        mask_files = set(
            f.replace('.json', '') for f in os.listdir(mask_dir) if f.endswith('.json'))

        annos = []
        skipped_no_mask = 0
        skipped_no_frames = 0

        for _, row in df.iterrows():
            entry_id = row['entry_id']
            video_path = row['video_path']

            if entry_id not in mask_files:
                skipped_no_mask += 1
                continue

            try:
                candidates = ast.literal_eval(row['candidates'])
            except Exception:
                continue

            ans_idx = candidates.index(row['answer']) if row['answer'] in candidates else -1
            if ans_idx == -1:
                continue
            ans_letter = chr(ord('A') + ans_idx)

            mask_path = os.path.join(mask_dir, f'{entry_id}.json')
            with open(mask_path, 'r') as f:
                mask_data = json.load(f)

            evidence_objects = mask_data.get('evidence_objects', [])
            if not evidence_objects:
                continue

            max_frame = max(
                m['frame'] for obj in evidence_objects for m in obj.get('masks', []))
            total_frames = max_frame + 1

            video_name = os.path.splitext(video_path)[0]
            frame_dir = os.path.join(self.FRAME_ROOT, video_name)
            frames = [os.path.join(frame_dir, f'frame_{i:04d}.jpg')
                      for i in range(total_frames)]

            if not os.path.isfile(frames[0]):
                skipped_no_frames += 1
                continue

            combined_per_frame = [[] for _ in range(total_frames)]
            for obj in evidence_objects:
                for m_entry in obj.get('masks', []):
                    frame_idx = m_entry['frame']
                    mask_poly = m_entry.get('mask', {})
                    if mask_poly and mask_poly.get('exterior'):
                        combined_per_frame[frame_idx].append(
                            {'points': mask_poly['exterior'], 'label': 'target'})
                        for hole in mask_poly.get('holes', []):
                            if hole:
                                combined_per_frame[frame_idx].append(
                                    {'points': hole, 'label': 'ignore'})

            if not any(len(shapes) > 0 for shapes in combined_per_frame):
                continue

            # Get temporal evidence from CSV column
            temporal_segments = row.get('temporal_evidence', '[]')
            if isinstance(temporal_segments, str):
                try:
                    temporal_segments = ast.literal_eval(temporal_segments)
                except Exception:
                    temporal_segments = []
            if not isinstance(temporal_segments, list):
                temporal_segments = []

            question_with_options = row['question'] + '\nOptions:\n'
            for i, option in enumerate(candidates):
                question_with_options += f"({chr(ord('A') + i)}) {option}\n"
            question_with_options = question_with_options.strip()

            answer_parts = [f"Answer: {ans_letter}."]
            if temporal_segments:
                segments_list = [f"[{s[0]:.1f}, {s[1]:.1f}]" for s in temporal_segments]
                temporal_str = '[' + ', '.join(segments_list) + ']'
                answer_parts.append(f"Temporal Evidence: {temporal_str}.")
            answer_parts.append("Spatial Evidence:")
            answer_with_evidence = ' '.join(answer_parts)

            anno = dict(
                source=self.SOURCE,
                data_type='seg_video_st_evidence',
                frames=frames,
                vid=entry_id,
                samples=[dict(
                    qid=entry_id,
                    type='st_evidence',
                    query=question_with_options,
                    answer=answer_with_evidence,
                    task='st_evidence',
                    mask_type='polygon',
                    masks=[combined_per_frame])])

            annos.append(anno)

        print(f"[{self.SOURCE}] Loaded {len(annos)} entries (raw polygon), "
              f"skipped {skipped_no_mask} (no mask), {skipped_no_frames} (no frames)")

        annos = [[{k: [s] if k == 'samples' else v for k, v in a.items()}
                  for s in a['samples']] for a in annos]
        annos = nncore.flatten(annos)
        return annos
