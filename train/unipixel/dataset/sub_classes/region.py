# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import copy
import random
import re
from collections import OrderedDict

import nncore
import numpy as np
import termplotlib as tpl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from unipixel.constants import MEM_TOKEN
from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import process_masks
from unipixel.utils.io import load_frames, load_frames_with_inds

REG_DETAILED_PROMPTS = [
    'Can you provide me with a detailed description of the region in the picture marked by {}?',
    "I'm curious about the region represented by {} in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by {} in the image?',
    "I'd like to know more about the area in the photo labeled {}. Can you give me a detailed description?",
    'Could you describe the region shown as {} in the picture in great detail?',
    'What details can you give me about the region outlined by {} in the photo?',
    'Please provide me with a comprehensive description of the region marked with {} in the image.',
    'Can you give me a detailed account of the region labeled as {} in the picture?',
    "I'm interested in learning more about the region represented by {} in the photo. Can you describe it in detail?",
    'What is the region outlined by {} in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by {}, please?',
    "I'm curious about the region represented by {} in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by {} in the image, exactly?',
    "I'd like to know more about the area in the photo labeled {}, please. Can you give me a detailed description?",
    'Could you describe the region shown as {} in the picture in great detail, please?',
    'What details can you give me about the region outlined by {} in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with {} in the image, please.',
    'Can you give me a detailed account of the region labeled as {} in the picture, please?',
    "I'm interested in learning more about the region represented by {} in the photo. Can you describe it in detail, please?",
    'What is the region outlined by {} in the picture like, please? Could you give me a detailed description?',
    'Please describe the region {} in the image in detail.',
    'Can you offer a thorough analysis of the region {} in the image?',
    'Could you elaborate on the region highlighted by {} in the picture provided?',
    'Please share more information about the zone emphasized with {} in the photo.',
    'What insights can you give about the area denoted by {} in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by {} in the presented image?',
    "I'd like to know more about the region highlighted by {} in the picture provided.",
    'Work through the important details of the area {} in the image.',
    'Illustrate the area represtented by {} through a descriptive explanation.',
    'Examine the region {} closely and share its details.',
]


class RegionDataset(Dataset):

    def __init__(self, processor, model_args, data_args, training_args, repeat=1):
        super().__init__()

        raw_annos = self.load_annos()

        annos = []
        for anno in raw_annos:
            num_words = 0
            for k in ('question', 'response'):
                texts = anno[k] if isinstance(anno[k], list) else [anno[k]]
                num_words += sum([len(t.split(' ')) if isinstance(t, str) else -100 for t in texts])
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

            if isinstance(annos[0]['question'], list):
                d = [len(a['question']) for a in annos]
                d, _ = torch.Tensor(d).sort()
                n, r = min(d.size(0), 10), d.flip(0)
                print(f'Top-{n} max number of rounds: {[round(r[i].item(), 1) for i in range(n)]}')
                print(f'Top-{n} min number of rounds: {[round(d[i].item(), 1) for i in range(n)]}')
                print(f'Average number of rounds ({d.size(0)} samples): {round(d.mean().item(), 1)}')

                print('Number of rounds histogram:')
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

        question, response, oids, masks = anno['question'], anno['response'], anno.get('oids'), anno['masks']

        if 'video_path' in anno:
            frames, paths, inds = load_frames_with_inds(
                anno['video_path'],
                anno['all_frame_inds'],
                single_frame_mode=anno['single_frame_mode'],
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

        if anno.get('single_frame_mode'):
            # map the only mask back to the original video
            if isinstance(masks, str):
                masks = nncore.load(masks)
            for i in range(len(masks)):
                assert len(masks[i]) == len(anno['all_frame_inds']) == 1
                masks[i] = [masks[i][0] if j == anno['all_frame_inds'][0] else [None] for j in range(max(inds) + 1)]

        if any(key in anno['source'] for key in ('inst_it', 'visual_genome')):
            # restrict the maximum number of converstion rounds
            assert isinstance(question, list) and isinstance(response, list)
            assert len(question) == len(response)

            sample_inds = random.sample(list(range(len(question))), min(self.data_args.max_conv_turns, len(question)))

            question = [question[i] for i in sample_inds]
            response = [response[i] for i in sample_inds]
            obj_inds = list(set(nncore.flatten([anno['obj_inds'][i] for i in sample_inds])))
            oids, masks = [oids[i] for i in obj_inds], [masks[i] for i in obj_inds]

        refer_mask = dict(
            mask_type=anno['mask_type'], masks=masks, oids=oids, height=anno.get('height'), width=anno.get('width'))
        refer_mask = process_masks(refer_mask, frame_size, inds)

        refer_mask = torch.stack([torch.stack(o) for o in refer_mask]).transpose(0, 1)
        assert refer_mask.shape[2:] == frame_size

        if 'inst_it' in anno['source'] and random.random() < 0.5:
            # randomly set half of the inst_it samples to single_frame_mode
            all_frame_inds = (refer_mask != 0).any(dim=(2, 3)).all(dim=1).nonzero()[:, 0].tolist()
            if len(all_frame_inds) > 0:
                prompt_frame_idx = random.choice(all_frame_inds)
                _refer_mask = torch.zeros_like(refer_mask)
                _refer_mask[prompt_frame_idx] = refer_mask[prompt_frame_idx]
                refer_mask = _refer_mask

        # ensure refer mask has the correct number of frames
        num_objs = refer_mask.size(1)
        if refer_mask.size(0) % 2 != 0:
            refer_mask = torch.cat((refer_mask, refer_mask[-1, None]))
        refer_mask = refer_mask.flatten(1)
        refer_mask = F.max_pool1d(refer_mask.transpose(-1, -2), kernel_size=2, stride=2).transpose(-1, -2)
        refer_mask = refer_mask.view(-1, num_objs, *frame_size)

        # refer_mask: num_sampled_frames * num_objs * sam2_image_size * sam2_image_size
        if (refer_mask == 0).all(dim=(0, 2, 3)).any():
            raise ValueError(f'empty refer mask: {refer_mask.size()}')

        regions = []
        for obj_idx in range(num_objs):
            tids = (refer_mask[:, obj_idx].any(dim=(-1, -2)).nonzero()[:, 0] * 2 + 1).tolist()
            regions.append(tids)

        if is_video := len(paths) > 1:
            prefix = f'Here is a video with {len(paths)} frames denoted as <1> to <{len(paths)}>. The highlighted regions are as follows:\n'
        else:
            assert all(tids == [1] for tids in regions), regions
            prefix = 'Here is an image with the following highlighted regions:\n'

        if anno['source'] == 'videorefer_qa' and '\n' in question:
            intro, question = question.split('\n')
            matches = re.findall(r'<object\d+><region>', intro)
            assert len(matches) == len(regions) > 0, question
            for match, tids in zip(matches, regions):
                # '<object0><region>' -> '0' '<object0>'
                oid, obj = match[7:-9], match[:-8]
                assert oid == str(int(oid)), match
                prefix += f'[{oid}]: ' + ' '.join([(f'<{tid}>-<{tid + 1}> ' if is_video else '') + MEM_TOKEN
                                                   for tid in tids]) + '\n'
                question = question.replace(obj, f'[{oid}]')
                response = response.replace(obj, f'[{oid}]')
            assert '<region>' not in question, question
            assert '<object' not in question and '<object' not in response
        elif 'videorefer' in anno['source']:
            assert len(regions) == 1, 'caption sample should only contain one object'
            oid = random.randint(0, 15)
            question = question.replace('<region>', f'[{oid}]')
            assert '<object' not in question and '<object' not in response
            prefix += f'[{oid}]: ' + ' '.join([(f'<{tid}>-<{tid + 1}> ' if is_video else '') + MEM_TOKEN
                                               for tid in regions[0]]) + '\n'
        else:
            for oid, tids in zip(oids, regions):
                prefix += f'[{oid}]: ' + ' '.join([(f'<{tid}>-<{tid + 1}> ' if is_video else '') + MEM_TOKEN
                                                   for tid in tids]) + '\n'

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

        questions = question if isinstance(question, list) else [question]
        responses = response if isinstance(response, list) else [response]

        for question, response in zip(questions, responses):
            if anno['source'] == 'osprey_detail_description':
                question = random.choice(REG_DETAILED_PROMPTS).format(f"[{question['oid']}]")

            if len(messages) == 1:
                messages[0]['content'].append({'type': 'text', 'text': prefix + question})
            else:
                messages.append({'role': 'user', 'content': question})

            messages.append({'role': 'assistant', 'content': response})

        meta = dict(messages=messages, refer_mask=refer_mask)
        return meta


@DATASETS.register(name='videorefer_short_caption')
class VideoReferShortCaptionDataset(RegionDataset):

    ANNO_PATH = 'data/videorefer/VideoRefer-700K/videorefer-short-caption-500k-wo-masks.json'

    DATA_ROOT = 'data/videorefer/VideoRefer-700K/videos'
    MASK_ROOT = 'data/videorefer/VideoRefer-700K/masks_short_caption'

    SOURCE = 'videorefer_short_caption'

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

            # data sanity check
            if self.SOURCE == 'videorefer_detailed_caption':
                assert question.count('<region>') == 1
            elif self.SOURCE == 'videorefer_qa':
                assert question.count('\n') in (0, 1)
                if '\n' in question:
                    assert question.startswith('There')
                    assert '<object' in question
                else:
                    assert question.count('<region>') == 1
                    assert '<object' not in question

            # obj_frame_inds: num_objs * num_frames_per_obj
            # all_frame_inds: num_frames
            obj_frame_inds = [sorted([int(k) for k in o.keys()]) for o in raw_anno['annotation']]
            all_frame_inds = sorted(list(set(nncore.flatten(obj_frame_inds))))

            # videorefer_short_caption: single_frame_mode for all samples
            # videorefer_detailed_caption and videorefer_qa: single_frame_mode for around half of the samples
            single_frame_mode = self.SOURCE == 'videorefer_short_caption' or len(all_frame_inds) == 1

            anno = dict(
                source=self.SOURCE,
                data_type='region_video',
                video_path=nncore.join(self.DATA_ROOT, raw_anno['video']),
                vid=nncore.pure_name(raw_anno['video']),
                qid=raw_anno['id'],
                obj_frame_inds=obj_frame_inds,
                all_frame_inds=all_frame_inds,
                single_frame_mode=single_frame_mode,
                question=question,
                response=response,
                mask_type='rle',
                masks=nncore.join(self.MASK_ROOT, f'{idx}.json'))

            annos.append(anno)

        return annos


@DATASETS.register(name='videorefer_detailed_caption')
class VideoReferDetailedCaptionDataset(VideoReferShortCaptionDataset):

    ANNO_PATH = 'data/videorefer/VideoRefer-700K/videorefer-detailed-caption-125k-wo-masks.json'

    MASK_ROOT = 'data/videorefer/VideoRefer-700K/masks_detailed_caption'

    SOURCE = 'videorefer_detailed_caption'


@DATASETS.register(name='videorefer_qa')
class VideoReferQADataset(VideoReferShortCaptionDataset):

    ANNO_PATH = 'data/videorefer/VideoRefer-700K/videorefer-qa-75k-wo-masks.json'

    MASK_ROOT = 'data/videorefer/VideoRefer-700K/masks_qa'

    SOURCE = 'videorefer_qa'


@DATASETS.register(name='inst_it_video_qa_raw')
class InstITVideoQARawDataset(RegionDataset):

    ANNO_PATH = 'data/inst_it/Inst-IT-Dataset/inst_it_dataset_video_21k_w_mask.json'

    DATA_ROOT = 'data/inst_it/Inst-IT-Dataset/videos_raw'

    SOURCE = 'inst_it_video_qa_raw'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)

        annos = []
        for raw_anno in raw_annos:
            paths = nncore.ls(
                nncore.join(self.DATA_ROOT, raw_anno['video_path'].split('/', 1)[1]),
                ext=('png', 'jpg'),
                join_path=True)
            paths.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))

            # skip samples with more than 10 frames
            if len(paths) > 10:
                continue

            oids, masks, question, response, obj_inds = [], [], [], [], []
            for i, meta in enumerate(raw_anno['question_answer_pairs']):
                que, ans = meta['question'], meta['answer']

                # both oids and tids should start from 1
                assert '[0]' not in que and '[0]' not in ans

                # potential bugs in annotations
                que = que.replace('<0>', '<1>')
                for j in range(1, 10):
                    que = que.replace(f'[0{j}]', f'[{j}]')

                ans = ans.replace('<0>', '<1>')
                for j in range(1, 10):
                    ans = ans.replace(f'[0{j}]', f'[{j}]')

                matches = list(set(re.findall(r'\[\d+\]', que) + re.findall(r'\[\d+\]', ans)))

                # skip samples with no object
                if len(matches) == 0:
                    continue

                inds = set()
                for match in matches:
                    oid = match[1:-1]
                    assert oid == str(int(oid)), que

                    if oid in oids:
                        inds.add(oids.index(oid))
                        continue

                    oids.append(oid)
                    inds.add(len(oids) - 1)

                    obj_masks = []
                    for path in paths:
                        frame = nncore.base_name(path)
                        if frame in raw_anno['segmentations'] and oid in raw_anno['segmentations'][frame]:
                            obj_masks.append([raw_anno['segmentations'][frame][oid]['mask']])
                        else:
                            obj_masks.append([None])
                    masks.append(obj_masks)

                question.append(que)
                response.append(ans)
                obj_inds.append(list(inds))

            assert len(oids) == len(set(oids)) == len(masks) and len(question) == len(response) == len(obj_inds)

            if len(oids) > 5 or len(question) == 0:
                continue

            anno = dict(
                source=self.SOURCE,
                data_type='region_video',
                frames=paths,
                vid=raw_anno['video_id'],
                question=question,
                response=response,
                obj_inds=obj_inds,
                oids=oids,
                mask_type='rle',
                masks=masks)

            annos.append(anno)

        flattened_annos = []
        for anno in annos:
            for i in range(len(anno['question'])):
                flattened_anno = copy.deepcopy(anno)
                flattened_anno['question'] = [flattened_anno['question'][i]]
                flattened_anno['response'] = [flattened_anno['response'][i]]
                flattened_anno['oids'] = [flattened_anno['oids'][j] for j in flattened_anno['obj_inds'][i]]
                flattened_anno['masks'] = [flattened_anno['masks'][j] for j in flattened_anno['obj_inds'][i]]
                flattened_anno['obj_inds'] = [list(range(len(flattened_anno['oids'])))]
                flattened_annos.append(flattened_anno)
        annos = flattened_annos

        return annos


@DATASETS.register(name='osprey_conversation')
class OspreyConversationDataset(RegionDataset):

    ANNO_PATH = 'data/osprey/Osprey-724K/osprey_conversation.json'

    DATA_ROOT = 'data/osprey/coco/train2014'

    SOURCE = 'osprey_conversation'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH)

        annos, inv_cnt = [], 0
        for raw_anno in raw_annos:
            oids, masks, question, response = [], [], [], []

            invalid = False
            for i, conv in enumerate(raw_anno['conversations']):
                assert conv['from'] == 'human' if i % 2 == 0 else 'gpt'

                value = conv['value']

                # potential bugs in annotations
                if '<region>' in value:
                    invalid = True
                    break

                value = re.sub(r'<region(\d+)]', r'<region\1>', value)
                value = re.sub(r'<region(\d+).', r'<region\1>', value)
                value = re.sub(r'<region (\d+)>', r'<region\1>', value)
                value = re.sub(r'<region of(\d+)>', r'<region\1>', value)

                # potential bugs in annotations
                if '<region ' in value:
                    invalid = True
                    break

                matches = re.findall(r'<region\d+>', value)
                for match in matches:
                    oid = match[7:-1]
                    assert oid == str(int(oid)), value

                    if int(oid) > len(raw_anno['annotation']):
                        invalid = True
                        break

                    if oid not in oids:
                        masks.append([[raw_anno['annotation'][int(oid) - 1]['segmentation']]])
                        oids.append(oid)

                    value = value.replace(match, f'[{oid}]', 1)

                if invalid:
                    break

                assert '<region' not in value, value

                if i % 2 == 0:
                    question.append(value)
                else:
                    response.append(value)

            if invalid or len(masks) == 0:
                inv_cnt += 1
                continue

            assert len(oids) == len(masks) > 0

            anno = dict(
                source=self.SOURCE,
                data_type='region_image',
                frames=[nncore.join(self.DATA_ROOT, raw_anno['file_name'])],
                vid=nncore.pure_name(raw_anno['file_name']),
                qid=raw_anno['id'],
                question=question,
                response=response,
                oids=oids,
                mask_type='rle',
                masks=masks,
                height=raw_anno['height'],
                width=raw_anno['width'])

            annos.append(anno)

        print(f'Invalid samples ({self.SOURCE}): {inv_cnt} / {len(annos)} ({inv_cnt / len(annos) * 100:.2f}%)')

        return annos


@DATASETS.register(name='osprey_pos_neg')
class OspreyPosNegDataset(OspreyConversationDataset):

    ANNO_PATH = 'data/osprey/Osprey-724K/osprey_lvis_positive_negative.json'

    DATA_ROOT = 'data/osprey/coco/imgs'

    SOURCE = 'osprey_pos_neg'


@DATASETS.register(name='osprey_detail_description')
class OspreyDetailDescriptionDataset(RegionDataset):

    ANNO_PATH = 'data/osprey/Osprey-724K/osprey_detail_description.json'

    DATA_ROOT = 'data/osprey/coco/train2014'

    SOURCE = 'osprey_detail_description'

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH)

        annos, inv_cnt = [], 0
        for raw_anno in raw_annos:
            oids, masks, question, response = [], [], [], []

            invalid = False
            for desc in raw_anno['description']:
                # potential bugs in annotations
                desc = re.sub(r'<Region(\d+)>', r'<region\1>', desc)
                desc = re.sub(r'<regin(\d+)>', r'<region\1>', desc)
                desc = re.sub(r'<regin (\d+)>', r'<region\1>', desc)
                desc = desc.strip()

                assert desc.startswith('<region')

                matches = re.findall(r'<region(\d+)>', desc)
                if len(matches) != 1:
                    invalid = True
                    break

                oid = matches[0]
                assert oid == str(int(oid)), desc

                if int(oid) > len(raw_anno['annotation']):
                    invalid = True
                    break

                if oid not in oids:
                    masks.append([[raw_anno['annotation'][int(oid) - 1]['segmentation']]])
                    oids.append(oid)

                desc = re.findall(r'<region\d+>: (.*)', desc)[0]
                assert len(desc) > 0

                question.append(dict(oid=oid))
                response.append(desc)

            if invalid or len(masks) == 0:
                inv_cnt += 1
                continue

            assert len(oids) == len(masks) > 0

            anno = dict(
                source=self.SOURCE,
                data_type='region_image',
                frames=[nncore.join(self.DATA_ROOT, raw_anno['file_name'])],
                vid=nncore.pure_name(raw_anno['file_name']),
                qid=raw_anno['id'],
                question=question,
                response=response,
                oids=oids,
                mask_type='rle',
                masks=masks,
                height=raw_anno['height'],
                width=raw_anno['width'])

            annos.append(anno)

        print(f'Invalid samples ({self.SOURCE}): {inv_cnt} / {len(annos)} ({inv_cnt / len(annos) * 100:.2f}%)')

        return annos
