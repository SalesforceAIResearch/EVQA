# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import copy
import io

import cv2
import imageio.v3 as iio
import nncore
import numpy as np
import torch
from nncore.ops import bbox_iou
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from unipixel.dataset.hybrid import DATASETS
from unipixel.dataset.utils import process_masks, process_vision_info
from unipixel.model.builder import build_model
from unipixel.utils.io import load_frames
from unipixel.utils.transforms import get_sam2_transform
from unipixel.utils.visualizer import Visualizer, random_color


def get_min_box(tensor, min_area=10):
    binary_img = tensor.cpu().numpy().astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    filtered_img = np.zeros_like(binary_img)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_img[labels == i] = 255

    coords = np.column_stack(np.where(filtered_img > 0))
    if coords.shape[0] == 0:
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    box = torch.Tensor([[x_min, y_min, x_max, y_max]])
    return box


def compute_iou_volume(pred, gt):
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    intersection = (pred_flat & gt_flat).sum().float()
    union = (pred_flat | gt_flat).sum().float()
    return (intersection / (union + 1e-6)).item()


def compute_f_score_volume(pred, gt):
    pred_flat = pred.view(-1)
    gt_flat = gt.view(-1)
    tp = (pred_flat & gt_flat).sum().float()
    fp = (pred_flat & ~gt_flat).sum().float()
    fn = (~pred_flat & gt_flat).sum().float()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return (2 * precision * recall / (precision + recall + 1e-6)).item()


def compute_j_and_f_volume(pred, gt):
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    j = compute_iou_volume(pred, gt)
    f = compute_f_score_volume(pred, gt)
    return j, f, (j + f) / 2


def collate(batch):
    return batch[0]


class EvalDataset(Dataset):
    def __init__(self, annos, sam2_transform, sample_frames, dataset_name=''):
        super().__init__()
        self.annos = annos
        self.sam2_transform = sam2_transform
        self.sample_frames = sample_frames
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annos[idx])

        # For st_evidence datasets: update anno to only include sampled frames
        is_st_evidence = 'st_evidence' in self.dataset_name

        frames, paths, inds = load_frames(anno['frames'], sample_frames=self.sample_frames, sample_for_llm_only=True)
        frame_size = frames.shape[1:3]

        # For st_evidence: update anno to only include sampled frames
        if is_st_evidence:
            anno['frames'] = paths
            anno['sampled_inds'] = inds

        assert len(anno['samples']) == 1
        sample = anno['samples'][0]

        if sample['type'] == 'query':
            media = 'image' if 'image' in anno['data_type'] else 'video'
            query = sample['query'].strip('.')
            query = query[0].lower() + query[1:]
            for key in ('the ', 'an ', 'a '):
                if query.startswith(key):
                    query = query[len(key):]
            question = f'Please segment the {query} in this {media}.'
        elif sample.get('type') == 'st_evidence':
            question = f"{sample['query']} Answer the question and provide evidence in the form of temporal ([[start1, end1], [start2, end2], ...]) and spatial evidence (masks)."
        elif sample['type'] == 'sentence':
            question = f"{sample['query']} Please provide the segmentation mask."
        elif sample['type'] == 'explanatory':
            question = f"{sample['query']} Please provide the segmentation mask and explain why."
        elif sample['type'] == 'question':
            question = f"{sample['query']} Please answer the question and provide the segmentation mask."
        elif sample['type'] == 'description':
            question = f"Please segment the object according to the description: {sample['query']}"
        else:
            raise KeyError(f'unknown sample type: {sample}')

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': paths,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28 * int(self.sample_frames / len(paths))
            }, {
                'type': 'text',
                'text': question
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)

        data = processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)

        data['frames'] = [self.sam2_transform(frames)]
        data['frame_size'] = [frame_size]

        data['question'] = question
        data['raw_frames'] = frames
        data['anno'] = anno  # Use the potentially modified anno (with sampled frames for st_evidence)

        return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--split')
    parser.add_argument('--model_path')
    parser.add_argument('--seg_pred_path')
    parser.add_argument('--vis_pred_path')
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--sample_frames', type=int, default=16)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dump', type=int, default=-1)
    parser.add_argument('--dump_thr', type=float, default=-1)
    parser.add_argument('--max_samples', type=int, default=-1,
                       help='Maximum number of samples to process (default: -1 for all, dataset already limits valid/test to 1000)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model, processor = build_model(args.model_path, image_size=args.image_size, device=args.device, dtype=args.dtype)
    device = next(model.parameters()).device

    sam2_transform = get_sam2_transform(model.config.sam2_image_size)

    annos = DATASETS.get(args.dataset).load_annos(split=args.split)
    print(f"Loaded {len(annos)} annotations")
    
    # Limit number of samples if specified
    if args.max_samples > 0:
        original_len = len(annos)
        annos = annos[:args.max_samples]
        print(f"Limited to first {len(annos)} samples (out of {original_len})")
    
    # print(annos[0])
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dataset = EvalDataset(annos, sam2_transform, args.sample_frames, args.dataset)
    data_loader = DataLoader(dataset, collate_fn=collate, num_workers=args.workers)

    collected, failed = [], []
    for i, data in enumerate(nncore.ProgressBar(data_loader)):
        anno, question, frames = data.pop('anno'), data.pop('question'), data.pop('raw_frames')

        sample, frame_size = anno['samples'][0], data['frame_size'][0]

        data = data.to(device)
        data['frames'] = [data['frames'][0].to(model.sam2.dtype)]

        try:
            output_ids = model.generate(
                **data,
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                repetition_penalty=None,
                max_new_tokens=512)

            assert data.input_ids.size(0) == output_ids.size(0) == 1
            output_ids = output_ids[0, data.input_ids.size(1):]

            if output_ids[-1] == processor.tokenizer.eos_token_id:
                output_ids = output_ids[:-1]

            response = processor.decode(output_ids, clean_up_tokenization_spaces=False)

            assert len(model.seg) >= 1, (len(model.seg), response, output_ids)

            if args.verbose:
                print()
                print(frames.size())
                print(question)
                print(response)
                print(len(model.seg))
                print()
        except Exception as e:
            # print(f'\nInference failed ({i} of rank {args.index}/{args.chunk}): {anno} ({e})')
            print(f'\nInference failed ({i} of rank {args.index}/{args.chunk}): ({e})')
            print(getattr(model, 'seg', 'not found'))
            model.seg = [torch.zeros(1, len(anno['frames']), *frame_size)]
            # For st_evidence: set empty response to avoid NameError when saving text responses
            if 'st_evidence' in args.dataset:
                response = ""
            failed.append(dict(anno=anno, err=str(e)))

        assert model.seg[0].size(0) == 1
        assert model.seg[0].size(1) == frames.size(0) == len(anno['frames'])

        if any(
                key in args.dataset
                for key in ('revos', 'mevis', 'ref_youtube_vos', 'ref_davis17', 'ref_sav', 'groundmore', 'st_evidence')):
            out = model.seg[0][0].to(torch.uint8)
            out[out > 0] = 255
            out = out.numpy()
            for frm_idx in range(model.seg[0].size(1)):
                path = nncore.join(args.seg_pred_path, anno['vid'], sample['qid'],
                                   nncore.pure_name(anno['frames'][frm_idx]) + '.png')
                nncore.imwrite(out[frm_idx], path)
            
            # For st_evidence, also save text responses
            if 'st_evidence' in args.dataset:
                collected.append(dict(
                    qid=sample['qid'],
                    vid=anno['vid'],
                    response=response,
                    task=sample.get('task', 'st_evidence'),
                    type=sample.get('type', 'st_evidence')
                ))
        elif any(key in args.dataset for key in ('refcoco', 'refcoco+', 'refcocog', 'reason_seg')):
            masks = process_masks(sample, frame_size, [0])
            assert len(masks) == 1  # one <seg> token
            assert len(masks[0]) == model.seg[0][0].size(0) == 1  # one frame
            keep = masks[0][0] != -1
            mask = masks[0][0].bool()
            pred = model.seg[0][0][0].bool()
            assert keep.size() == mask.size() == pred.size()
            inter = ((mask * pred) * keep).sum().item()
            union = ((mask + pred) * keep).sum().item()
            iou = 1 if union == 0 else inter / union
            out = dict(vid=anno['vid'], qid=sample['qid'], inter=inter, union=union, iou=iou)
            if 'boxes' in sample:
                assert len(sample['boxes']) == 1, sample['boxes']
                box = get_min_box(pred)
                out['hit'] = False if box is None else bbox_iou(box, torch.Tensor(sample['boxes'][0])).item() >= 0.5
            collected.append(out)
        else:
            raise KeyError(f'unknown dataset: {args.dataset}')

        if args.dump > 0 and i % args.dump == 0:
            if 'masks' in sample and 'mask_type' in sample and args.dump_thr > 0:
                num_masks = len(sample['masks']) if sample['mask_type'] == 'image' else len(sample['masks'][0])

                if num_masks != frames.size(0):
                    print(f"Skip incorrect ground truth: {num_masks} {frames.size(0)}")
                    continue

                masks = process_masks(sample, frame_size, range(frames.size(0)))

                jf_scores = []
                for k in range(len(masks)):
                    mask = torch.stack(masks[k]).bool()
                    pred = model.seg[0][k].bool()
                    _, _, jf_score = compute_j_and_f_volume(pred, mask)
                    jf_scores.append(jf_score)

                jf_mean = sum(jf_scores) / len(jf_scores)
                if jf_mean < args.dump_thr:
                    continue

                jf_str = f'_{jf_mean:.3f}'
            else:
                jf_str = ''

            colors = [random_color(rgb=True, maximum=1) for _ in range(model.seg[0].size(0))]

            sample_inds = np.linspace(0, frames.size(0) - 1, 6, dtype=int)

            gifs, jpgs = [], []
            for frm_idx in range(frames.size(0)):
                vis = Visualizer(frames[frm_idx].numpy())

                for obj_idx in range(model.seg[0].size(0)):
                    fig = vis.draw_binary_mask_with_number(
                        model.seg[0][obj_idx, frm_idx].bool().numpy(), color=colors[obj_idx], alpha=0.3)

                buffer = io.BytesIO()
                fig.save(buffer)
                buffer.seek(0)
                gif = iio.imread(buffer)
                gifs.append(gif)

                if frm_idx in sample_inds:
                    buffer = io.BytesIO()
                    fig.save(buffer)
                    buffer.seek(0)
                    jpg = Image.open(buffer).convert('RGB')
                    jpgs.append(jpg)
                    buffer.close()

            nncore.mkdir(args.vis_pred_path)

            name = f"{args.index}_{i}_{anno['vid'].replace('/', '_')}"

            # 1: gif video with masks
            iio.imwrite(nncore.join(args.vis_pred_path, f'{name}{jf_str}.gif'), gifs, duration=100, loop=0)

            ws, hs = zip(*(jpg.size for jpg in jpgs))
            img = Image.new('RGB', (sum(ws), max(hs)), color=(255, 255, 255))

            offset = 0
            for jpg in jpgs:
                img.paste(jpg, (offset, 0))
                offset += jpg.width

            # 2: jpg video with masks
            img.save(nncore.join(args.vis_pred_path, f'{name}{jf_str}.jpg'))

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"Rank {args.index}/{args.chunk} Summary:")
    print(f"  Total: {len(annos)}, Successful: {len(annos) - len(failed)}, Failed: {len(failed)}")
    print(f"{'=' * 80}")

    if len(failed) > 0:
        print(f'Number of failed samples (rank {args.index}/{args.chunk}): {len(failed)}')
        nncore.dump(failed, nncore.join(args.seg_pred_path, f'failed_{args.index}.json'))

    if any(key in args.dataset for key in ('refcoco', 'refcoco+', 'refcocog', 'reason_seg')):
        nncore.dump(collected, nncore.join(args.seg_pred_path, f'output_{args.index}.json'))
    
    # Save text responses for st_evidence
    if 'st_evidence' in args.dataset and len(collected) > 0:
        print(f'Saving {len(collected)} text responses for st_evidence (rank {args.index}/{args.chunk})')
        nncore.dump(collected, nncore.join(args.seg_pred_path, f'output_{args.index}.json'))