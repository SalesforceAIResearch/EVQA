# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import time
import warnings

import cv2
import numpy as np
from pycocotools import mask as cocomask

NUM_WOEKERS = 128

warnings.filterwarnings('ignore', category=RuntimeWarning)


def db_eval_iou(annotation, segmentation, void_pixels=None):
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[
                frame_id,
                :,
                :,
            ]
            f_res[frame_id] = f_measure(
                segmentation[
                    frame_id,
                    :,
                    :,
                ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) >
                0.01), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def eval_queue(q, rank, out_dict, pred_path):
    while not q.empty():
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        if not os.path.exists(f'{pred_path}/{vid_name}'):
            print(f'{vid_name} not found')
            out_dict[exp_name] = [0, 0]
            continue

        pred_0_path = f"{pred_path}/{vid_name}/{exp}/{vid['frames'][0]}.png"
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_masks[frame_idx] = cv2.imread(f'{pred_path}/{vid_name}/{exp}/{frame_name}.png', cv2.IMREAD_GRAYSCALE)

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


def get_meta_exp(exp_path):
    with open(str(exp_path), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = sorted(list(subset_expressions_by_video.keys()))

    anno_count = 0  # serve as anno_id
    for vid in videos:
        vid_data = subset_expressions_by_video[vid]
        exp_id_list = sorted(list(vid_data['expressions'].keys()))
        for exp_id in exp_id_list:
            subset_expressions_by_video[vid]['expressions'][exp_id]['anno_id'] = [anno_count]
            anno_count += 1

    return subset_expressions_by_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('pred_path')
    args = parser.parse_args()

    queue = mp.Queue()

    if args.dataset == 'mevis':
        exp_dict = json.load(open('data/mevis/valid_u/meta_expressions.json'))['videos']
        mask_dict = json.load(open('data/mevis/valid_u/mask_dict.json'))
    elif args.dataset == 'ref_sav':
        exp_dict = json.load(open('data/ref_sav/valid/meta_expressions_valid.json'))['videos']
        mask_dict = json.load(open('data/ref_sav/valid/mask_dict.json'))
    elif args.dataset == 'ref_davis17':
        exp_dict = get_meta_exp('data/ref_davis17/meta_expressions/valid/meta_expressions.json')
        mask_dict = pickle.load(open('data/ref_davis17/valid/mask_dict.pkl', 'rb'))
    else:
        raise KeyError(f'unknown dataset: {args.dataset}')

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.pred_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    output_path = os.path.join(args.pred_path, f'{args.dataset}.json')
    results = {
        'J': round(100 * float(np.mean(j)), 2),
        'F': round(100 * float(np.mean(f)), 2),
        'J&F': round(100 * float((np.mean(j) + np.mean(f)) / 2), 2)
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

    end_time = time.time()
    total_time = end_time - start_time
    print('time: %.4f s' % (total_time))
