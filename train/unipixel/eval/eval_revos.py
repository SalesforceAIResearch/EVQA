# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import json
import math
import multiprocessing as mp
import os
import os.path as osp
import time
import traceback
import warnings
from typing import List

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as cocomask

NUM_WOEKERS = 128

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_r2vos_accuracy(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
):
    """ Compute per-pixel accuracy.
    Args:
        gt_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
        pred_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
    Return:
        accs: np.ndarray, shape: (n_frames,), dtype: np.float32
    """
    assert len(gt_masks) == len(pred_masks), "The number of frames in gt_masks and pred_masks should be the same"
    accs = []
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        assert gt_mask.shape == pred_mask.shape, "The shape of gt_mask and pred_mask should be the same"
        acc = np.mean(gt_mask == pred_mask)
        accs.append(acc)
    return np.array(accs)


def get_r2vos_robustness(
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    foreground_masks: List[np.ndarray],
):
    """
    Args:
        gt_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
        pred_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
        foreground_masks: List[np.ndarray], shape: (n_frames, h, w), dtype: np.uint8
    Return:
        robustness: np.ndarray, shape: (n_frames,), dtype: np.float32
    """
    assert len(gt_masks) == len(pred_masks) == len(
        foreground_masks), "The number of frames in gt_masks, pred_masks and foreground_masks should be the same"
    robustness = []
    for gt_mask, pred_mask, foreground_mask in zip(gt_masks, pred_masks, foreground_masks):
        assert gt_mask.shape == pred_mask.shape == foreground_mask.shape, "The shape of gt_mask, pred_mask and foreground_mask should be the same"
        neg = ((1 - gt_mask) * pred_mask).sum()
        pos = foreground_mask.sum()
        robust = max(1 - neg / (pos + 1e-6), 0.0)
        robustness.append(robust)
    return np.array(robustness)


def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
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
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
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
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) >
                0.01), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

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


def eval_queue(q, rank, out_dict, visa_pred_path):
    while not q.empty():
        vid_name, exp = q.get()
        vid = exp_dict[vid_name]
        exp_name = f'{vid_name}_{exp}'

        try:

            if not os.path.exists(f'{visa_pred_path}/{vid_name}'):
                print(f'{vid_name} not found')
                out_dict[exp_name] = [0, 0, 0, 0]
                continue

            pred_0_path = f'{visa_pred_path}/{vid_name}/{exp}/{vid["frames"][0]}.png'
            pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
            h, w = pred_0.shape
            vid_len = len(vid['frames'])
            gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
            pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
            foreground_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

            anno_ids = vid['expressions'][exp]['anno_id']

            for frame_idx, frame_name in enumerate(vid['frames']):
                # all instances in the same frame
                for anno_id in anno_ids:
                    mask_rle = mask_dict[str(anno_id)][frame_idx]
                    if mask_rle:
                        gt_masks[frame_idx] += cocomask.decode(mask_rle)

                # foreground mask
                mask_fore_rle = mask_dict_foreground[vid_name]["masks_rle"][frame_idx]
                mask_fore = cocomask.decode(mask_fore_rle)
                mask_fore = mask_fore.sum(
                    axis=2).astype(np.uint8) if mask_fore.ndim == 3 else mask_fore.astype(np.uint8)
                foreground_masks[frame_idx] = mask_fore

                pred_masks[frame_idx] = cv2.imread(f'{visa_pred_path}/{vid_name}/{exp}/{frame_name}.png',
                                                   cv2.IMREAD_GRAYSCALE)

            j = db_eval_iou(gt_masks, pred_masks).mean()
            f = db_eval_boundary(gt_masks, pred_masks).mean()
            a = get_r2vos_accuracy(gt_masks, pred_masks).mean()
            r = get_r2vos_robustness(gt_masks, pred_masks, foreground_masks).mean()

            out_dict[exp_name] = [j, f, a, r]
        except Exception:
            print(f'error: {exp_name}, {traceback.format_exc()}', 'red')
            out_dict[exp_name] = [0.0, 0.0, 0.0, 0.0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("visa_pred_path", type=str)
    parser.add_argument("--visa_exp_path", type=str, default="data/revos/meta_expressions_valid_.json")
    parser.add_argument("--visa_mask_path", type=str, default="data/revos/mask_dict.json")
    parser.add_argument("--visa_foreground_mask_path", type=str, default="data/revos/mask_dict_foreground.json")
    parser.add_argument("--save_json_name", type=str, default="revos_valid.json")
    parser.add_argument("--save_csv_name", type=str, default="revos_valid.csv")
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = json.load(open(args.visa_exp_path))['videos']
    mask_dict = json.load(open(args.visa_mask_path))
    mask_dict_foreground = json.load(open(args.visa_foreground_mask_path, 'r'))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    shared_mask_dict_foreground = mp.Manager().dict(mask_dict_foreground)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.visa_pred_path))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict, args.visa_pred_path)

    # save average results
    output_json_path = osp.join(args.visa_pred_path, args.save_json_name)
    output_csv_path = osp.join(args.visa_pred_path, args.save_csv_name)

    data_list = []
    for videxp, (j, f, a, r) in output_dict.items():
        vid_name, exp = videxp.rsplit('_', 1)
        data = {}

        data['video_name'] = vid_name
        data['exp_id'] = exp
        data['exp'] = exp_dict[vid_name]['expressions'][exp]['exp']
        data['videxp'] = videxp
        data['J'] = round(100 * j, 2)
        data['F'] = round(100 * f, 2)
        data['JF'] = round(100 * (j + f) / 2, 2)
        data['A'] = round(100 * a, 2)
        data['R'] = round(100 * r, 2)
        data['type_id'] = exp_dict[vid_name]['expressions'][exp]['type_id']

        data_list.append(data)

    j_referring = np.array([d['J'] for d in data_list if d['type_id'] == 0]).mean()
    f_referring = np.array([d['F'] for d in data_list if d['type_id'] == 0]).mean()
    a_referring = np.array([d['A'] for d in data_list if d['type_id'] == 0]).mean()
    r_referring = np.array([d['R'] for d in data_list if d['type_id'] == 0]).mean()
    jf_referring = (j_referring + f_referring) / 2

    j_reason = np.array([d['J'] for d in data_list if d['type_id'] == 1]).mean()
    f_reason = np.array([d['F'] for d in data_list if d['type_id'] == 1]).mean()
    a_reason = np.array([d['A'] for d in data_list if d['type_id'] == 1]).mean()
    r_reason = np.array([d['R'] for d in data_list if d['type_id'] == 1]).mean()
    jf_reason = (j_reason + f_reason) / 2

    j_referring_reason = (j_referring + j_reason) / 2
    f_referring_reason = (f_referring + f_reason) / 2
    a_referring_reason = (a_referring + a_reason) / 2
    r_referring_reason = (r_referring + r_reason) / 2
    jf_referring_reason = (jf_referring + jf_reason) / 2

    results = {
        "referring": {
            "J": j_referring,
            "F": f_referring,
            "A": a_referring,
            "R": r_referring,
            "JF": jf_referring
        },
        "reason": {
            "J": j_reason,
            "F": f_reason,
            "A": a_reason,
            "R": r_reason,
            "JF": jf_reason
        },
        "overall": {
            "J": j_referring_reason,
            "F": f_referring_reason,
            "A": a_referring_reason,
            "R": r_referring_reason,
            "JF": jf_referring_reason
        }
    }

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json_path}")

    data4csv = {}
    for data in data_list:
        for k, v in data.items():
            data4csv[k] = data4csv.get(k, []) + [v]

    df = pd.DataFrame(data4csv)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))
