# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import warnings

import cv2
import nncore
import numpy as np
from PIL import Image

NUM_WOEKERS = 128

warnings.filterwarnings("ignore", category=RuntimeWarning)


def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])

    return M, O, D


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

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

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


def time_str_to_seconds(time_str):
    """Converts a time string to seconds."""
    parts = time_str.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def eval_queue(q, rank, out_dict, pred_path, video_root):
    while not q.empty():
        video, exp_idx = q.get()

        expressions = metadata[video]["questions"]
        expression_list = list(expressions.keys())

        clip_start = video[-9:].split("_")[0][:2] + ":" + video[-9:].split("_")[0][2:]
        # clip_end = video[-9:].split("_")[1][:2] + ":" + video[-9:].split("_")[1][2:]

        # read all the anno meta
        meta = {}
        meta["video_id"] = video
        meta["exp"] = expressions[expression_list[exp_idx]]["question"]
        meta["ans"] = expressions[expression_list[exp_idx]]["answer"]
        meta["obj_id"] = int(expressions[expression_list[exp_idx]]["obj_id"])
        meta["q_type"] = expressions[expression_list[exp_idx]]["q_type"]
        meta["exp_id"] = expression_list[exp_idx]

        start = expressions[expression_list[exp_idx]]["action_start"]
        end = expressions[expression_list[exp_idx]]["action_end"]
        action_start = (time_str_to_seconds(start) - time_str_to_seconds(clip_start)) * 6  # fps=6
        action_end = (time_str_to_seconds(end) - time_str_to_seconds(clip_start)) * 6 - 1

        meta["action_start"] = action_start
        meta["action_end"] = action_end
        # meta["frame_dir"] = frame_start.zfill(4) + "_" + frame_end.zfill(4)

        # 2. For each expression
        video_id = meta["video_id"]
        exp_id = meta["exp_id"]
        obj_id = meta["obj_id"]
        q_type = meta["q_type"]

        # action start and end is used to obtain gt masks in temporal dimension
        action_start = meta["action_start"]
        action_end = meta["action_end"]

        frame_dir = os.path.join(video_root, video_id, "images/")
        if not os.path.exists(frame_dir):
            print("Missing frames: {}.".format(video_id))
            continue
        raw_frames = nncore.ls(frame_dir, ext='jpg')  # all the frames
        has_frm = nncore.pure_name(raw_frames[0]).startswith('frame_')
        sample_indices = np.linspace(0, len(raw_frames) - 1, num=20, dtype=int)  # uniformly sample 20 frames
        assert len(sample_indices) == 20

        preds = nncore.ls(f'{args.pred_path}/{video_id}/{exp_id}', ext='png', join_path=True)
        preds.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))
        # assert len(preds) == len(sample_indices), (f'{video_id}/{exp_id}', len(preds), len(sample_indices))

        pred_0 = cv2.imread(preds[0], cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        origin_h, origin_w = pred_0.shape
        all_pred_masks = np.zeros((len(sample_indices), h, w), dtype=np.uint8)

        for frame_idx, index in enumerate(sample_indices):
            mask_id = "frame_" + str(index).zfill(6) + ".png" if has_frm else str(index).zfill(7) + ".png"
            if nncore.is_file(f'{args.pred_path}/{video_id}/{exp_id}/{mask_id}'):
                all_pred_masks[frame_idx] = cv2.imread(f'{args.pred_path}/{video_id}/{exp_id}/{mask_id}',
                                                       cv2.IMREAD_GRAYSCALE)

        all_pred_masks[all_pred_masks > 0] = 1

        # load gt masks
        mask_dir = os.path.join(video_root, video_id, "masks/")
        gt_masks_list = []
        for index in sample_indices:
            if action_start <= index <= action_end:
                mask_id = "frame_" + str(index).zfill(6) + ".png" if has_frm else str(index).zfill(7) + ".png"
                mask_path = os.path.join(mask_dir, mask_id)
                if os.path.exists(mask_path):
                    raw_mask = Image.open(mask_path).convert('P')
                else:
                    raw_mask = np.zeros((origin_h, origin_w), dtype=np.int32)
                raw_mask = np.array(raw_mask)
                gt_mask = (raw_mask == obj_id).astype(np.float32)
            else:
                gt_mask = np.zeros((origin_h, origin_w), dtype=np.int32)

            gt_masks_list.append(gt_mask)  # list[mask]
        gt_masks = np.stack(gt_masks_list, axis=0)

        # calculate J & F
        # print(gt_masks.shape, all_pred_masks.shape, gt_masks.min(), gt_masks.max(), all_pred_masks.min(),
        #       all_pred_masks.max())
        j_metric = db_eval_iou(gt_masks, all_pred_masks)
        f_metric = db_eval_boundary(gt_masks, all_pred_masks)
        [JM, JR, JD] = db_statistics(j_metric)
        [FM, FR, FD] = db_statistics(f_metric)

        # print(video_id, JM, FM)
        # JF = (JM + FM) / 2
        # print(JF)

        out_dict[f'{video_id}_{exp_id}_{obj_id}_{exp_idx}_{video}'] = [JM, FM, q_type]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_path')
    args = parser.parse_args()

    # load data
    video_root = "data/groundmore/annotations"
    meta_file = "data/groundmore/test_v2.json"

    with open(meta_file, "r") as f:
        metadata = json.load(f)["videos"]

    video_list = list(metadata.keys())

    queue = mp.Queue()

    shared_metadata = mp.Manager().dict(metadata)
    output_dict = mp.Manager().dict()

    cnt = 0
    for video in video_list:
        expressions = metadata[video]["questions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for exp_idx in range(num_expressions):
            queue.put([video, exp_idx])
            cnt += 1

    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.pred_path, video_root))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    assert len(output_dict) == cnt
    print(cnt)

    J_list = [output_dict[x][0] for x in output_dict]
    F_list = [output_dict[x][1] for x in output_dict]

    causal_J_list = [output_dict[x][0] for x in output_dict if output_dict[x][2] == "Causal"]
    causal_F_list = [output_dict[x][1] for x in output_dict if output_dict[x][2] == "Causal"]

    sequential_J_list = [output_dict[x][0] for x in output_dict if output_dict[x][2] == "Sequential"]
    sequential_F_list = [output_dict[x][1] for x in output_dict if output_dict[x][2] == "Sequential"]

    counterfactual_J_list = [output_dict[x][0] for x in output_dict if output_dict[x][2] == "Counterfactual"]
    counterfactual_F_list = [output_dict[x][1] for x in output_dict if output_dict[x][2] == "Counterfactual"]

    descriptive_J_list = [output_dict[x][0] for x in output_dict if output_dict[x][2] == "Descriptive"]
    descriptive_F_list = [output_dict[x][1] for x in output_dict if output_dict[x][2] == "Descriptive"]

    final_J = np.mean(J_list)
    final_F = np.mean(F_list)
    final_JF = (final_J + final_F) / 2

    final_causal_J = np.mean(causal_J_list)
    final_causal_F = np.mean(causal_F_list)
    final_sequential_J = np.mean(sequential_J_list)
    final_sequential_F = np.mean(sequential_F_list)
    final_counterfactual_J = np.mean(counterfactual_J_list)
    final_counterfactual_F = np.mean(counterfactual_F_list)
    final_descriptive_J = np.mean(descriptive_J_list)
    final_descriptive_F = np.mean(descriptive_F_list)

    final_causal_JF = (final_causal_J + final_causal_F) / 2
    final_sequential_JF = (final_sequential_J + final_sequential_F) / 2
    final_counterfactual_JF = (final_counterfactual_J + final_counterfactual_F) / 2
    final_descriptive_JF = (final_descriptive_J + final_descriptive_F) / 2

    print(f"Final J (Jaccard Index): {final_J:.4f}")
    print(f"Final F (F-measure): {final_F:.4f}")
    print(f"Final JF (Average of J and F): {final_JF:.4f}\n")

    print(f"Final Causal J: {final_causal_J:.4f}")
    print(f"Final Causal F: {final_causal_F:.4f}")
    print(f"Final Causal JF (Average of Causal J and F): {final_causal_JF:.4f}\n")

    print(f"Final Sequential J: {final_sequential_J:.4f}")
    print(f"Final Sequential F: {final_sequential_F:.4f}")
    print(f"Final Sequential JF (Average of Sequential J and F): {final_sequential_JF:.4f}\n")

    print(f"Final Counterfactual J: {final_counterfactual_J:.4f}")
    print(f"Final Counterfactual F: {final_counterfactual_F:.4f}")
    print(f"Final Counterfactual JF (Average of Counterfactual J and F): {final_counterfactual_JF:.4f}\n")

    print(f"Final Descriptive J: {final_descriptive_J:.4f}")
    print(f"Final Descriptive F: {final_descriptive_F:.4f}")
    print(f"Final Descriptive JF (Average of Descriptive J and F): {final_descriptive_JF:.4f}")

    results = {
        'final_J': final_J,
        'final_F': final_F,
        'final_JF': final_JF,
        'final_causal_J': final_causal_J,
        'final_causal_F': final_causal_F,
        'final_causal_JF': final_causal_JF,
        'final_sequential_J': final_sequential_J,
        'final_sequential_F': final_sequential_F,
        'final_sequential_JF': final_sequential_JF,
        'final_counterfactual_J': final_counterfactual_J,
        'final_counterfactual_F': final_counterfactual_F,
        'final_counterfactual_JF': final_counterfactual_JF,
        'final_descriptive_J': final_descriptive_J,
        'final_descriptive_F': final_descriptive_F,
        'final_descriptive_JF': final_descriptive_JF,
    }

    output_path = os.path.join(args.pred_path, 'groundmore.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
