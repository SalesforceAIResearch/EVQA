# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random
import re

import decord
import nncore
import numpy as np
import pysrt
import torch
from decord import VideoReader
from PIL import Image


def load_image(path):
    image = Image.open(path).convert('RGB')
    image = torch.from_numpy(np.array(image)).unsqueeze(0)
    return image


def load_video(path, sample_frames=-1):
    frame_mode = nncore.is_dir(path)

    if frame_mode:
        paths = nncore.ls(path, ext=('jpg', 'png'), join_path=True)
        paths.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))
        vlen = len(paths)
    else:
        decord.bridge.set_bridge('torch')
        vr = VideoReader(path, num_threads=1)
        vlen = len(vr)

    if sample_frames > 0 and vlen > sample_frames:
        inds = np.arange(0, vlen, (vlen - 1) / (sample_frames - 1))[:sample_frames].round().astype(int).tolist()
        assert len(inds) == sample_frames
    else:
        inds = list(range(vlen))

    if frame_mode:
        images = [paths[i] for i in inds]
        frames = torch.cat([load_image(i) for i in images])
    else:
        frames = vr.get_batch(inds)
        images = [Image.fromarray(t.numpy()) for t in frames]

    return frames, images


def load_frames(paths, sample_frames=-1, sample_type='uniform', sample_for_llm_only=False):
    assert sample_type in ('uniform', 'random')

    vlen = len(paths)

    if isinstance(sample_frames, str):
        sep = [int(n) for n in sample_frames.split(',')]
        assert len(sep) in (1, 2)
        sample_frames = int(random.randint(*sep)) if len(sep) > 1 else int(sep[0])

    # NOTE: some videos and images are shorter than sample_frames
    if sample_frames > 0 and vlen > sample_frames:
        if sample_type == 'uniform':
            inds = np.arange(0, vlen, (vlen - 1) / (sample_frames - 1))[:sample_frames].round().astype(int).tolist()
        else:
            seps = np.arange(0, vlen, (vlen - 1) / sample_frames)[:sample_frames + 1].round().astype(int).tolist()
            inds = [random.choice(range(sep, max(sep + 1, seps[i + 1]))) for i, sep in enumerate(seps[:-1])]
        assert len(inds) == sample_frames
    else:
        inds = list(range(len(paths)))

    if sample_for_llm_only:
        frames = torch.cat([load_image(p) for p in paths])
    else:
        frames = torch.cat([load_image(paths[i]) for i in inds])

    paths = [paths[i] for i in inds]
    return frames, paths, inds


def load_frames_with_inds(path,
                          keep,
                          single_frame_mode=False,
                          sample_frames=-1,
                          sample_type='uniform',
                          sample_for_llm_only=False,
                          num_threads=0):
    assert sample_type in ('uniform', 'random')

    frame_mode = nncore.is_dir(path)

    if frame_mode:
        paths = nncore.ls(path, ext='jpg', join_path=True)
        paths.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))
    else:
        decord.bridge.set_bridge('torch')
        vr = VideoReader(path, num_threads=num_threads)

    if single_frame_mode:
        vlen = len(paths) if frame_mode else len(vr)
        assert vlen > 1 and len(keep) == 1
        imap = list(range(vlen))
    else:
        vlen = len(keep)
        imap = keep

    if isinstance(sample_frames, str):
        sep = [int(n) for n in sample_frames.split(',')]
        assert len(sep) in (1, 2)
        sample_frames = int(random.randint(*sep)) if len(sep) > 1 else int(sep[0])

    # some videos and images are shorter than sample_frames
    if sample_frames > 0 and vlen > sample_frames:
        if sample_type == 'uniform':
            inds = np.arange(0, vlen, (vlen - 1) / (sample_frames - 1))[:sample_frames].round().astype(int).tolist()
        else:
            seps = np.arange(0, vlen, (vlen - 1) / sample_frames)[:sample_frames + 1].round().astype(int).tolist()
            inds = [random.choice(range(sep, max(sep + 1, seps[i + 1]))) for i, sep in enumerate(seps[:-1])]

        if single_frame_mode:
            # ensure that keep is in the sampled indices
            dist = [abs(keep[0] - i) for i in inds]
            inds[dist.index(min(dist))] = keep[0]

        assert len(inds) == sample_frames
    else:
        inds = list(range(vlen))

    if frame_mode:
        images = [paths[imap[i]] for i in inds]
    else:
        img_tensor = vr.get_batch([imap[i] for i in inds])
        images = [Image.fromarray(t.numpy()) for t in img_tensor]

    if single_frame_mode:
        frames = load_image(paths[keep[0]]) if frame_mode else vr.get_batch(keep)
    elif sample_for_llm_only:
        frames = torch.cat([load_image(p) for p in paths]) if frame_mode else vr.get_batch(imap)
    else:
        frames = torch.cat([load_image(paths[imap[i]]) for i in inds]) if frame_mode else img_tensor.clone()

    return frames, images, inds


def load_frames_with_inds_keep(path,
                               all_frame_inds,
                               frame_idx,
                               sample_frames=-1,
                               sample_type='uniform',
                               sample_for_llm_only=False,
                               num_threads=0):
    assert sample_type in ('uniform', 'random')

    frame_mode = nncore.is_dir(path)

    if frame_mode:
        paths = nncore.ls(path, ext='jpg', join_path=True)
        paths.sort(key=lambda p: int(re.sub(r'^\D*', '', nncore.pure_name(p))))
    else:
        decord.bridge.set_bridge('torch')
        vr = VideoReader(path, num_threads=num_threads)

    vlen = len(all_frame_inds)
    imap = all_frame_inds

    if isinstance(sample_frames, str):
        sep = [int(n) for n in sample_frames.split(',')]
        assert len(sep) in (1, 2)
        sample_frames = int(random.randint(*sep)) if len(sep) > 1 else int(sep[0])

    # some videos and images are shorter than sample_frames
    if sample_frames > 0 and vlen > sample_frames:
        if sample_type == 'uniform':
            inds = np.arange(0, vlen, (vlen - 1) / (sample_frames - 1))[:sample_frames].round().astype(int).tolist()
        else:
            seps = np.arange(0, vlen, (vlen - 1) / sample_frames)[:sample_frames + 1].round().astype(int).tolist()
            inds = [random.choice(range(sep, max(sep + 1, seps[i + 1]))) for i, sep in enumerate(seps[:-1])]

        # ensure that keep is in the sampled indices
        keep = all_frame_inds.index(frame_idx)
        dist = [abs(keep - i) for i in inds]
        inds[dist.index(min(dist))] = keep

        assert len(inds) == sample_frames
    else:
        inds = list(range(vlen))

    if frame_mode:
        images = [paths[imap[i]] for i in inds]
    else:
        img_tensor = vr.get_batch([imap[i] for i in inds])
        images = [Image.fromarray(t.numpy()) for t in img_tensor]

    if sample_for_llm_only:
        frames = torch.cat([load_image(p) for p in paths]) if frame_mode else vr.get_batch(imap)
    else:
        frames = torch.cat([load_image(paths[imap[i]]) for i in inds]) if frame_mode else img_tensor.clone()

    return frames, images, inds


def load_frames_with_stride(path,
                            every_n_frames=4,
                            sample_frames=-1,
                            sample_type='uniform',
                            sample_for_llm_only=False,
                            num_threads=0):
    assert sample_type in ('uniform', 'random')

    decord.bridge.set_bridge('torch')
    vr = VideoReader(path, num_threads=num_threads)

    keep = list(range(0, len(vr), every_n_frames))
    vlen = len(keep)

    if isinstance(sample_frames, str):
        sep = [int(n) for n in sample_frames.split(',')]
        assert len(sep) in (1, 2)
        sample_frames = int(random.randint(*sep)) if len(sep) > 1 else int(sep[0])

    # some videos and images are shorter than sample_frames
    if sample_frames > 0 and vlen > sample_frames:
        if sample_type == 'uniform':
            inds = np.arange(0, vlen, (vlen - 1) / (sample_frames - 1))[:sample_frames].round().astype(int).tolist()
        else:
            seps = np.arange(0, vlen, (vlen - 1) / sample_frames)[:sample_frames + 1].round().astype(int).tolist()
            inds = [random.choice(range(sep, max(sep + 1, seps[i + 1]))) for i, sep in enumerate(seps[:-1])]
        assert len(inds) == sample_frames
    else:
        inds = list(range(vlen))

    img_tensor = vr.get_batch([keep[i] for i in inds])
    images = [Image.fromarray(t.numpy()) for t in img_tensor]
    frames = vr.get_batch(keep) if sample_for_llm_only else img_tensor.clone()

    return frames, images, inds


def load_subtitle(path):
    subs = pysrt.open(path)

    parsed = []
    for sub in subs:
        s, e = sub.start.to_time(), sub.end.to_time()
        s = (s.hour * 60 + s.minute) * 60 + s.second + s.microsecond / 1000000
        e = (e.hour * 60 + e.minute) * 60 + e.second + e.microsecond / 1000000
        parsed.append((s, e, sub.text))

    return parsed


def get_duration(path, num_threads=1):
    # sometimes the video is loaded as a list of frames
    if isinstance(path, list):
        return len(path)

    vr = VideoReader(path, num_threads=num_threads)
    duration = len(vr) / vr.get_avg_fps()
    return duration
