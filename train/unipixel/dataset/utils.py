# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import base64
import copy
import math
import os
import warnings
from io import BytesIO
from typing import Optional

import cv2
import decord
import nncore
import numpy as np
import requests
import torch
import torchvision.transforms.functional as T
from PIL import Image
from pycocotools.mask import decode, frPyObjects, merge
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from unipixel.constants import IGNORE_INDEX
from unipixel.conversation import get_conv

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int,
                 width: int,
                 factor: int = IMAGE_FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    # change order here to ensure not exceeding max_pixels
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def calculate_video_frame_range(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> tuple[int, int, int]:
    """
    Calculate the start and end frame indices based on the given time range.

    Args:
        ele (dict): A dictionary containing optional 'video_start' and 'video_end' keys (in seconds).
        total_frames (int): Total number of frames in the video.
        video_fps (float): Frames per second of the video.

    Returns:
        tuple: A tuple containing (start_frame, end_frame, frame_count).

    Raises:
        ValueError: If input parameters are invalid or the time range is inconsistent.
    """
    # Validate essential parameters
    if video_fps <= 0:
        raise ValueError("video_fps must be a positive number")
    if total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    # Get start and end time in seconds
    video_start = ele.get("video_start", None)
    video_end = ele.get("video_end", None)
    if video_start is None and video_end is None:
        return 0, total_frames - 1, total_frames

    max_duration = total_frames / video_fps
    # Process start frame
    if video_start is not None:
        video_start_clamped = max(0.0, min(video_start, max_duration))
        start_frame = math.ceil(video_start_clamped * video_fps)
    else:
        start_frame = 0
    # Process end frame
    if video_end is not None:
        video_end_clamped = max(0.0, min(video_end, max_duration))
        end_frame = math.floor(video_end_clamped * video_fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1

    # Validate frame order
    if start_frame >= end_frame:
        raise ValueError(
            f"Invalid time range: Start frame {start_frame} (at {video_start_clamped if video_start is not None else 0}s) "
            f"exceeds end frame {end_frame} (at {video_end_clamped if video_end is not None else max_duration}s). "
            f"Video duration: {max_duration:.2f}s ({total_frames} frames @ {video_fps}fps)")

    return start_frame, end_frame, end_frame - start_frame + 1


def _read_video_decord(ele: dict, ) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    decord.bridge.set_bridge("torch")
    video_path = ele["video"]
    vr = decord.VideoReader(video_path, num_threads=ele.get('num_threads', 0))
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    start_frame, end_frame, total_frames = calculate_video_frame_range(
        ele,
        total_frames,
        video_fps,
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    video = vr.get_batch(idx).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps


def fetch_video(ele: dict,
                image_factor: int = IMAGE_FACTOR,
                return_video_sample_fps: bool = False,
                sanity_check=False) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video, sample_fps = _read_video_decord(ele)
        nframes, _, height, width = video.shape
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels_supposed = ele.get("max_pixels", max_pixels)
        max_pixels = min(max_pixels_supposed, max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

        if sanity_check and (video == 0).all():
            raise ValueError("video '{}' contains all zeros".format(ele["video"]))

        if return_video_sample_fps:
            return video, sample_fps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({
                "image": video_element,
                **process_info
            }, size_factor=image_factor) for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0)
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ("image" in ele or "image_url" in ele or "video" in ele
                            or ele.get("type", "") in ("image", "image_url", "video")):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
    return_video_kwargs: bool = False,
    sanity_check=False
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    # Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(
                vision_info, return_video_sample_fps=True, sanity_check=sanity_check)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs


def resize(mask, size):
    return T.resize(mask.unsqueeze(0).unsqueeze(0), size)[0, 0]


def process_masks(sample, frame_size, inds):
    if sample['mask_type'] == 'image':
        # case 1: list of masks or paths to masks
        masks = []
        for obj_oids in sample['oids']:
            obj_masks = []
            for i in inds:
                label = sample['masks'][i]
                if isinstance(label, str):
                    label = np.array(Image.open(label))
                elif label is None:
                    label = np.full(frame_size, -1)
                obj_masks.append(torch.from_numpy(sum([label == oid for oid in obj_oids])).float())
            masks.append(obj_masks)
    elif sample['mask_type'] == 'image_sep':
        # case 2: list of masks or paths to masks (one object per image)
        masks = []
        for raw_obj_masks in sample['masks']:
            obj_masks = []
            for i in inds:
                label = raw_obj_masks[i]
                if isinstance(label, str):
                    label = np.array(Image.open(label))
                elif label is None:
                    label = np.full(frame_size, -1)
                obj_masks.append(torch.from_numpy(label == 255).float())
            masks.append(obj_masks)
    elif sample['mask_type'] == 'rle':
        # case 3: list of lists of multi-region RLE masks
        raw_masks = nncore.load(sample['masks']) if isinstance(sample['masks'], str) else sample['masks']
        masks = []
        for raw_obj_masks in raw_masks:
            obj_masks = []
            for i in inds:
                mask = torch.zeros(frame_size)
                for rle in raw_obj_masks[i]:
                    if isinstance(rle, list):
                        rles = frPyObjects(rle, sample.get('height', frame_size[0]), sample.get('width', frame_size[1]))
                        mask += resize(torch.from_numpy(decode(merge(rles))).float(), frame_size)
                    elif isinstance(rle, dict):
                        if isinstance(rle['counts'], list):
                            rle = frPyObjects(rle, *rle['size'])
                        mask += resize(torch.from_numpy(decode(rle)).float(), frame_size)
                    elif rle is None:
                        mask += 0
                    else:
                        raise TypeError(f'unknown rle mask: {rle}')
                obj_masks.append((mask > 0).float())
            masks.append(obj_masks)
    elif sample['mask_type'] == 'polygon':
        # case 4: list of lists of polygons
        masks = []
        for raw_obj_masks in sample['masks']:
            obj_masks = []
            for i in inds:
                # step 1: sort shapes
                areas = []
                for shape in raw_obj_masks[i]:
                    tmp = np.zeros(frame_size, dtype=np.uint8)
                    cv2.polylines(tmp, np.array([shape['points']], dtype=np.int32), True, 1, 1)
                    cv2.fillPoly(tmp, np.array([shape['points']], dtype=np.int32), 1)
                    areas.append(tmp.sum())
                shapes = [raw_obj_masks[i][j] for j in list(np.argsort(areas)[::-1].astype(np.int32))]
                # step 2: draw masks
                mask = np.zeros(frame_size, dtype=np.uint8)
                for shape in shapes:
                    assert shape['label'] in ('target', 'ignore'), shape
                    label = 1 if shape['label'] == 'target' else -1  # replacing 255 with -1 here
                    cv2.polylines(mask, np.array([shape['points']], dtype=np.int32), True, label, 1)
                    cv2.fillPoly(mask, np.array([shape['points']], dtype=np.int32), label)
                obj_masks.append(torch.from_numpy(mask).float())
            masks.append(obj_masks)
    elif sample['mask_type'] == 'vicas':
        # case 5: special case for vicas dataset
        masks = []
        for obj_rle_path in sample['masks']:
            obj_rles, obj_masks = nncore.load(obj_rle_path), []
            for i in inds:
                mask = torch.zeros(frame_size)
                for rle in obj_rles[i]:
                    mask += 0 if rle is None else resize(torch.from_numpy(decode(rle)).float(), frame_size)
                obj_masks.append((mask > 0).float())
            masks.append(obj_masks)
    elif sample['mask_type'] == 'sav':
        # case 6: special case for sav dataset
        annos = nncore.load(sample['masks'])['masklet']
        masks = [[]]
        for i in inds:
            mask = resize(torch.from_numpy(decode(annos[i][int(sample['qid'])])).float(), frame_size)
            masks[0].append(mask)
    else:
        raise TypeError(f"unknown mask type: {sample['mask_type']}")

    return masks


def build_obj_to_frame_idx(label_mask, batch_mode):
    step_t_obj_to_frame_idx = [[]] if batch_mode else [[] for _ in range(label_mask.size(0))]

    # t: frame_idx v: video_idx
    for t in range(len(step_t_obj_to_frame_idx)):
        if batch_mode:
            for v in range(label_mask.size(0)):
                for _ in range(label_mask.size(1)):
                    step_t_obj_to_frame_idx[t].append(torch.IntTensor([t, v]))
        else:
            for _ in range(label_mask.size(1)):
                step_t_obj_to_frame_idx[t].append(torch.IntTensor([t, 0]))

    label_obj_to_frame_idx = torch.stack([torch.stack(o) for o in step_t_obj_to_frame_idx])
    return label_obj_to_frame_idx


def preprocess_chatml(input_ids, text, tokenizer):
    conv = get_conv('chatml')

    rounds = [m + conv.seps[0] for m in text.split(conv.seps[0])]
    assert (len(rounds) % 2 == 0) == (conv.system is not None)
    assert rounds[-1] == conv.seps[0]
    rounds = rounds[:-1]

    if conv.system is None:
        rounds = [''.join(rounds[i:i + 2]) for i in range(0, len(rounds), 2)]
    else:
        rounds = [''.join(rounds[:3])] + [''.join(rounds[i:i + 2]) for i in range(3, len(rounds), 2)]

    labels = input_ids.clone()

    sep = conv.seps[0] + conv.roles[1]
    cur_len = 0

    for i, rou in enumerate(rounds):
        if len(rou) == 0:
            break

        ins = sep.join(rou.split(sep)[:-1]) + sep

        rou_len = tokenizer(rou, return_length=True).length[0]
        ins_len = tokenizer(ins, return_length=True).length[0]

        labels[cur_len:cur_len + ins_len] = IGNORE_INDEX
        cur_len += rou_len

    if labels.size(0) != cur_len:
        warnings.warn(f'Tokenization mismatch: {labels.size(0)} and {cur_len}')

    return labels


def preprocess(input_ids, text, tokenizer, conv_type):
    if conv_type == 'chatml':
        return preprocess_chatml(input_ids, text, tokenizer)
    else:
        raise ValueError(f'unknown conversation type: {conv_type}')
