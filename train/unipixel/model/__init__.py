# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from .qwen2_5_vl import PatchedQwen2_5_VLProcessor, PixelQwen2_5_VLConfig, PixelQwen2_5_VLForConditionalGeneration

MODELS = {'qwen2_5_vl': (PixelQwen2_5_VLConfig, PixelQwen2_5_VLForConditionalGeneration, PatchedQwen2_5_VLProcessor)}
