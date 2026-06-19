# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from unipixel.utils.env import get_auto_device


def build_model(model_path,
                config=None,
                image_size=None,
                is_trainable=False,
                merge_adapter=False,
                attn_implementation='flash_attention_2',
                device='auto',
                dtype='bfloat16'):
    # set do_resize to false to avoid duplicated resizing
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, do_resize=False)

    config = config or AutoConfig.from_pretrained(model_path)
    config.sam2_inference_mode = not is_trainable

    # override sam2 image size
    if image_size is not None:
        config.sam2_image_size = image_size

    adapter_path = nncore.join(model_path, 'adapter_model.safetensors')
    partial_path = nncore.join(model_path, 'pytorch_model.safetensors')

    if nncore.is_file(adapter_path) or nncore.is_file(partial_path):
        print(f'Loading base model from {config.base_model_path}...')
        model = AutoModel.from_pretrained(
            config.base_model_path,
            config=config,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            device_map='auto' if device == 'all' else None)

        meta_state_dict = {
            n: torch.empty_like(p, device='cpu')
            for n, p in model.named_parameters() if p.device == torch.device('meta')
        }
        model.load_state_dict(meta_state_dict, strict=False, assign=True)

        # sam2 weights might be replaced later
        if model.config.sam2_checkpoint:
            model.load_sam2_weights()

        embed_tokens = model.get_input_embeddings()
        size = (embed_tokens.num_embeddings, embed_tokens.embedding_dim)
        if embed_tokens.weight.size() != size:
            print(f'Resizing embed_tokens from {embed_tokens.weight.size()} to {size}...')
            model.model.language_model.embed_tokens.weight = nn.Parameter(embed_tokens.weight.new_empty(size))

        size = (model.lm_head.out_features, model.lm_head.in_features)
        if model.lm_head.weight.size() != size:
            print(f'Resizing lm_head from {model.lm_head.weight.size()} to {size}...')
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.new_empty(size))

        if nncore.is_file(adapter_path):
            print(f'Loading adapter from {model_path}...')
            # transformers integration does not support merge_and_unload, use peft instead
            model = PeftModel.from_pretrained(
                model,
                model_path,
                is_trainable=is_trainable,
                low_cpu_mem_usage=True,
                # load adapters to the same device as embed_tokens
                torch_device=str(embed_tokens.weight.device))

        if nncore.is_file(partial_path):
            print(f'Loading state dict from {partial_path}...')
            _, unexpected = load_model(model, partial_path, strict=False, device=str(model.device))
            assert len(unexpected) == 0, f'unexpected parameters: {unexpected}'

        if (not is_trainable or merge_adapter) and nncore.is_file(adapter_path):
            print('Merging adapter and unloading...')
            model = model.merge_and_unload()
            model._hf_peft_config_loaded = False
    else:
        print(f'Loading full model from {model_path}...')

        if config.model_type == 'qwen2_5_vl':
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            model_cls = AutoModel

        model = model_cls.from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
            device_map='auto' if device == 'all' else None)

        model.requires_grad_(False)

    if not is_trainable and device != 'all':
        device = get_auto_device() if device == 'auto' else device
        model = model.to(device).eval()

    return model, processor
