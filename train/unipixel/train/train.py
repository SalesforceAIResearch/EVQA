# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

from dataclasses import dataclass, field
from typing import List, Optional

import nncore
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import HfArgumentParser, TrainingArguments

from unipixel.constants import MEM_TOKEN, REF_TOKEN, SEG_TOKEN
from unipixel.dataset import HybridDataCollator, HybridDataset
from unipixel.model import MODELS
from unipixel.model.builder import build_model
from unipixel.train.custom_trainer import CustomTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    conv_type: Optional[str] = field(default=None)
    partial_checkpoint: Optional[str] = field(default=None)
    sam2_config: Optional[str] = field(default=None)
    sam2_checkpoint: Optional[str] = field(default=None)
    sam2_image_size: Optional[int] = field(default=1024)
    sam2_hidden_tokens: Optional[int] = field(default=2)
    sam2_batch_mode: Optional[bool] = field(default=False)
    sam2_apply_postprocessing: Optional[bool] = field(default=False)
    sam2_enable_decoder: Optional[bool] = field(default=True)
    sam2_inference_mode: Optional[bool] = field(default=False)
    sample_objects: Optional[int] = field(default=-1)


@dataclass
class DataArguments:
    datasets: Optional[str] = field(default=None)
    sample_frames: Optional[str] = field(default='-1')
    sample_type: Optional[str] = field(default='uniform')
    sample_for_llm_only: Optional[bool] = field(default=False)
    num_threads: Optional[int] = field(default=0)
    min_video_frames: Optional[int] = field(default=-1)
    max_video_frames: Optional[int] = field(default=-1)
    min_video_len: Optional[int] = field(default=-1)
    max_video_len: Optional[int] = field(default=-1)
    min_num_words: Optional[int] = field(default=-1)
    max_num_words: Optional[int] = field(default=-1)
    max_num_tokens: Optional[int] = field(default=-1)
    max_num_objects: Optional[int] = field(default=-1)
    max_conv_turns: Optional[int] = field(default=1)
    max_retries: Optional[int] = field(default=10)
    debug_print_samples: Optional[int] = field(default=0)  # Print first N samples for debugging


@dataclass
class TrainingArguments(TrainingArguments):
    optim: Optional[str] = field(default='adamw_torch')
    group_by_data_type: Optional[bool] = field(default=True)
    merge_adapter: Optional[bool] = field(default=False)
    lora_enable: Optional[bool] = field(default=False)
    lora_type: Optional[str] = field(default='qkvo_all')
    lora_r: Optional[int] = field(default=128)
    lora_alpha: Optional[int] = field(default=256)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_bias: Optional[str] = field(default='none')
    lora_lr: Optional[float] = field(default=None)
    sam2_lr: Optional[float] = field(default=None)
    sam2_enc_lr: Optional[float] = field(default=None)
    sam2_dec_lr: Optional[float] = field(default=None)
    tuning_modules: Optional[str] = field(default=None)
    chunk_steps: Optional[int] = field(default=-1)
    save_full_model: Optional[bool] = field(default=False)
    remove_unused_columns: Optional[bool] = field(default=False)
    label_names: Optional[List[str]] = field(default_factory=lambda: ['label'])


def get_target_modules(model, lora_type, base_model):
    layer_type, modules = lora_type.split('_')
    assert layer_type in ('qkvo', 'linear') and modules in ('llm', 'visual', 'all')

    if base_model == 'qwen2_5_vl':
        # all qkvo layers in the visual encoder and the llm
        qkvo_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn.qkv', 'attn.proj']

        target_modules = []
        for n, m in model.named_modules():
            if 'sam2' in n or not isinstance(m, nn.Linear):
                continue
            if modules == 'llm' and 'visual' in n:
                continue
            if modules == 'visual' and 'visual' not in n:
                continue
            if layer_type == 'qkvo' and not any(n.endswith(k) for k in qkvo_keys):
                continue
            if n in target_modules:
                continue
            target_modules.append(n)
    else:
        raise ValueError(f'unknown base model: {base_model}')

    return target_modules


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config_cls, model_cls, processor_cls = MODELS[model_args.base_model]

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    config = config_cls.from_pretrained(model_args.model_name_or_path, torch_dtype=dtype)
    config.update(model_args.__dict__)

    if config.model_type in ('pixel_qwen2_vl', 'pixel_qwen2_5_vl'):
        model, processor = build_model(
            model_args.model_name_or_path,
            config=config,
            is_trainable=True,
            merge_adapter=training_args.merge_adapter,
            dtype=dtype)
    else:
        # set do_resize to false to avoid duplicated resizing
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
        processor = processor_cls.from_pretrained(model_args.model_name_or_path, use_fast=True, do_resize=False)

        # eager/sdpa attention has known & unknown bugs
        # [4.46.2] broken causality fp16: https://github.com/huggingface/transformers/issues/35151
        # [4.48.1] broken sliding window: https://github.com/huggingface/transformers/issues/35924
        # [4.52.4] masked window attention: https://github.com/huggingface/transformers/pull/37363
        model = model_cls.from_pretrained(
            model_args.model_name_or_path, config=config, attn_implementation='flash_attention_2')

        # save base model path for inference
        model.config.base_model_path = model_args.model_name_or_path

        # load sam2 checkpoint only when initialized from scratch
        if model.config.sam2_checkpoint:
            model.load_sam2_weights()

        # initialize extra parameters to ensure no overflow
        model.init_parameters()

        # load partial checkpoints
        if model_args.partial_checkpoint is not None:
            for partial_path in model_args.partial_checkpoint.split(','):
                if not partial_path.endswith('.safetensors'):
                    partial_path = nncore.join(partial_path, 'pytorch_model.safetensors')
                print(f'Loading state dict from {partial_path}...')
                state_dict = load_file(partial_path)
                print(f'State dict keys: {list(state_dict.keys())}')
                _, unexpected = model.load_state_dict(state_dict, strict=False)
                assert len(unexpected) == 0, f'unexpected parameters: {unexpected}'

        model.requires_grad_(False)

    if training_args.lora_enable and not isinstance(model, PeftModel):
        target_modules = get_target_modules(model, training_args.lora_type, model.config.base_model)
        print(f'LoRA target modules: {target_modules}')
        lora_config = LoraConfig(
            task_type='CAUSAL_LM',
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            target_modules=target_modules)
        # transformers integration does not support merge_and_unload, use peft instead
        model = get_peft_model(model, lora_config)

    new_tokens = 0
    if model.config.sam2_config is not None:
        special_tokens = [REF_TOKEN, SEG_TOKEN, MEM_TOKEN]
        new_tokens = processor.tokenizer.add_special_tokens(dict(additional_special_tokens=special_tokens))
        print(f'Added {new_tokens} new token(s)')

        model.config.ref_token_id = processor.tokenizer.convert_tokens_to_ids(REF_TOKEN)
        model.config.seg_token_id = processor.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        model.config.mem_token_id = processor.tokenizer.convert_tokens_to_ids(MEM_TOKEN)

        if new_tokens > 0 and len(processor.tokenizer) > model.config.vocab_size:
            print(f'Expanding vocab size: {model.config.vocab_size} -> {len(processor.tokenizer)}')
            model.resize_token_embeddings(len(processor.tokenizer))
            i_emb = model.get_input_embeddings().weight.data
            o_emb = model.get_output_embeddings().weight.data
            i_emb[-new_tokens:] = i_emb[:-new_tokens].mean(0, keepdim=True)
            o_emb[-new_tokens:] = o_emb[:-new_tokens].mean(0, keepdim=True)

    tuning_modules = [] if training_args.tuning_modules is None else training_args.tuning_modules.split(',')

    for n, p in model.named_parameters():
        if 'embedding' in tuning_modules and any(k in n for k in ('embed_tokens', 'lm_head')):
            p.requires_grad = True

        if 'projector' in tuning_modules and 'visual.merger' in n:
            p.requires_grad = True

        if 'ref' in tuning_modules and any(k in n for k in ('ref_proj', 'tem_pe', 'tem_emb', 'tem_proj')):
            p.requires_grad = True

        if 'ref_enc' in tuning_modules and any(k in n for k in ('ref_encoder', )):
            p.requires_grad = True

        if 'msk' in tuning_modules and any(k in n for k in ('msk_proj', )):
            p.requires_grad = True

        if 'seg' in tuning_modules and any(k in n for k in ('seg_head', )):
            p.requires_grad = True

        if 'sam2' in tuning_modules and 'sam2' in n:
            p.requires_grad = True

    if training_args.local_rank in (0, -1):
        for n, p in model.named_parameters():
            assert p.isfinite().all(), f"Parameter '{n}' ({p.dtype}) has infinite value(s)"
            print(p.requires_grad, p.dtype, p.shape, n)

        total_params = sum(p.numel() for p in model.parameters())
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = round(learnable_params / total_params * 100, 2) if total_params > 0 else 0
        print(f'Total params: {total_params} Learnable params: {learnable_params} ({ratio}%)')

        i_size = model.get_input_embeddings().num_embeddings
        o_size = model.get_output_embeddings().out_features
        assert i_size == o_size, (i_size, o_size)
        print(f'Tokenizer size: {len(processor.tokenizer)} Vocab size: {model.config.vocab_size} Embed size: {i_size}')

    training_args.run_name = f'{nncore.cwd(base=True)}/{nncore.base_name(training_args.output_dir)}'

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=HybridDataCollator(processor.tokenizer),
        train_dataset=HybridDataset(processor, model_args, data_args, training_args),
        processing_class=processor)

    has_ckpt = bool(nncore.find(training_args.output_dir, 'checkpoint-*'))
    trainer.train(resume_from_checkpoint=has_ckpt)

    if getattr(trainer, 'finished', True):
        trainer.save_state()
        trainer.gather_and_save_model()


if __name__ == '__main__':
    train()
