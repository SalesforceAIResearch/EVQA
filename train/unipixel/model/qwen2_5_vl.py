# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random

import torch
import torch.nn as nn
from hydra import compose
from hydra.utils import instantiate
from nncore.nn import constant_init_, xavier_init_
from transformers import (AutoConfig, AutoModel, AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2_5_VLModel, Qwen2_5_VLProcessor, Qwen2_5_VLTextModel)
from transformers.models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2RMSNorm

from sam2.loss_fns import MultiStepMultiMasksAndIous
from sam2.modeling.position_encoding import PositionEmbedding1DRandom
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.sam2_train import BatchedVideoDatapoint


def cache_state_hook(module, inputs, outputs=None):
    module.state = inputs[0] if isinstance(inputs, tuple) else inputs


class PatchedQwen2_5_VLProcessor(Qwen2_5_VLProcessor):

    def _check_special_mm_tokens(self, text, *args, **kwargs):
        self.cache_text = text
        return super()._check_special_mm_tokens(text, *args, **kwargs)


class PixelQwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = 'pixel_qwen2_5_vl'


class PixelQwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VisionTransformerPretrainedModel):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.merger.mlp.register_forward_pre_hook(cache_state_hook)


class PixelQwen2_5_VLModel(Qwen2_5_VLModel):
    config_class = PixelQwen2_5_VLConfig

    def __init__(self, config):
        super(Qwen2_5_VLModel, self).__init__(config)
        self.visual = PixelQwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()
        self.language_model.norm.register_forward_pre_hook(cache_state_hook)


class PixelQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    config_class = PixelQwen2_5_VLConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = PixelQwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if self.config.sam2_config is not None:
            overrides = [f'++model.image_size={self.config.sam2_image_size}']
            if self.config.sam2_inference_mode:
                overrides.append('++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor')

            cfg = compose(config_name=self.config.sam2_config, overrides=overrides)
            self.sam2 = instantiate(cfg.model)

            sam_dim, llm_dim = self.sam2.hidden_dim, self.config.hidden_size

            self.seg_head = nn.Sequential(
                Qwen2RMSNorm(llm_dim), nn.Linear(llm_dim, llm_dim), nn.GELU(),
                nn.Linear(llm_dim, sam_dim * self.config.sam2_hidden_tokens))

            self.ref_encoder = PromptEncoder(
                embed_dim=sam_dim,
                image_embedding_size=(self.sam2.sam_image_embedding_size, self.sam2.sam_image_embedding_size),
                input_image_size=(self.config.sam2_image_size, self.config.sam2_image_size),
                mask_in_chans=16)

            self.ref_proj_single = nn.Linear(sam_dim * 2, sam_dim * 3)
            self.ref_proj_double = nn.Linear(sam_dim * 3, sam_dim * 3)
            self.ref_proj = nn.Sequential(nn.GELU(), nn.Linear(sam_dim * 6, llm_dim))

            self.tem_pe = PositionEmbedding1DRandom(sam_dim // 2)
            self.tem_emb = nn.Embedding(1, sam_dim)
            self.tem_proj = nn.Linear(sam_dim, sam_dim * 3)

            self.msk_proj = nn.Sequential(
                nn.Linear(self.visual.merger.hidden_size, self.visual.merger.hidden_size), nn.GELU(),
                nn.Linear(self.visual.merger.hidden_size, llm_dim))

            self.loss_seg = MultiStepMultiMasksAndIous(
                dict(loss_mask=100, loss_dice=5, loss_iou=5, loss_class=5),
                supervise_all_iou=True,
                iou_use_l1_loss=True,
                pred_obj_scores=True,
                focal_alpha=0.25,
                focal_gamma=2.0,
                focal_alpha_obj_score=-1.0,
                focal_gamma_obj_score=0.0)

        self.post_init()

    @torch.no_grad()
    def init_parameters(self):
        # initialize ref_encoder with weights from sam2.sam_prompt_encoder
        for p0, p1 in zip(self.ref_encoder.parameters(), self.sam2.sam_prompt_encoder.parameters()):
            p0.copy_(p1)

        # initialize msk_proj with weights from visual.merger.mlp
        for p0, p1 in zip(self.msk_proj.parameters(), self.visual.merger.mlp.parameters()):
            p0.copy_(p1)

        # reset extra parameters
        for s in ('seg_head', 'ref_proj_single', 'ref_proj_double', 'ref_proj', 'tem_proj'):
            b = getattr(self, s, None)
            if b is None:
                continue
            for n, m in b.named_modules():
                if isinstance(m, nn.Linear):
                    print(f'Reset parameters of {b.__class__.__name__} {n} ({m.__class__.__name__})')
                    xavier_init_(m, distribution='uniform')
                elif isinstance(m, nn.LayerNorm):
                    print(f'Reset parameters of {b.__class__.__name__} {n} ({m.__class__.__name__})')
                    constant_init_(m)

    def load_sam2_weights(self):
        state_dict = torch.load(self.config.sam2_checkpoint, map_location=self.sam2.device, weights_only=True)['model']
        state_dict['memory_encoder.fuser.layers.0.weight'] = state_dict.pop('memory_encoder.fuser.layers.0.gamma')
        state_dict['memory_encoder.fuser.layers.1.weight'] = state_dict.pop('memory_encoder.fuser.layers.1.gamma')
        self.sam2.load_state_dict(state_dict)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
                rope_deltas=None,
                cache_position=None,
                second_per_grid_ts=None,
                frames=None,
                frame_size=None,
                point_coords=None,
                point_labels=None,
                point_frames=None,
                refer_mask=None,
                label_obj_to_frame_idx=None,
                label_mask=None):
        if caching := not self.training and (past_key_values is None or len(past_key_values) == 0):
            self.seg = []

        # move input_ids to the correct device (in case of auto device map)
        input_ids = input_ids.to(self.model.language_model.embed_tokens.weight.device)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            device, dtype = inputs_embeds.device, inputs_embeds.dtype

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_embeds = torch.cat(image_embeds)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_embeds.shape[0]
                assert n_image_tokens == n_image_features

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(device)

                image_embeds = image_embeds.to(device, dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds)
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_embeds.shape[0]
                assert n_video_tokens == n_video_features

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(device)

                video_embeds = video_embeds.to(device, dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if any(k is not None for k in (point_coords, point_labels, point_frames)):
                assert all(k is not None for k in (point_coords, point_labels, point_frames))

                ref = []
                for batch_idx in range(video_grid_thw.size(0)):
                    for obj_point_coords, obj_point_labels in zip(point_coords[batch_idx], point_labels[batch_idx]):
                        obj_ref, _ = self.ref_encoder((obj_point_coords, obj_point_labels), None, None, None)
                        assert obj_ref.size(1) in (2, 3), obj_ref.size()
                        if obj_ref.size(1) == 2:
                            obj_ref = self.ref_proj_single(obj_ref.flatten(1))
                        else:
                            obj_ref = self.ref_proj_double(obj_ref.flatten(1))
                        ref.append(obj_ref)
                ref = torch.cat(ref)

                tem = []
                for batch_idx in range(video_grid_thw.size(0)):
                    # temporal merge size set to 2
                    size = video_grid_thw[batch_idx][0].item() * 2
                    for obj_point_frames in point_frames[batch_idx]:
                        obj_tem = obj_point_frames.unsqueeze(0).float()
                        obj_tem = self.tem_pe.forward_with_coords(obj_tem, size)
                        assert obj_tem.size(0) == 1, obj_tem.size()
                        tem.append(obj_tem[0])
                tem = torch.cat(tem)
                tem = tem + self.tem_emb(torch.LongTensor([0]).to(device))
                tem = self.tem_proj(tem)

                ref_emb = self.ref_proj(torch.cat((ref, tem), dim=1)).to(device, dtype)
                ref_mask = input_ids == self.config.ref_token_id
                # replace only the <ref> tokens in the instruction
                # ref_mask = ref_mask * (labels == IGNORE_INDEX) if self.training else ref_mask
                ref_mask = ref_mask.unsqueeze(-1).expand_as(inputs_embeds).to(device)
                inputs_embeds = inputs_embeds.masked_scatter(ref_mask, ref_emb)

            if refer_mask is not None:
                mem, base_idx = [], 0
                for batch_idx in range(video_grid_thw.size(0)):
                    size = video_grid_thw[batch_idx].prod().item() // 4
                    step = video_grid_thw[batch_idx][1] * video_grid_thw[batch_idx][2] // 4

                    # emb = self.model.visual.merger.ln_q.state[base_idx:base_idx + size]
                    # map grouped order back to raster scan order
                    # dim = emb.size(1)
                    # emb = emb.permute(1, 0).reshape(dim, -1, 2, 2).permute(0, 2, 1, 3).reshape(dim, -1).permute(1, 0)
                    emb = self.model.visual.merger.mlp.state[base_idx:base_idx + size]
                    batch_refer_mask = refer_mask[batch_idx]

                    for obj_idx in range(batch_refer_mask.size(1)):
                        mask = batch_refer_mask[:, obj_idx].flatten()
                        assert mask.size(0) == emb.size(0) == size
                        obj_emb = []
                        for i in range(0, size, step):
                            frame_mask = mask[i:i + step]
                            if frame_mask.any():
                                obj_emb.append(emb[i:i + step][frame_mask].mean(dim=0))
                        if len(obj_emb) > 0:
                            obj_emb = torch.stack(obj_emb)
                            mem.append(obj_emb)

                    base_idx = base_idx + size

                mem_mask = input_ids == self.config.mem_token_id

                if len(mem) > 0:
                    mem_emb = self.msk_proj(torch.cat(mem))
                    mem_mask = mem_mask.unsqueeze(-1).expand_as(inputs_embeds).to(device)
                    assert mem_emb.size(0) == mem_mask.all(dim=-1).sum(), (mem_emb.size(), mem_mask.all(dim=-1).sum())
                    inputs_embeds = inputs_embeds.masked_scatter(mem_mask, mem_emb)
                else:
                    assert not mem_mask.any()

        # ensure gradient tracking (in case that embed_tokens has been frozen)
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=not self.training,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts)

        if self.config.sam2_config is not None and self.config.sam2_enable_decoder and frames is not None:
            # decoder block -> -2 -> decoder block -> state -> norm -> -1
            seg_tokens_all = self.seg_head(self.model.language_model.norm.state)
            seg_tokens_all = seg_tokens_all.reshape(*seg_tokens_all.shape[:2], self.config.sam2_hidden_tokens, -1)

            if self.training and label_obj_to_frame_idx is not None and label_mask is not None:
                loss_seg_all, avg_factor = 0, 0
                shift_inputs = input_ids[..., 1:].contiguous()

                for batch_idx, (obj_to_frame_idx, mask) in enumerate(zip(label_obj_to_frame_idx, label_mask)):
                    # supervise all <seg> tokens (including those in inputs)
                    inds = torch.where(shift_inputs[batch_idx] == self.config.seg_token_id)[0]
                    assert inds.size(0) == mask.size(1)

                    if self.config.sample_objects > 0 and inds.size(0) > self.config.sample_objects:
                        sample_inds = random.sample(list(range(inds.size(0))), self.config.sample_objects)
                        obj_to_frame_idx = obj_to_frame_idx[:, sample_inds]
                        inds = inds[sample_inds]
                        mask = mask[:, sample_inds]

                    if self.config.sam2_batch_mode:
                        seg_tokens = seg_tokens_all[batch_idx][inds].repeat(mask.size(0), 1, 1)  # (t * o) * 2 * c
                        img_batch = frames[batch_idx].unsqueeze(0)  # 1 * t * c * h * w
                        masks = mask.view(1, -1, mask.size(2), mask.size(3))  # 1 * (t * o) * h * w
                    else:
                        seg_tokens = seg_tokens_all[batch_idx][inds]  # o * 2 * c
                        img_batch = frames[batch_idx].unsqueeze(1)  # t * 1 * c * h * w
                        masks = mask  # t * o * h * w

                    data = BatchedVideoDatapoint(img_batch=img_batch, obj_to_frame_idx=obj_to_frame_idx, masks=masks)
                    pred = self.sam2(data, seg_tokens)

                    loss_seg = self.loss_seg(pred, masks)
                    loss_seg = loss_seg['core_loss'] / masks.size(0)

                    loss_seg_all += loss_seg
                    avg_factor += 1

                assert avg_factor > 0
                outputs.loss = outputs.loss + loss_seg_all / avg_factor
            else:
                assert len(frames) == len(frame_size) == 1
                seg_tokens = []

                if caching:
                    # case 1: input contains <seg>
                    shift_inputs = input_ids[..., 1:].contiguous()
                    inds = torch.where(shift_inputs[0] == self.config.seg_token_id)[0].to(seg_tokens_all.device)
                    seg_tokens += [t for t in seg_tokens_all[0][inds].unsqueeze(1)]

                if outputs.logits[0, -1].argmax() == self.config.seg_token_id:
                    # case 2: output contains <seg>
                    seg_tokens.append(seg_tokens_all[0, -1, None])

                for seg_token in seg_tokens:
                    if self.config.sam2_batch_mode:
                        pred_mask = []
                        for idx in range(frames[0].size(0)):
                            state = self.sam2.init_state(frames[0][idx, None], frame_size[0])
                            self.sam2.add_new_hidden_state(state, 0, 0, seg_token)
                            pred_mask += [o[2] for o in self.sam2.propagate_in_video(state, verbose=False)]
                        pred_mask = torch.cat(pred_mask, dim=1)
                    else:
                        state = self.sam2.init_state(frames[0], frame_size[0])
                        self.sam2.add_new_hidden_state(state, 0, 0, seg_token)
                        pred_mask = torch.cat([o[2] for o in self.sam2.propagate_in_video(state, verbose=False)], dim=1)

                    assert pred_mask.size(1) == frames[0].size(0)
                    self.seg.append((pred_mask > 0).cpu())

        return outputs

    def prepare_inputs_for_generation(self,
                                      *args,
                                      cache_position=None,
                                      frames=None,
                                      frame_size=None,
                                      point_coords=None,
                                      point_labels=None,
                                      point_frames=None,
                                      refer_mask=None,
                                      **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, cache_position=cache_position, **kwargs)

        model_inputs.update({
            'frames': frames,
            'frame_size': frame_size,
            'point_coords': point_coords if cache_position[0] == 0 else None,
            'point_labels': point_labels if cache_position[0] == 0 else None,
            'point_frames': point_frames if cache_position[0] == 0 else None,
            'refer_mask': refer_mask if cache_position[0] == 0 else None
        })

        return model_inputs


# set the patched model to a vision model
MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES[PixelQwen2_5_VLConfig.model_type] = 'PixelQwen2_5_VLForConditionalGeneration'

AutoConfig.register(PixelQwen2_5_VLConfig.model_type, PixelQwen2_5_VLConfig)
AutoModel.register(PixelQwen2_5_VLConfig, PixelQwen2_5_VLForConditionalGeneration)
AutoProcessor.register(PixelQwen2_5_VLConfig, PatchedQwen2_5_VLProcessor)
