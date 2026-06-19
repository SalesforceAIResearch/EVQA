# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
            self,
            num_pos_feats,
            temperature: int = 10000,
            normalize: bool = True,
            scale: Optional[float] = None,
            # Following settings only relevant
            # for warmping up cache for compilation
            warmup_cache: bool = True,
            image_size: int = 1024,
            strides: Tuple[int] = (4, 8, 16, 32),
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}
        if warmup_cache:
            # Warmup cache for cuda and npu, to help with compilation
            try:
                import torch_npu
                has_npu = torch_npu.npu.is_available()
            except ImportError:
                has_npu = False
            if torch.cuda.is_available() or has_npu:
                device = torch.device("cuda" if torch.cuda.is_available() else "npu")
                for stride in strides:
                    cache_key = (image_size // stride, image_size // stride)
                    self._pe(1, device, None, *cache_key)

    def _encode_xy(self, x, y):
        # NOTE: disable autocasting here
        raise NotImplementedError
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        # NOTE: disable autocasting here
        raise NotImplementedError
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        # NOTE: disable autocasting here
        raise NotImplementedError
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def _pe(self, B, device, dtype, *cache_key):
        H, W = cache_key
        if cache_key in self.cache:
            return self.cache[cache_key].to(device)[None].repeat(B, 1, 1, 1)

        # Force fp32 (https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=device.type, enabled=False):
            y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).view(1, -1, 1).repeat(B, 1, W)
            x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).view(1, 1, -1).repeat(B, H, 1)

            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
            dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        if dtype is not None:
            pos = pos.to(dtype)

        self.cache[cache_key] = pos[0]
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        cache_key = (x.shape[-2], x.shape[-1])
        return self._pe(B, x.device, x.dtype, *cache_key)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    @torch.no_grad()
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    @torch.no_grad()
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device

        # Force fp32 (https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=device.type, enabled=False):
            grid = torch.ones((h, w), device=device, dtype=torch.float32)
            y_embed = grid.cumsum(dim=0) - 0.5
            x_embed = grid.cumsum(dim=1) - 0.5
            y_embed = y_embed / h
            x_embed = x_embed / w
            pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))

        pe = pe.to(self.positional_encoding_gaussian_matrix.dtype)
        return pe.permute(2, 0, 1)  # C x H x W

    @torch.no_grad()
    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        assert coords_input.dtype == torch.float, 'coords_input must be in float32'

        # Force fp32 (https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=coords_input.device.type, enabled=False):
            coords = coords_input.clone()
            coords[:, :, 0] = coords[:, :, 0] / image_size[1]
            coords[:, :, 1] = coords[:, :, 1] / image_size[0]
            pe = self._pe_encoding(coords.to(torch.float))  # B x N x C

        pe = pe.to(self.positional_encoding_gaussian_matrix.dtype)
        return pe


class PositionEmbedding1DRandom(nn.Module):
    """
    Positional encoding using random frequencies for 1D inputs.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((1, num_pos_feats)),
        )

    @torch.no_grad()
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix.to(coords.dtype)
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    @torch.no_grad()
    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a sequence of the specified length."""
        device = self.positional_encoding_gaussian_matrix.device

        # Force fp32 (https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=device.type, enabled=False):
            positions = torch.arange(size, device=device, dtype=torch.float32)
            positions = positions / (size - 1)
            positions = positions.unsqueeze(-1)
            pe = self._pe_encoding(positions)

        pe = pe.to(self.positional_encoding_gaussian_matrix.dtype)
        return pe.permute(1, 0)  # C x L

    @torch.no_grad()
    def forward_with_coords(self, coords_input: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Positionally encode raw coordinates by normalizing to [0,1]."""
        assert coords_input.dtype == torch.float, 'coords_input must be in float32'

        # Force fp32 (https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=coords_input.device.type, enabled=False):
            coords = coords_input.clone()
            coords = coords / (seq_length - 1)
            if coords.dim() == 2:
                coords = coords.unsqueeze(-1)
            pe = self._pe_encoding(coords.to(torch.float))  # B x N x C

        pe = pe.to(self.positional_encoding_gaussian_matrix.dtype)
        return pe


# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


@torch.no_grad()
def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


@torch.no_grad()
def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    # Force fp32 on CPU (see https://github.com/huggingface/transformers/pull/29285)
    with torch.autocast(device_type='cpu', enabled=False):
        freqs_x = 1.0 / (theta**(torch.arange(0, dim, 4)[:(dim // 4)].float() / dim))
        freqs_y = 1.0 / (theta**(torch.arange(0, dim, 4)[:(dim // 4)].float() / dim))

        t_x, t_y = init_t_xy(end_x, end_y)
        freqs_x = torch.outer(t_x, freqs_x)
        freqs_y = torch.outer(t_y, freqs_y)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


@torch.no_grad()
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


@torch.no_grad()
def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    # Force fp32 (https://github.com/huggingface/transformers/pull/29285)
    with torch.autocast(device_type=freqs_cis.device.type, enabled=False):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = (torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None)
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        if xk_ is None:
            # no keys to rotate, due to dropout
            return xq_out.type_as(xq).to(xq.device), xk
        # repeat freqs along seq_len dim to match k seq_len
        if repeat_freqs_k:
            r = xk_.shape[-2] // xq_.shape[-2]
            if freqs_cis.is_cuda:
                freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
            else:
                # torch.repeat on complex numbers may not be supported on non-CUDA devices
                # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
                freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
