# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import math
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint_state_dict(checkpoint_path: str) -> dict:
    """Load a checkpoint state-dict from torch or safetensors formats."""
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors requires safetensors package to be installed."
            ) from exc
        return load_file(checkpoint_path)

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            return payload["state_dict"]
        if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            return payload["model_state_dict"]
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class LearnedRelativePositionalEmbedding(nn.Module):
    """
    Learned relative positional embedding used for encoder self-attention.
    """

    def __init__(self, max_relative_pos: int, num_heads: int, embedding_dim: int):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        num_embeddings = 2 * max_relative_pos - 1
        # Keep the same final singleton dimension to stay close to previous checkpoints.
        self.embeddings = nn.Parameter(torch.zeros(num_heads, num_embeddings, embedding_dim, 1))
        nn.init.normal_(self.embeddings, mean=0.0, std=embedding_dim ** (-0.5))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (length, batch_size * num_heads, embed_dim)

        Returns:
            positional_logits: (batch_size * num_heads, length, length)
        """
        length = query.shape[0]
        used_embeddings = self.get_embeddings_for_query(length)
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        return self.relative_to_absolute_indexing(positional_logits)

    def get_embeddings_for_query(self, length: int) -> torch.Tensor:
        """
        Build the relative-position window required for the given sequence length.
        """
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)

        with torch.no_grad():
            padded_embeddings = nn.functional.pad(
                self.embeddings,
                (0, 0, 0, 0, pad_length, pad_length),
            )
        return padded_embeddings.narrow(-3, start_pos, 2 * length - 1)

    def calculate_positional_logits(
        self,
        query: torch.Tensor,
        relative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
        positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
        positional_logits = positional_logits.contiguous().view(
            positional_logits.shape[0],
            -1,
            positional_logits.shape[-1],
        )

        length = query.size(0)
        if length > self.max_relative_pos:
            pad_length = length - self.max_relative_pos
            positional_logits[:, :, :pad_length] -= 1e8
            positional_logits[:, :, -pad_length:] -= 1e8

        return positional_logits

    def relative_to_absolute_indexing(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert logits indexed by relative position to absolute position indexing.
        """
        length, bsz_heads, _ = x.shape
        x = nn.functional.pad(x, (0, 1))
        x = x.transpose(0, 1)
        x = x.contiguous().view(bsz_heads, length * 2 * length)
        x = nn.functional.pad(x, (0, length - 1))
        x = x.view(bsz_heads, length + 1, 2 * length - 1)
        return x[:, :length, length - 1 :]


class ResBlock(nn.Module):
    def __init__(self, num_ins: int, num_outs: int, stride: int = 1, pre_activation: bool = False, beta: float = 1.0):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.norm1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.norm2 = nn.BatchNorm1d(num_outs)
        self.act = nn.GELU()
        self.beta = beta

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
            if pre_activation:
                self.skip = nn.Sequential(self.res_norm, self.residual_path)
            else:
                self.skip = nn.Sequential(self.residual_path, self.res_norm)
        else:
            self.skip = nn.Identity()

        self.pre_activation = pre_activation

        if pre_activation:
            self.block = nn.Sequential(self.norm1, self.act, self.conv1, self.norm2, self.act, self.conv2)
        else:
            self.block = nn.Sequential(self.conv1, self.norm1, self.act, self.conv2, self.norm2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x) * self.beta
        x = self.skip(x)

        if self.pre_activation:
            return x + res
        return self.act(x + res)


class LRPEAttention(nn.Module):
    """
    Multi-head attention with learned relative positional encoding (LRPE) on logits.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        relative_positional_distance: int = 100,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.hd = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.relative_positional = LearnedRelativePositionalEmbedding(
            max_relative_pos=relative_positional_distance,
            num_heads=num_heads,
            embedding_dim=self.hd,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, d_model = x.shape
        qkv = self.qkv(x).reshape(bsz, length, 3, self.num_heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        logits = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))

        q_pos = q.permute(0, 2, 1, 3)  # (B, L, H, D)
        b, l, h, d = q_pos.size()
        position_logits = self.relative_positional(q_pos.reshape(l, b * h, d))
        position_logits = position_logits.view(b, h, l, l)

        probs = F.softmax(logits + position_logits, dim=-1)
        probs = self.attn_drop(probs)

        out = (probs @ v).transpose(1, 2).reshape(bsz, length, d_model)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.1,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class CustomAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.attn = LRPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            relative_positional_distance=100,
        )
        self.norm1 = norm_layer(dim)

        ffn_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=ffn_dim,
            out_features=dim,
            dropout=proj_drop,
            act_layer=act_layer,
        )
        self.dropout1 = nn.Dropout(proj_drop)
        self.dropout2 = nn.Dropout(proj_drop)
        self.norm2 = norm_layer(dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.norm1(src + self.dropout1(self.attn(src)))
        src = self.norm2(src + self.dropout2(self.mlp(src)))
        return src


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EMGTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_outs: int,
        num_aux_outs: Optional[int] = None,
        in_chans: int = 8,
        embed_dim: int = 192,
        n_layer: int = 8,
        n_head: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        freeze_blocks: bool = False,
        pretrained_path: Optional[str] = None,
        pretrained_strict: bool = False,
    ):
        super().__init__()

        _ = num_features  # kept for compatibility with model factory signatures
        self.in_chans = in_chans

        self.conv_blocks = nn.Sequential(
            ResBlock(in_chans, embed_dim, 2),
            ResBlock(embed_dim, embed_dim, 2),
            ResBlock(embed_dim, embed_dim, 2),
        )
        self.w_raw_in = nn.Linear(embed_dim, embed_dim)

        self.blocks = nn.ModuleList(
            [
                CustomAttentionBlock(
                    dim=embed_dim,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(n_layer)
            ]
        )

        self.w_out = nn.Linear(embed_dim, num_outs)
        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            assert num_aux_outs is not None
            self.w_aux = nn.Linear(embed_dim, num_aux_outs)

        self.initialize_weights()

        if freeze_blocks:
            for param in self.blocks.parameters():
                param.requires_grad = False

        if pretrained_path:
            self.load_pretrained_weights(pretrained_path, strict=pretrained_strict)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_with_best_prefix(self, state_dict: dict, strict: bool) -> Tuple[list, list]:
        model_keys = set(self.state_dict().keys())
        prefixes = ("", "model.", "module.", "backbone.")

        best_prefix = ""
        best_overlap = -1
        for prefix in prefixes:
            if not prefix:
                candidate_keys = set(state_dict.keys())
            else:
                candidate_keys = {k[len(prefix) :] for k in state_dict.keys() if k.startswith(prefix)}

            overlap = len(candidate_keys & model_keys)
            if overlap > best_overlap:
                best_overlap = overlap
                best_prefix = prefix

        if best_prefix:
            state_dict = {
                k[len(best_prefix) :]: v for k, v in state_dict.items() if k.startswith(best_prefix)
            }

        if best_overlap <= 0:
            raise RuntimeError(
                "Could not match checkpoint keys with EMGTransformer parameters. "
                "Please verify that architecture and checkpoint are compatible."
            )

        return self.load_state_dict(state_dict, strict=strict)

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False) -> None:
        state_dict = load_checkpoint_state_dict(checkpoint_path)
        missing, unexpected = self._load_with_best_prefix(state_dict, strict=strict)

        print(
            f"Loaded pretrained weights from {checkpoint_path}. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, T) or (B, T, C), with C == in_chans

        Returns:
            logits over time: (B, T', num_outs)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {tuple(x.shape)}")

        if x.shape[1] == self.in_chans:
            x = x.transpose(1, 2)
        elif x.shape[-1] != self.in_chans:
            raise ValueError(
                f"Input channel mismatch. Expected one dimension to be {self.in_chans}, "
                f"got shape {tuple(x.shape)}"
            )

        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv_blocks(x)
        x = x.transpose(1, 2)  # (B, T', D)
        x = self.w_raw_in(x)

        for block in self.blocks:
            x = block(x)

        if self.has_aux_out:
            return self.w_out(x), self.w_aux(x)

        return self.w_out(x)
