# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
SpeechNetTransformer Architecture
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T_audio


class SinusoidalPositionalEncoding(nn.Module):
    """Parameter-free sinusoidal positional encoding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = int(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        length = x.size(1)
        device = x.device
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)


class SpeechNetTransformer(nn.Module):
    """
    SpeechNet with a Transformer encoder.

    Input:  (B, 1, C, T)                 if domain='time' 
            (B, C, N_MFCC, T_frames)     if domain='mfcc'
            (B, C, Freq_bins, T_frames)  if domain='stft'
    Output: (B, output_classes)          if loss_name='cross_entropy'
            (B, T_frames, output_classes) if loss_name='ctc'
    
    blocks_config: list of blocks, each with:
        out_channels: int
        kernel: (k_c, k_t) where k_c can be int or "full"
        pool:   (p_c, p_t)
        stride: optional, default (1,1)
        padding: optional, default (0,0)
    """

    def __init__(
        self,
        C: int,
        T: int = 1000,
        output_classes: int = 11,
        blocks_config: Optional[List[Dict[str, Any]]] = None,
        p_dropout: float = 0.0,
        global_pool: str = "avg",
        train_cfg: Optional[Dict[str, Any]] = None,
        domain: str = "time",
        mfcc_cfg: Optional[Dict[str, Any]] = None,
        stft_cfg: Optional[Dict[str, Any]] = None,
        spec_augment: bool = False,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        transformer_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.C = C
        self.T = T
        self.output_classes = output_classes
        self.domain = str(domain).lower()

        if train_cfg is not None:
            self.loss_name = str(train_cfg.get("loss_name", "cross_entropy")).lower()
        else:
            self.loss_name = "cross_entropy"
        if self.loss_name not in ["ctc", "cross_entropy"]:
            raise ValueError("loss_name must be either 'ctc' or 'cross_entropy'")

        # Input transform
        if self.domain == "mfcc":
            if mfcc_cfg is None:
                raise ValueError("mfcc_cfg required when using domain='mfcc'")
            self.transform = T_audio.MFCC(**mfcc_cfg)
            in_ch = C
        elif self.domain == "stft":
            if stft_cfg is None:
                raise ValueError("stft_cfg required when using domain='stft'")
            self.transform = T_audio.Spectrogram(**stft_cfg)
            in_ch = C
        else:
            in_ch = 1

        self.spec_augment = bool(spec_augment) and self.domain in ("mfcc", "stft")
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        if self.spec_augment:
            self.freq_mask = (T_audio.FrequencyMasking(freq_mask_param=self.freq_mask_param)
                              if self.freq_mask_param > 0 else None)
            self.time_mask = (T_audio.TimeMasking(time_mask_param=self.time_mask_param)
                              if self.time_mask_param > 0 else None)

        # Convolutional blocks
        if blocks_config is None:
            blocks_config = [
                dict(out_channels=8, kernel=(1, 4), pool=(1, 2)),
                dict(out_channels=16, kernel=(1, 16), pool=(1, 2)),
                dict(out_channels=16, kernel=(1, 8), pool=(1, 2)),
                dict(out_channels=32, kernel=("full", 1), pool=(1, 1)),
                dict(out_channels=32, kernel=(1, 1), pool=(1, 1)),
            ]

        self.blocks = nn.ModuleList()
        for cfg in blocks_config:
            out_ch = int(cfg["out_channels"])

            k_c, k_t = cfg["kernel"]
            if k_c == "full":
                k_c = C
            k_c = int(k_c)
            k_t = int(k_t)

            pool_c, pool_t = cfg.get("pool", (1, 1))
            pool_c, pool_t = int(pool_c), int(pool_t)

            stride = cfg.get("stride", (1, 1))
            if isinstance(stride, int):
                stride = (stride, stride)

            pad_c = 0 if k_c == C else k_c // 2
            conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(k_c, k_t),
                stride=stride,
                padding=(pad_c, k_t // 2),
                padding_mode="zeros",
            )

            layers = [conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            layers += [nn.MaxPool2d(kernel_size=(pool_c, pool_t), stride=(pool_c, pool_t))]

            self.blocks.append(nn.Sequential(*layers))
            in_ch = out_ch

        if global_pool not in ("avg", "max"):
            raise ValueError("global_pool must be 'avg' or 'max'")

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()

        # Transformer
        self.d_model = int(d_model)
        self.input_proj = (
            nn.Linear(in_ch, self.d_model) if in_ch != self.d_model else nn.Identity()
        )
        self.input_norm = nn.LayerNorm(self.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.d_model),
        )
        self._reset_encoder_parameters()

        self.fc = nn.Linear(self.d_model, output_classes)

    def _reset_encoder_parameters(self) -> None:
        """Xavier-init the encoder's linear/projection weights."""
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency-domain transform
        if self.domain in ["mfcc", "stft"]:
            x = self.transform(x)
            if self.domain == "stft":
                x = 10.0 * torch.log10(x + 1e-10)
            if self.spec_augment and self.training:
                if self.freq_mask is not None:
                    x = self.freq_mask(x)
                if self.time_mask is not None:
                    x = self.time_mask(x)
        else:
            x = x[:, None]  # (B, 1, C, T)

        # Convolutional feature extraction
        for block in self.blocks:
            x = block(x)

        # (B, channels, Freq_remaining, T') -> (B, T', channels)
        x_seq = x.mean(dim=2)
        x_seq = x_seq.permute(0, 2, 1)

        # Transformer encoder over the temporal frames
        x_seq = self.input_proj(x_seq)     # (B, T', d_model)
        x_seq = self.input_norm(x_seq)
        x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer(x_seq)    # (B, T', d_model)
        x_seq = self.dropout(x_seq)

        if self.loss_name == "ctc":
            return self.fc(x_seq)          # (B, T', output_classes)

        # Cross-entropy
        x_pooled = x_seq.mean(dim=1)       # (B, d_model)
        return self.fc(x_pooled)           # (B, output_classes)
