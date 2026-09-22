# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026 
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
SpeechNet Architecture
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T_audio
from typing import Any, Dict, List, Optional


class SpeechNet(nn.Module):
    """
    Base Parametric SpeechNet.

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
        global_pool: str = "avg",  # "avg" or "max"
        train_cfg: Optional[Dict[str, Any]] = None,
        use_bilstm: bool = False,
        domain: str = "time",
        mfcc_cfg: Optional[Dict[str, Any]] = None,
        stft_cfg: Optional[Dict[str, Any]] = None,
        spec_augment: bool = False,
        freq_mask_param: int = 8,
        time_mask_param: int = 20,
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

        # Convolutional blocks configuration
        if blocks_config is None:
            blocks_config = [
                dict(out_channels=8, kernel=(1, 4), pool=(1, 2)),
                dict(out_channels=16, kernel=(1, 16), pool=(1, 2)),
                dict(out_channels=16, kernel=(1, 8), pool=(1, 2)),
                dict(out_channels=32, kernel=("full", 1), pool=(1, 1)),
                dict(out_channels=32, kernel=(1, 1), pool=(1, 1)),
            ]

        self.blocks = nn.ModuleList()

        for i, cfg in enumerate(blocks_config):
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

            layers = []

            # Padding calculation
            pad_c = 0 if k_c == C else k_c // 2
            conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=(int(k_c), int(k_t)),
                stride=stride,
                padding=(pad_c, int(k_t) // 2),
                padding_mode="zeros",
            )

            layers += [conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]

            layers += [nn.MaxPool2d(kernel_size=(pool_c, pool_t), stride=(pool_c, pool_t))]

            self.blocks.append(nn.Sequential(*layers))
            in_ch = out_ch

        # Classification and pooling layers
        if global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
        self.use_bilstm = bool(use_bilstm)

        # RNN initialization
        if self.use_bilstm:
            rnn_hidden_dim = kwargs.get("rnn_hidden_dim", 128)
            self.rnn = nn.LSTM(
                input_size=in_ch,
                hidden_size=rnn_hidden_dim,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=p_dropout if p_dropout > 0 else 0.0
            )
            fc_in_features = rnn_hidden_dim * 2
        else:
            fc_in_features = in_ch          
        self.fc = nn.Linear(fc_in_features, output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (B, C, T)
        
        # Frequency domain transformation
        if self.domain in ["mfcc", "stft"]:
            x = self.transform(x)
            if self.domain == "stft":
                # Converts the power spectrogram to decibel (dB) scale
                x = 10.0 * torch.log10(x + 1e-10)
            if self.spec_augment and self.training:
                if self.freq_mask is not None:
                    x = self.freq_mask(x)
                if self.time_mask is not None:
                    x = self.time_mask(x)
        else:
            x = x[:, None]  # (B, 1, C, T)

        # Feature extraction through convolutional blocks
        for block in self.blocks:
            x = block(x)

        # Output
        if self.use_bilstm or self.loss_name == "ctc":
            # x is currently (B, channels, Freq_remaining, T_remaining)
            x_seq = x.mean(dim=2)           # (B, channels, T_remaining)
            x_seq = x_seq.permute(0, 2, 1)  # (B, T_remaining, channels)

            if self.use_bilstm:
                x_seq, _ = self.rnn(x_seq)  # (B, T_remaining, rnn_hidden_dim * 2)
                
            x_seq = self.dropout(x_seq)

            if self.loss_name == "ctc":
                # Return predictions for each time step for CTC loss
                out = self.fc(x_seq)        # (B, T_remaining, output_classes)
                return out
            else:
                # Cross Entropy with BiLSTM: pool over the remaining time dimension
                x_pooled = x_seq.mean(dim=1)    # (B, fc_in_features)
                out = self.fc(x_pooled)         # (B, output_classes)
                return out

        else:
            # CNN-only classification
            x = self.global_pool(x)  # (B, channels_last, 1, 1)
            x = torch.flatten(x, 1)  # (B, channels_last)
            x = self.dropout(x)
            x = self.fc(x)           # (B, output_classes)
            return x
