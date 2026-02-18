"""
SpeechNet Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]   
sys.path.insert(0, str(PROJECT_ROOT))

class SpeechNet(nn.Module):
    """
    Base Parametric SpeechNet.

    Input:  (B, 1, C, T)
    Output: (B, output_classes)

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
        **kwargs,
        ):
        
        #print("Speech net initialized with dropout,", p_dropout)

        #print("Other kwargs", kwargs)
        super().__init__()
        self.C = C
        self.T = T
        self.output_classes = output_classes

        if blocks_config is None:
            # Example default that is close in spirit to your original (time-only kernels)
            blocks_config = [
                dict(out_channels=4,  kernel=(1, 4),  pool=(1, 8)),
                dict(out_channels=16, kernel=(1, 16), pool=(1, 4)),
                dict(out_channels=16, kernel=(1, 8),  pool=(1, 4)),
                dict(out_channels=16, kernel=(14, 1), pool=(1, 1)),  
                dict(out_channels=16, kernel=(14, 1),  pool=(1, 1)),
            ]

        self.blocks = nn.ModuleList()
        in_ch = 1

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

            # Apply Padding on the time dimension
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=(int(k_c), int(k_t)), stride=stride, padding=(0, int(k_t)//2), padding_mode='zeros')

            layers += [conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]

            layers += [nn.MaxPool2d(kernel_size=(pool_c, pool_t), stride=(pool_c, pool_t))]

            self.blocks.append(nn.Sequential(*layers))
            in_ch = out_ch


        # Dynamic pooling over remaining (channel-height, time-width) -> (1,1)
        if global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError("global_pool must be 'avg' or 'max'")

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x: (B, C, T)
        x = x[:, None]  # (B, 1, C, T)

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x)      # (B, channels_last, 1, 1)
        x = torch.flatten(x, 1)      # (B, channels_last)
        x = self.dropout(x)
        x = self.fc(x)               # (B, output_classes)
        return x
