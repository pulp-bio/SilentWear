"""
Variation of SpeechNet, with Padding on Time Dimension
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Union, Optional
from torchinfo import summary
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]   
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
from models.utils import count_params


KernelH = Union[int, str]  # int or "full"


class BaseSpeechNetWithPadding(nn.Module):
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
                dict(out_channels=16, kernel=(14, 1), pool=(1, 1)),  # full --> collapses channels. Set equal to channel numbers for now
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

            #padding = cfg.get("padding", (0, 0))

            layers = []

            # here the variation -> apply padding on the TimeDimension
            #x_pad = F.pad(dummy_emg_input, (pad, pad, 0, 0), mode="replicate")  # (left,right,top,bottom)
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





def is_combo_valid(kname, pname):

    invalid = {
        ("K_default", "P_minimal_shared"),  # the one that crashed
        ("K_simple", "P_default"),  # the one that crashed
        ("K_simple_shared", "P_default"),  # the one that crashed
        ("K_simple_shared", "P_minimal_shared"),  # the one that crashed
        # add more here...
    }
    return (kname, pname) not in invalid


def build_blocks_config(channels, kernels, pools):
    assert len(channels) == len(kernels) == len(pools) == 5, "Expected 5 blocks"
    blocks = []
    for out_ch, ker, pool in zip(channels, kernels, pools):
        blocks.append(dict(
            out_channels=int(out_ch),
            kernel=(ker[0], ker[1]),   # ker[0] can be int or "full"
            pool=(pool[0], pool[1]),
        ))
    return blocks


if __name__ == "__main__":
    
    num_emg_channels = 14
    num_time_samples = 1400
    num_classes = 9

    out_channels_sweep = [
        ("C_sel", [8, 16, 16, 32, 32]),                
    ]
    kernel_sweep = [
        ("K_sel",       [[1, 4], [1, 16], [1, 8], [7, 1], [7, 1]]),                # full -> channels collapse. 
    ]

    pool_sweep = [
        ("P_sel",        [[1, 8], [1, 4], [1, 4], [1, 1], [1, 1]]),
    ]

    # Print header
    print(f"{'variant':40s} | {'total_params':>12s} | {'trainable':>12s}")
    print("-" * 75)

    variant= 0
    for cname, chans in out_channels_sweep:
        for kname, kernels in kernel_sweep:
            for pname, pools in pool_sweep:
                variant_name = f"{cname}__{kname}__{pname}"
                #print("Running variant", variant_name)

                if not is_combo_valid(kname, pname):
                    #print(f"SKIP (full-kernel would break): {cname}__{kname}__{pname}\n")
                    continue

                blocks_config = build_blocks_config(chans, kernels, pools)

                model = BaseSpeechNetWithPadding(
                    C=num_emg_channels,
                    T=num_time_samples,
                    output_classes=num_classes,
                    blocks_config=blocks_config,
                    p_dropout=0.1,
                )

                total, trainable = count_params(model)
                print(f"{variant_name:40s} | {total:12d} | {trainable:12d}")
                print("\n\n")
                dummy_emg_input = torch.rand(2, num_emg_channels, num_time_samples)
                summary(model, 
                    input_data=dummy_emg_input)
                
                # just random to check if model can work
                with torch.no_grad():
                    y = model(dummy_emg_input)
                    assert y.shape == (2, num_classes)

                variant +=1


    print("Total variants:", variant)