# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Reproducibility Utilities

Sets deterministic seeds for:
- Python
- NumPy
- PyTorch (CPU + CUDA)

Follows PyTorch reproducibility guidelines:
https://pytorch.org/docs/stable/notes/randomness.html
"""

import os

# needed for GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random

import numpy as np
import torch

PD_SAMPLE_SEED = 42  # 42, 52, 62
TORCH_MANUAL_SEED = 42  # 42, 52, 62
RANDOM_SEED = 0  # 0,  10, 20
RGN_SEED = 42  # 42, 52,62

if torch.cuda.is_available():
    print(os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Cuda is available")
torch.use_deterministic_algorithms(True)


def set_seeds() -> None:
    """
    Reset the global Python, NumPy and PyTorch RNGs.

    Called before every model is built, so each run starts from the same random
    state regardless of what was trained earlier in the same process.
    """
    torch.manual_seed(TORCH_MANUAL_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


set_seeds()
rng = np.random.default_rng(RGN_SEED)
print("SEEDS SET!")
