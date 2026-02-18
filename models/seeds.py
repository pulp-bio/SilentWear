# Definitions of seeds for Data Preparation Loaders and Training script
# Following the guidelines at: https://docs.pytorch.org/docs/2.5/notes/randomness.html
import os
# needed for GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
PD_SAMPLE_SEED = 42             #42, 52, 62
TORCH_MANUAL_SEED =42           #42, 52, 62
RANDOM_SEED = 0                #0,  10, 20
RGN_SEED = 42                   #42, 52,62
torch.manual_seed(TORCH_MANUAL_SEED)

if torch.cuda.is_available():
    print(os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("Cuda is available")
torch.use_deterministic_algorithms(True)
# For custom operation, might want to set python seed as well:
import random
random.seed(RANDOM_SEED)
# set also numpy seed
import numpy as np
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RGN_SEED)
print("SEEDS SET!")
