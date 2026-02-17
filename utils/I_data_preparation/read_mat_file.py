"""
File used to parse a .mat file
Needed to map recordings from original paper (https://ieeexplore.ieee.org/abstract/document/11330464) into new data format
"""

import struct
import numpy as np
from pathlib import Path
import sys
import pandas as pd
from pathlib import Path
import re 
import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
from I_data_preparation.emg_processing import apply_filters

# Channel configuration
emg_num_channels_all_on_device = 16
channels_without_useful_data = [12, 13]  # 1-indexed channels to exclude

MAT_RE = re.compile(
    r"sess_(?P<session>\d+)_batch_(?P<batch>\d+).mat$"
)

MAT_LABLES = {
        10 : "rest", 
        50: "left",
        60: "right",
        70: "up",
        80: "down",
        90: "forward",
        100: "backward",
        110: "stop",
        120: "go"
    }

FS = 500

def parse_mat_filename(path: Path):
    """
    Returns (session_id:int, batch_id:int) or None if no match from a .mat file
    """
    m = MAT_RE.search(path.name)
    if not m:
        return None
    return int(m.group("session")), int(m.group("batch")), np.nan


def load_emg_data_from_mat(mat_file: Path) -> tuple:
    """Load EMG data and trigger from mat file.

    Args:
        mat_file: Path to mat file.

    Returns:
        Tuple of (emg_raw, trigger).
    """
    emg_mat = scipy.io.loadmat(mat_file)
    exg_data = emg_mat["ExGData"]
    emg_raw_data = exg_data["Data"][0][0]
    trigger = exg_data["Trigger"][0][0].flatten()

    # Filter channels
    all_channel_idx = np.arange(0, emg_num_channels_all_on_device)
    channels_without_useful_data_idx = (
        np.array(channels_without_useful_data) - 1
    )
    used_channel_idx = all_channel_idx[
        ~np.isin(all_channel_idx, channels_without_useful_data_idx)
    ]
    emg_raw = emg_raw_data[:, used_channel_idx].copy()

    #print(np.unique(trigger))
    return emg_raw, trigger


def prepare_dataset_from_mat(emg_raw, trigger, hp_cutoff, notch_cutoff):

    emg_data = emg_raw
    trigger = trigger.astype(int, copy=False)

    # --- Drop samples whose trigger is not in MAT_LABLES 
    valid_keys = set(MAT_LABLES.keys())

    trigger_series = pd.Series(trigger)
    is_valid = trigger_series.isin(valid_keys)

    # Apply mask to data and trigger
    emg_data = emg_data[is_valid.values, :]
    trigger = trigger[is_valid.values]

    print("Unique triggers after dropping invalid:", np.unique(trigger))

    # Map labels (now guaranteed to exist)
    labels = pd.Series(trigger).map(MAT_LABLES)  # will be non-NaN now

    # Build EMG DataFrame
    emg_df = pd.DataFrame(
        emg_data,
        columns=[f"Ch_{i}" for i in range(emg_data.shape[1])]
    )
    emg_df["Label_int"] = trigger
    emg_df["Label_str"] = labels.values

    # Filter data
    for i in range(emg_data.shape[1]):
        emg_df[f"Ch_{i}_filt"] = apply_filters(
            emg_data[:, i],
            FS,
            highpass_cutoff=hp_cutoff,
            notch_cutoff=notch_cutoff
        )

    # Trim recording considering only first and last non-rest label (assuming rest is 10; adjust if needed)
    non_rest = emg_df["Label_int"] != 10
    if non_rest.any():
        first_label_loc = emg_df.index[non_rest][0] - FS
        last_label_loc = emg_df.index[non_rest][-1] + FS
        first_label_loc = max(first_label_loc, emg_df.index[0])
        last_label_loc = min(last_label_loc, emg_df.index[-1])
        emg_df = emg_df.loc[first_label_loc:last_label_loc]

    return emg_df
