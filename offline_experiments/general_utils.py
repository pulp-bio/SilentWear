# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
General Utilities for Offline Experiments
"""

import sys
from pathlib import Path
import re
import json
import datetime

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.II_feature_extraction.FeatExtractorManager import FeatureRegistry

#################################### Utils for Data Preparation ######################################


def feature_names_to_consider(
    consider_time_feats: bool = True,
    consider_freq_feats: bool = True,
    consider_wavelet_feats: bool = True,
):
    """
    Returns the base feature names to consider depending on flags.
    """
    features = []

    if consider_time_feats:
        features += FeatureRegistry.TIME_DOMAIN

    if consider_freq_feats:
        features += FeatureRegistry.FREQUENCY_DOMAIN

    if consider_wavelet_feats:
        features += FeatureRegistry.WAVELET_DOMAIN

    return features


def feature_columns_to_consider(feature_names, df):
    """
    Returns only the DataFrame columns corresponding to selected base feature names.

    feature_names: list[str]
    """

    if not feature_names:
        raise ValueError("No feature names provided!")

    # <feature>_<win>_Ch_<idx>_filt
    pattern = r"^(" + "|".join(map(re.escape, feature_names)) + r")_\d+_Ch_\d+_filt$"

    selected_cols = [c for c in df.columns if re.search(pattern, c)]
    return selected_cols


def reorder_ml_features_by_channel(cols, channel_order):
    """
    Reorder ML feature columns based on channel_order.
    Feature names must contain pattern: _Ch_<idx>_filt
    """

    order_position = {ch: i for i, ch in enumerate(channel_order)}

    parsed = []
    for col in cols:
        match = re.search(r"_Ch_(\d+)", col)
        if match:
            ch_idx = int(match.group(1))
            ch_rank = order_position.get(ch_idx, 10**9)
        else:
            # if no channel info, push to end
            ch_rank = 10**9

        parsed.append((ch_rank, col))

    # stable sort: python sort is stable → preserves feature grouping inside channel
    parsed_sorted = sorted(parsed, key=lambda x: x[0])

    return [col for _, col in parsed_sorted]


#################################### Utils to override configs ######################################


def deep_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with dict u (u wins)."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


#################### Utils to Keep Track of Runs ##############################


def mark_running(run_dir: Path, meta: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUNNING").write_text(datetime.now().isoformat())
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def mark_done(run_dir: Path):
    running = run_dir / "RUNNING"
    if running.exists():
        running.unlink()
    (run_dir / "DONE").write_text(datetime.now().isoformat())


def mark_failed(run_dir: Path, err: str):
    (run_dir / "FAILED").write_text(err)
    # keep RUNNING as evidence if you want; or remove it:
    running = run_dir / "RUNNING"
    if running.exists():
        running.unlink()


def should_skip(run_dir: Path, *, rerun_failed=False, rerun_running=False) -> bool:
    if (run_dir / "DONE").exists():
        return True
    if (run_dir / "RUNNING").exists() and not rerun_running:
        return True
    if (run_dir / "FAILED").exists() and not rerun_failed:
        return True
    return False


#################### Utils for data loading ################################


## TO-DO: check which functions use this and replace with load_function in utils/general_utils.py
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def check_data_directories(
    main_data_directory: Path,
    all_subjects_models: bool,
    sub_id,
    condition: str,
    window_size_ms: int,
    base_config: dict,
):
    """
    Returns data directories contaning data for training, depending on the desired training config

    Returns
    -------
    List[Path]
        List of valid data directories.

    Raises
    ------
    FileNotFoundError
        If any expected directory does not exist.
    """
    data_dirs = []

    win_root = base_config["paths"]["win_and_feats"]

    def add_dirs_for_subject(subject):
        if condition != "voc_and_silent":
            data_dirs.append(
                main_data_directory / Path(f"{win_root}/{subject}/{condition}/WIN_{window_size_ms}")
            )
        else:
            data_dirs.append(
                main_data_directory / Path(f"{win_root}/{subject}/silent/WIN_{window_size_ms}")
            )
            data_dirs.append(
                main_data_directory / Path(f"{win_root}/{subject}/vocalized/WIN_{window_size_ms}")
            )

    # ---- single subject ----
    if not all_subjects_models:
        add_dirs_for_subject(sub_id)
    else:
        for curr_sub_id in sub_id:
            add_dirs_for_subject(curr_sub_id)

    # ---- existence check ----
    missing = [p for p in data_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The following data directories do not exist:\n" + "\n".join(str(p) for p in missing)
        )

    return data_dirs
