# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
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
from datetime import datetime
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.II_feature_extraction.FeatExtractorManager import FeatureRegistry
from models.seeds import TORCH_MANUAL_SEED, RANDOM_SEED, RGN_SEED
from utils.I_data_preparation.read_bio_file import parse_bio_filename
from utils.I_data_preparation.experimental_config import (
    RAW_DIRNAME,
    RAW_AND_FILTERED_DIRNAME,
    WINS_AND_FEATURES_DIRNAME,
    WINDOW_DIR_PREFIX,
    SILENT_DIRNAME,
    VOCALIZED_DIRNAME,
)

# Matches the session id in processed/window filenames (sess_<id>_batch_<id>.h5).
SESSION_RE = re.compile(r"sess_(\d+)")


# ---------------------------------------------------------------------------
# Utils for Data Preparation
# ---------------------------------------------------------------------------


def feature_names_to_consider(
    consider_time_feats: bool = True,
    consider_freq_feats: bool = True,
    consider_wavelet_feats: bool = True,
) -> List[str]:
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


def feature_columns_to_consider(feature_names: List[str], df: pd.DataFrame) -> List[str]:
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


def reorder_ml_features_by_channel(cols: List[str], channel_order: List[int]) -> List[str]:
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


# ---------------------------------------------------------------------------
# Utils to override configs
# ---------------------------------------------------------------------------


def deep_update(d: dict, u: dict) -> dict:
    """Recursively update dict d with dict u (u wins)."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


# ---------------------------------------------------------------------------
# Utils to Keep Track of Runs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Utils for data loading
# ---------------------------------------------------------------------------


## TO-DO: check which functions use this and replace with load_function in utils/general_utils.py
def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Dataset directories and splits
# ---------------------------------------------------------------------------


def build_wins_feats_dirs(
    main_data_directory: Path,
    base_config: dict,
    sub_id: Union[str, List[str]],
    condition: str,
    window_size_ms: int,
) -> List[Path]:
    """Build the windows/features directories for the requested subject(s)/condition.

    The windows folder name is taken from base_config["paths"]["win_and_feats"]
    when present, otherwise from the canonical WINS_AND_FEATURES_DIRNAME constant,
    keeping a single source of truth for the folder layout.
    """
    win_root = base_config.get("paths", {}).get("win_and_feats", WINS_AND_FEATURES_DIRNAME)
    subjects = [sub_id] if isinstance(sub_id, str) else list(sub_id)
    conditions = (
        [SILENT_DIRNAME, VOCALIZED_DIRNAME] if condition == "voc_and_silent" else [condition]
    )
    win_subdir = f"{WINDOW_DIR_PREFIX}{window_size_ms}"

    dirs: List[Path] = []
    for subject in subjects:
        for cond in conditions:
            dirs.append(Path(main_data_directory) / win_root / str(subject) / str(cond) / win_subdir)
    return dirs


def check_data_directories(
    main_data_directory: Path,
    all_subjects_models: bool,
    sub_id: Union[str, List[str]],
    condition: str,
    window_size_ms: int,
    base_config: dict,
) -> List[Path]:
    """
    Returns data directories containing data for training, depending on the desired training config.

    Returns
    -------
    List[Path]
        List of valid data directories.

    Raises
    ------
    FileNotFoundError
        If any expected directory does not exist.
    """
    data_dirs = build_wins_feats_dirs(
        main_data_directory, base_config, sub_id, condition, window_size_ms
    )

    # ---- existence check ----
    missing = [p for p in data_dirs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "The following data directories do not exist:\n" + "\n".join(str(p) for p in missing)
        )

    return data_dirs


def base_window_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Returns only the rows corresponding to the base window (i.e., no augmentation)."""
    if "augmentation_direction" not in df.columns:
        return df.copy()
    return df[df["augmentation_direction"] == "base"].copy()


def training_rows_with_augmentation(
    augmented_size_df: pd.DataFrame,
    train_base_df: pd.DataFrame,
    mode: str = "augmented_size",
    seed: int = 0,
) -> pd.DataFrame:
    """Returns the rows for training.

    - mode='augmented_size': keep all augmented rows derived from the base training split.
    - mode='original_size': sample the same number of rows as the original training split
      from the augmented pool, preserving label balance as much as possible.
    """
    if "augmentation_source_id" not in augmented_size_df.columns:
        return train_base_df.copy()

    train_source_ids = train_base_df["augmentation_source_id"].drop_duplicates()
    candidate_df = augmented_size_df[augmented_size_df["augmentation_source_id"].isin(train_source_ids)].copy()

    if mode == "original_size":
        target_size = int(len(train_base_df))
        if target_size < len(candidate_df):
            print(
                f"[DEBUG] original_size training mode:"
                f"\nAugmented candidate windows={len(candidate_df)}. "
                f"\nSampled windows={target_size}."
                f"\nOriginal base windows={len(train_base_df)}."
            )
            stratify = candidate_df["Label_int"] if "Label_int" in candidate_df.columns else None
            sample_ratio = target_size / len(candidate_df)
            sampled_train, _ = train_test_split(
                candidate_df,
                train_size=sample_ratio,
                random_state=seed,
                stratify=stratify,
                shuffle=True,
            )
            return sampled_train

    return candidate_df


# ---------------------------------------------------------------------------
# Utils for input data normalization
# ---------------------------------------------------------------------------


GLOBAL_NORM_KEY = "__global__"
NORMALIZATION_METHODS = ("zscore", "minmax")
DEFAULT_NORM_PERCENTILE = 97.5


def _flatten_window_column(series: pd.Series) -> np.ndarray:
    """Flatten a column holding one array per window into a 1-D array of samples."""
    arrays = [np.asarray(value, dtype=np.float64).ravel() for value in series.to_numpy()]
    if not arrays:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(arrays)


def fit_normalization_stats(
    df_train: pd.DataFrame,
    cols: List[str],
    kind: str = "dl",
    group_col: str = "subject_id",
    eps: float = 1e-8,
    method: str = "zscore",
    percentile: float = DEFAULT_NORM_PERCENTILE,
    clip_sigma: float = 0.0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute per-group, per-column normalization statistics from the training split.

    The statistics are grouped by ``group_col`` (``subject_id`` by default), so
    that pooled runs put every subject on a common scale instead of letting the
    subject with the largest EMG amplitude dominate. Fitting only on the
    training rows keeps validation and test data out of the statistics.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split (augmented rows included, they are training data).
    cols : List[str]
        Columns to normalize: the channel columns (``Ch_<i>_filt``) for DL runs,
        the feature columns for ML runs.
    kind : str
        'dl' when each cell holds the array of samples of one window/channel,
        'ml' when each cell holds a scalar feature.
    group_col : str
        Column defining the normalization groups. If missing from the DataFrame,
        all rows are treated as a single group.
    eps : float
        Scales below this value are replaced by a unit scale, so that a constant
        (e.g. disconnected) channel is centred but not amplified.
    method : str
        'zscore' subtracts the mean and divides by the standard deviation.
        'minmax' maps the ``[100 - percentile, percentile]`` range onto [-1, 1].
    percentile : float
        Upper percentile bounding the min-max range, in (50, 100].
        Ignored by 'zscore'.
    clip_sigma : float
        'zscore' only. When positive, the standardized values are clipped to
        +/- this many standard deviations.

    Returns
    -------
    dict
        ``{group: {column: {...}}}``, always including the ``GLOBAL_NORM_KEY``
        fallback group. Each column dict carries its own ``method`` key, so the
        statistics describe how they must be applied.
    """
    if kind not in ("dl", "ml"):
        raise ValueError(f"Unknown model kind for normalization: {kind}")
    if method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"Unknown normalization method '{method}'; expected one of {NORMALIZATION_METHODS}."
        )
    if not 50.0 < float(percentile) <= 100.0:
        raise ValueError(
            f"normalization_percentile must lie in (50, 100], got {percentile}."
        )
    if float(clip_sigma) < 0.0:
        raise ValueError(
            f"normalization_clip_sigma must be >= 0 (0 disables it), got {clip_sigma}."
        )

    if group_col in df_train.columns:
        group_values = df_train[group_col].astype(str)
        groups = list(pd.unique(group_values))
    else:
        print(
            f"[NORMALIZATION] Column '{group_col}' not found: "
            "computing a single set of statistics over the whole training split."
        )
        group_values = pd.Series(GLOBAL_NORM_KEY, index=df_train.index)
        groups = []

    stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for group in groups + [GLOBAL_NORM_KEY]:
        rows = df_train if group == GLOBAL_NORM_KEY else df_train[group_values == group]
        if rows.empty:
            continue

        group_stats: Dict[str, Dict[str, Any]] = {}
        for col in cols:
            values = _flatten_window_column(rows[col]) if kind == "dl" else rows[col].to_numpy(
                dtype=np.float64
            )
            if method == "zscore":
                std = float(np.std(values))
                group_stats[col] = {
                    "method": "zscore",
                    "mean": float(np.mean(values)),
                    "std": std if std > eps else 1.0,
                    "clip_sigma": float(clip_sigma) if clip_sigma else 0.0,
                    "n": int(values.size),
                }
            else:
                lo = float(np.percentile(values, 100.0 - percentile))
                hi = float(np.percentile(values, percentile))
                span = hi - lo
                if span <= eps:
                    lo, hi, span = lo - 0.5, lo + 0.5, 1.0
                group_stats[col] = {
                    "method": "minmax",
                    "lo": lo,
                    "hi": hi,
                    "span": span,
                    "percentile": float(percentile),
                    "n": int(values.size),
                }
        stats[group] = group_stats

    return stats


def apply_normalization_stats(
    df: pd.DataFrame,
    cols: List[str],
    stats: Dict[str, Dict[str, Dict[str, float]]],
    kind: str = "dl",
    group_col: str = "subject_id",
    split_name: str = "",
) -> pd.DataFrame:
    """Apply normalization statistics fitted on the training split.

    Returns a copy of ``df`` with ``cols`` normalized in place of the original
    values; every other column is left untouched.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    if group_col in df.columns:
        group_values = df[group_col].astype(str).to_numpy()
    else:
        group_values = np.full(len(df), GLOBAL_NORM_KEY, dtype=object)

    group_positions = {}
    for group in np.unique(group_values):
        if group not in stats:
            print(
                f"[NORMALIZATION] {split_name or 'split'}: no training statistics for "
                f"'{group}', falling back to the pooled training statistics."
            )
        group_positions[group] = np.flatnonzero(group_values == group)

    for col in cols:
        col_values = df[col].to_numpy(copy=True)
        for group, positions in group_positions.items():
            col_stats = stats.get(group, stats[GLOBAL_NORM_KEY])[col]
            if col_stats.get("method", "zscore") == "zscore":
                mean, std = col_stats["mean"], col_stats["std"]
                sigma = float(col_stats.get("clip_sigma", 0.0))
                if sigma > 0.0:
                    transform = lambda v: np.clip((v - mean) / std, -sigma, sigma)
                else:
                    transform = lambda v: (v - mean) / std
            else:
                lo, span = col_stats["lo"], col_stats["span"]
                transform = lambda v: np.clip(2.0 * (v - lo) / span - 1.0, -1.0, 1.0)

            if kind == "dl":
                for pos in positions:
                    window = np.asarray(col_values[pos], dtype=np.float32)
                    col_values[pos] = transform(window)
            else:
                col_values[positions] = transform(col_values[positions])
        df[col] = col_values

    return df


def normalize_datasets(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: List[str],
    kind: str = "dl",
    group_col: str = "subject_id",
    method: str = "zscore",
    percentile: float = DEFAULT_NORM_PERCENTILE,
    clip_sigma: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Dict[str, float]]]]:
    """Normalize train/val/test using statistics fitted on the training split only."""
    stats = fit_normalization_stats(
        df_train, cols, kind=kind, group_col=group_col, method=method,
        percentile=percentile, clip_sigma=clip_sigma,
    )

    groups = [g for g in stats if g != GLOBAL_NORM_KEY]
    if method == "zscore":
        scale = "z-score" + (f" clipped at +/-{clip_sigma:g} sigma" if clip_sigma else "")
    else:
        scale = f"min-max on the [{100.0 - percentile:g}, {percentile:g}] percentile range"
    print(
        f"[NORMALIZATION] {scale} of {len(cols)} '{kind}' columns, "
        f"grouped by '{group_col}' ({len(groups)} group(s): {groups}), "
        "statistics fitted on the training split."
    )

    df_train = apply_normalization_stats(
        df_train, cols, stats, kind=kind, group_col=group_col, split_name="train"
    )
    df_val = apply_normalization_stats(
        df_val, cols, stats, kind=kind, group_col=group_col, split_name="val"
    )
    df_test = apply_normalization_stats(
        df_test, cols, stats, kind=kind, group_col=group_col, split_name="test"
    )

    return df_train, df_val, df_test, stats


def apply_datasets_normalization(
    model_master,
    base_config: dict,
    save_model_path: Optional[Path] = None,
    group_col: str = "subject_id",
) -> bool:
    """Normalize the splits held by ``model_master`` when the config asks for it.

    Enabled by ``experiment.data_normalization: true`` in the base config. The
    normalized columns are the ones the model consumes (channels for
    DL runs, features for ML runs), normalized per subject and per column with
    statistics fitted on the training split. Two further keys select the
    transform:

    ``experiment.normalization_kind``       'zscore' (default) or 'minmax'
    ``experiment.normalization_percentile`` upper bound of the min-max range,
                                            default DEFAULT_NORM_PERCENTILE
    ``experiment.normalization_clip_sigma`` z-score clip in standard deviations,
                                            default 0 (disabled)

    ``utils/II_feature_extraction/amplitude_percentile_analysis.py`` measures the
    last two from a dataset.

    Returns True when normalization was applied.
    """
    experiment_cfg = base_config.get("experiment", {})
    if not bool(experiment_cfg.get("data_normalization", False)):
        return False

    method = str(experiment_cfg.get("normalization_kind", "zscore")).strip().lower()
    percentile = float(experiment_cfg.get("normalization_percentile", DEFAULT_NORM_PERCENTILE))
    clip_sigma = float(experiment_cfg.get("normalization_clip_sigma", 0.0))

    cols = model_master.extract_dataset_train_columns()
    df_train, df_val, df_test, stats = normalize_datasets(
        model_master.df_train,
        model_master.df_val,
        model_master.df_test,
        cols,
        kind=model_master.kind,
        group_col=group_col,
        method=method,
        percentile=percentile,
        clip_sigma=clip_sigma,
    )
    model_master.df_train = df_train
    model_master.df_val = df_val
    model_master.df_test = df_test

    save_normalization_stats(stats, save_model_path)
    return True


def save_normalization_stats(
    stats: Dict[str, Dict[str, Dict[str, float]]], save_model_path: Optional[Path]
) -> None:
    """Dump the fold statistics next to the fold checkpoint, as JSON."""
    if save_model_path is None:
        return
    save_model_path = Path(save_model_path)
    out_path = save_model_path.with_name(save_model_path.stem + "_normalization.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(f"[NORMALIZATION] statistics -> {out_path}")


# ---------------------------------------------------------------------------
# Seeding and session discovery
# ---------------------------------------------------------------------------


def reset_all_seeds():
    """Resets all random seeds for PyTorch, NumPy, and Python's random module."""
    torch.manual_seed(TORCH_MANUAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_MANUAL_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    np.random.seed(RGN_SEED)
    random.seed(RANDOM_SEED)
    

def discover_sessions(data_dir: Path, subject: str, condition: str) -> List[int]:
    """Discover available session IDs for the given subject and condition.

    The primary source is the filtered recordings (``data_raw_and_filt``), which
    are part of the public dataset and are named ``sess_<id>_batch_<id>.h5``.
    Falls back to the raw ``.bio`` recordings (``raw``) for self-collected data.
    """
    data_dir = Path(data_dir)
    sessions = set()

    filt_dir = data_dir / RAW_AND_FILTERED_DIRNAME / subject / condition
    if filt_dir.exists():
        for h5_path in filt_dir.glob("*.h5"):
            match = SESSION_RE.search(h5_path.name)
            if match:
                sessions.add(int(match.group(1)))

    if not sessions:
        raw_dir = data_dir / RAW_DIRNAME / subject / condition
        if raw_dir.exists():
            for bio_path in sorted(raw_dir.glob("*.bio")):
                parsed = parse_bio_filename(bio_path)
                if parsed is None:
                    continue
                session_id, _, _ = parsed
                sessions.add(int(session_id))

    return sorted(sessions)
