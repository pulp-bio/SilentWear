# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Utilities for applying filters to EMG data, including handling of NaN values and loading filter configurations from JSON files.
"""

import json
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch

CONFIG_PATH = Path(__file__).parent / "config" / "plot_config.json"


def apply_single_filter(data: np.ndarray, fs: float, filter_def: dict) -> np.ndarray:
    """Apply a single filter to the data based on the filter definition."""
    filter_type = filter_def["type"].lower()

    if filter_type == "bandpass":
        sos = butter(
            int(filter_def["order"]),
            [float(filter_def["lowcut"]), float(filter_def["highcut"])],
            btype="bandpass",
            fs=float(fs),
            output="sos",
        )
        return sosfiltfilt(sos, data, axis=0)

    if filter_type == "highpass":
        sos = butter(
            int(filter_def["order"]),
            float(filter_def["cutoff"]),
            btype="highpass",
            fs=float(fs),
            output="sos",
        )
        return sosfiltfilt(sos, data, axis=0)

    if filter_type == "lowpass":
        sos = butter(
            int(filter_def["order"]),
            float(filter_def["cutoff"]),
            btype="lowpass",
            fs=float(fs),
            output="sos",
        )
        return sosfiltfilt(sos, data, axis=0)

    if filter_type == "notch":
        b, a = iirnotch(
            w0=float(filter_def["freq"]),
            Q=float(filter_def["q"]),
            fs=float(fs),
        )
        return filtfilt(b, a, data, axis=0)

    raise ValueError(f"Unsupported filter type in config: {filter_type}")


def _apply_filters_no_nan(data: np.ndarray, fs: float, filter_list: list[dict]) -> np.ndarray:
    out = np.asarray(data, dtype=np.float64)
    for filter_def in filter_list:
        out = apply_single_filter(out, fs, filter_def)
    return out


def interpolate_nans_1d(channel: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in a 1D channel by linear interpolation over finite samples."""
    interpolated = np.asarray(channel, dtype=np.float64).copy()
    finite_mask = np.isfinite(interpolated)

    if not np.any(finite_mask) or np.all(finite_mask):
        return interpolated

    sample_idx = np.arange(interpolated.size, dtype=np.float64)
    interpolated[~finite_mask] = np.interp(
        sample_idx[~finite_mask],
        sample_idx[finite_mask],
        interpolated[finite_mask],
    )
    return interpolated


def iter_true_runs(mask: np.ndarray):
    """Yield start and end indices of consecutive True runs in a boolean mask."""
    start = None
    for idx, is_true in enumerate(mask):
        if is_true and start is None:
            start = idx
        elif not is_true and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, mask.size


def apply_filters_nan(data: np.ndarray, fs: float, filter_list: list[dict]) -> np.ndarray:
    """Apply filters to data that may contain NaNs, processing only finite segments."""
    arr = np.asarray(data, dtype=np.float64)

    # If the array has no NaNs, apply the filters directly.
    if np.isfinite(arr).all():
        return _apply_filters_no_nan(arr, fs, filter_list)

    # Create an output array initialized with NaNs, and fill in the filtered values only for finite segments.
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for channel in range(arr.shape[1]):
        series = arr[:, channel]
        finite_mask = np.isfinite(series)
        for start, end in iter_true_runs(finite_mask):
            segment = series[start:end].reshape(-1, 1)
            filtered = _apply_filters_no_nan(segment, fs, filter_list).reshape(-1)
            out[start:end, channel] = filtered

    return out


def load_signal_filters(config_path: Path) -> dict[str, dict]:
    """Load filter configurations from a JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config["signals"]
