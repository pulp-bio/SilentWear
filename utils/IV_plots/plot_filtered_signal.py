# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Script for plotting signals from .bio files with optional filtering based on a JSON config.
"""

import sys
import os
import argparse
import numpy as np

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.screen=false")

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.I_data_preparation.read_bio_file import read_bio_file
from utils.IV_plots.filter import apply_filters_nan, load_signal_filters, iter_true_runs

CONFIG_PATH = Path(__file__).parent/"config"/"plot_config.json"

LINE_WIDTH = 0.8

# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------


SIGNAL_CONFIG = {
    "emg":     {"channel": 0, "threshold": 10000.0},
    "mic_emg": {"channel": 0, "threshold": 0.01},
    "eeg":     {"channel": 0, "threshold": 10000.0},
    "mic_eeg": {"channel": 0, "threshold": 0.01},
}

TRIM_SECONDS = 0.1           # [s]
CUTOFF_HZ = 20.0
HIGHPASS_FILTERS = [{"type": "highpass", "order": 4, "cutoff": CUTOFF_HZ}]


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------


def expand_timestamps_to_samples(
    timestamp_data: np.ndarray,
    fs_signal: float,
    fs_timestamp: float,
) -> np.ndarray:
    """Expand packet-level hardware timestamps (µs) to per-sample timestamps (µs), rescaled to t=0.

    Each hardware timestamp marks the last sample of its packet.
    The inter-sample step is ``1_000_000 / fs_signal`` µs.
    """
    ts = np.asarray(timestamp_data, dtype=np.float64).reshape(-1)
    samples_per_packet = int(round(fs_signal / fs_timestamp))
    sample_step_us = 1_000_000.0 / fs_signal
    offsets = (np.arange(samples_per_packet, dtype=np.float64) - (samples_per_packet - 1)) * sample_step_us
    expanded = (ts[:, np.newaxis] + offsets[np.newaxis, :]).reshape(-1)
    expanded -= expanded[0]
    return expanded


def extract_time_axis(
    sig_data: dict,
    ts_entry: dict | None,
) -> tuple[np.ndarray, int]:
    """Build the time axis in seconds for a signal.

    If a hardware timestamp entry is present, expands it to per-sample
    resolution via expand_timestamps_to_samples (rescaled to t=0).
    Otherwise builds a synthetic axis from sample index and fs.
    """
    n_samp = sig_data["data"].shape[0]

    if ts_entry is not None:
        # print(f"[timestamp] using hardware timestamps (fs={ts_entry['fs']} Hz, {len(np.asarray(ts_entry['data']).ravel())} packets)")
        ts_arr = np.asarray(ts_entry["data"]).ravel()
        t = expand_timestamps_to_samples(ts_arr, sig_data["fs"], ts_entry["fs"]) / 1_000_000.0
    else:
        # print(f"[timestamp] using synthetic timestamps (fs={sig_data['fs']} Hz, {n_samp} samples)")
        t = np.arange(n_samp, dtype=np.float64) / sig_data["fs"]

    min_len = min(n_samp, len(t))
    return t[:min_len], min_len


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _get_kept_channel_ids(n_ch: int, exclude_channels: list[int]) -> list[int]:
    """Return channel indices that should be kept after applying exclusions."""
    excluded = set()
    for channel_idx in exclude_channels:
        try:
            idx = int(channel_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_ch:
            excluded.add(idx)

    return [idx for idx in range(n_ch) if idx not in excluded]


def apply_channel_exclusions(signals: dict[str, dict], signal_cfg: dict[str, dict]) -> None:
    """Apply channel exclusions from config to every matching signal."""
    for sig_name, sig_data in signals.items():
        data = np.asarray(sig_data["data"])
        if data.ndim != 2:
            continue

        cfg = signal_cfg.get(sig_name, {})
        exclude_channels = cfg.get("exclude_channels")
        if not exclude_channels:
            sig_data["channel_ids"] = list(range(data.shape[1]))
            continue

        kept_ids = _get_kept_channel_ids(data.shape[1], exclude_channels)
        sig_data["data"] = data[:, kept_ids]
        sig_data["channel_ids"] = kept_ids


# ---------------------------------------------------------------------------
# Plot preparation utilities
# ---------------------------------------------------------------------------


def compute_channel_spacing(data: np.ndarray) -> float:
    """Compute vertical spacing for plotting multiple channels."""
    q95 = np.nanpercentile(data, 95, axis=0)
    q05 = np.nanpercentile(data, 5, axis=0)
    spread = np.nanmedian(q95 - q05)
    if not np.isfinite(spread) or spread <= 0:
        return 1.0
    return float(spread * 1.5)


def center_finite_runs(channel: np.ndarray) -> np.ndarray:
    """Center the finite values of a channel by subtracting the global mean of finite values."""
    centered = np.asarray(channel, dtype=np.float64).copy()
    finite_mask = np.isfinite(centered)
    if np.any(finite_mask):
        global_mean = np.mean(centered[finite_mask])
        centered[finite_mask] -= global_mean
    return centered


def color_nan_regions(ax, t: np.ndarray, data: np.ndarray) -> None:
    """Highlight regions in the plot where data contains NaNs."""
    nan_mask = np.any(~np.isfinite(data), axis=1)
    if not np.any(nan_mask):
        return

    dt = float(t[1] - t[0]) if t.size > 1 else 0.0
    for start, end in iter_true_runs(nan_mask):
        x0 = float(t[start - 1]) if start > 0 else float(t[start])
        x1 = float(t[end - 1] + dt)
        ax.axvspan(x0, x1, color="red", alpha=0.18, zorder=0)


def update_x_ticks(ax) -> None:
    """Set adaptive x ticks."""
    xmin, xmax = ax.get_xlim()
    span = max(float(xmax - xmin), 1e-12)
    base = 0.10  # 100 ms

    if span <= 5.0:
        spacing = base
    else:
        desired_ticks = 10
        spacing = max(base, np.ceil((span / desired_ticks) / base) * base)

    ax.xaxis.set_major_locator(MultipleLocator(spacing))
    ax.grid(axis="x", which="major", linestyle="-", color="#e0e0e0", linewidth=0.5, alpha=0.7)


def plot_signal_on_axis(
    ax,
    sig_name: str,
    sig_data: dict,
    t: np.ndarray,
    min_len: int,
    line_width: float = LINE_WIDTH,
) -> None:
    """Plot the signal on a provided Matplotlib axis."""
    data = np.asarray(sig_data["data"], dtype=np.float64)[:min_len]
    n_ch = data.shape[1]
    channel_ids = sig_data.get("channel_ids", list(range(n_ch)))
    if len(channel_ids) != n_ch:
        channel_ids = list(range(n_ch))

    ax.set_title(sig_name)
    color_nan_regions(ax, t, data)

    if n_ch > 1:
        spacing = compute_channel_spacing(data)
        offsets = np.arange(n_ch) * spacing
        for i in range(n_ch):
            channel = center_finite_runs(data[:, i])
            ax.plot(t, channel + offsets[i], lw=line_width)
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"Ch {channel_ids[i]}" for i in range(n_ch)])
        ax.set_ylabel("Channels")
    else:
        color = "C1" if sig_name.startswith("mic_") else "C0"
        ax.plot(t, data[:, 0], label=f"Ch {channel_ids[0]}", lw=line_width, color=color)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    ax.grid(True, linestyle="-", color="#e0e0e0", linewidth=0.5, alpha=0.7)


def load_and_prepare(file_path: str, signal_filters: dict, apply_filter: bool, preprocess_for_alignment: bool) -> dict:
    """Read a .bio file, apply shared alignment preprocessing (if requested), then optional filters."""
    signals = read_bio_file(file_path)

    if preprocess_for_alignment:

        # Trimming to remove initial artifacts
        for sig_name, sig_data in signals.items():
            fs = float(sig_data["fs"])
            trim_samples = int(round(TRIM_SECONDS * fs))
            data = np.asarray(sig_data["data"])

            if data.shape[0] > trim_samples:
                sig_data["data"] = data[trim_samples:] if data.ndim == 1 else data[trim_samples:, :]

        # HPF + Rectification
        target_signals = set(SIGNAL_CONFIG.keys())

        for sig_name, sig_data in signals.items():
            if sig_name not in target_signals:
                continue
            data = np.asarray(sig_data["data"])
            if data.ndim == 2 and data.size > 0:
                print(f"[{sig_name}] Applying {CUTOFF_HZ}Hz HPF + Rectification to mirror delay pipeline")
                fs = float(sig_data["fs"])
                sig_data["data"] = np.abs(
                    apply_filters_nan(
                        data=data,
                        fs=fs,
                        filter_list=HIGHPASS_FILTERS,
                    )
                )

    if apply_filter:
        for sig_name, sig_cfg in signal_filters.items():
            filter_list = sig_cfg.get("filters")
            if sig_name in signals and filter_list:
                signals[sig_name]["data"] = apply_filters_nan(
                    data=signals[sig_name]["data"],
                    fs=signals[sig_name]["fs"],
                    filter_list=filter_list,
                )

    apply_channel_exclusions(signals, signal_filters)
    return signals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot signal from one or two .bio files.")
    parser.add_argument("file_paths", nargs="+", help="Path(s) to the .bio file(s)")
    parser.add_argument("--filter", action="store_true", help="Apply filtering to the signals")
    parser.add_argument(
        "--preprocess-for-alignment",
        action="store_true",
        help="Apply exactly the same trimming and HPF+Rectification pipeline used by compute_peak_delay only to the configured signals",
    )
    args = parser.parse_args()

    signal_filters = load_signal_filters(CONFIG_PATH)

    if len(args.file_paths) == 2:
        signals_list = [
            load_and_prepare(fp, signal_filters, args.filter, args.preprocess_for_alignment)
            for fp in args.file_paths
        ]
        filenames = [Path(fp).name for fp in args.file_paths]

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), layout="constrained", sharex=True)
        fig.suptitle(f"{filenames[0]}  |  {filenames[1]}", fontsize=12)

        ax_slots = list(axes)
        slot = 0

        for signals, filename in zip(signals_list, filenames):
            for mic_name, mic_data in signals.items():
                if not mic_name.startswith("mic_"):
                    continue
                base_name = mic_name[4:]
                if base_name not in signals:
                    continue

                base_data = signals[base_name]

                t_base, len_base = extract_time_axis(base_data, signals.get(f"timestamp_{base_name}"))
                ax_base = ax_slots[slot]
                ax_base.set_title(f"[{filename}] {base_name}")
                plot_signal_on_axis(ax_base, base_name, base_data, t_base, len_base)

                t_mic, len_mic = extract_time_axis(mic_data, signals.get(f"timestamp_{mic_name}"))
                ax_mic = ax_slots[slot + 1]
                ax_mic.set_title(f"[{filename}] {mic_name}")
                plot_signal_on_axis(ax_mic, mic_name, mic_data, t_mic, len_mic)

                slot += 2
                break

        axes[-1].set_xlabel("Time [s]")
        axes[0].callbacks.connect("xlim_changed", update_x_ticks)
        update_x_ticks(axes[0])

    else:
        file_path = args.file_paths[0]
        filename = Path(file_path).name
        signals = load_and_prepare(file_path, signal_filters, args.filter, args.preprocess_for_alignment)

        for sig_name, sig_data in signals.items():
            fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
            fig.suptitle(filename, fontsize=12)
            ts_entry = signals.get(f"timestamp_{sig_name}")
            t, min_len = extract_time_axis(sig_data, ts_entry)
            plot_signal_on_axis(ax, sig_name, sig_data, t, min_len)
            if ts_entry is None:
                ax.callbacks.connect("xlim_changed", update_x_ticks)
                update_x_ticks(ax)
            ax.set_xlabel("Time [s]")

        for mic_name, mic_data in signals.items():
            if not mic_name.startswith("mic_"):
                continue
            base_name = mic_name[4:]
            if base_name not in signals:
                continue

            base_data = signals[base_name]
            fig, (ax_top, ax_bottom) = plt.subplots(
                2, 1, figsize=(16, 8), layout="constrained", sharex=True,
            )
            fig.suptitle(filename, fontsize=12)

            ts_base = signals.get(f"timestamp_{base_name}")
            t_base, len_base = extract_time_axis(base_data, ts_base)
            plot_signal_on_axis(ax_top, base_name, base_data, t_base, len_base)

            ts_mic = signals.get(f"timestamp_{mic_name}")
            t_mic, len_mic = extract_time_axis(mic_data, ts_mic)
            plot_signal_on_axis(ax_bottom, mic_name, mic_data, t_mic, len_mic)

            if ts_base is None:
                ax_top.callbacks.connect("xlim_changed", update_x_ticks)
                update_x_ticks(ax_top)
            if ts_mic is None:
                ax_bottom.callbacks.connect("xlim_changed", update_x_ticks)
                update_x_ticks(ax_bottom)

            ax_bottom.set_xlabel("Time [s]")

    plt.show()


if __name__ == "__main__":
    main()
