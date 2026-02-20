# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
General Visualization Utils
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import re


def plot_emg_color_by_label(
    emg_df,
    fs,
    use_filtered=True,
    channel_prefix="Ch_",
    filt_suffix="_filt",
    label_col="Label_str",
    figsize=(18, 10),
    robust_percentiles=(1, 99),
    margin_frac=0.5,
    max_legend_cols=8,
    legend_y=1.02,  # increase if you want more spacing above plots
    channels=None,  # optional explicit list of channel base names like ["Ch_0", ...]
    title=None,
    save_path=None,
):
    """
    Plot N channels stacked, with line color-coded by label_col (string labels).
    Agnostic to number of channels.

    - If channels=None: auto-detect columns like Ch_<int> (and use filtered if requested)
    - If channels provided: uses those bases (e.g., "Ch_0") and picks filtered if present
    """

    # --- choose channels ---
    if channels is None:
        # detect bases: Ch_<num> existing in df (raw)
        bases = []
        pat = re.compile(rf"^{re.escape(channel_prefix)}(\d+)$")
        for c in emg_df.columns:
            m = pat.match(c)
            if m:
                bases.append((int(m.group(1)), c))
        bases = [c for _, c in sorted(bases, key=lambda x: x[0])]
        if not bases:
            raise ValueError(f"No channel columns found matching pattern '{channel_prefix}<int>'")
    else:
        bases = channels

    # map bases -> plotted columns (prefer filtered if requested and exists)
    ch_cols = []
    for base in bases:
        filt_col = f"{base}{filt_suffix}"
        ch_cols.append(filt_col if use_filtered and filt_col in emg_df.columns else base)

    n_ch = len(ch_cols)

    # --- time axis ---
    n = len(emg_df)
    t = np.arange(n) / fs

    # --- labels and segment boundaries ---
    labels = emg_df[label_col].astype(str).to_numpy()
    change_idx = np.where(labels[1:] != labels[:-1])[0] + 1
    bounds = np.concatenate(([0], change_idx, [n]))

    seg_labels = [labels[bounds[k]] for k in range(len(bounds) - 1)]
    uniq_labels = list(dict.fromkeys(seg_labels))  # stable order

    # --- assign colors (categorical) ---
    cmap = plt.get_cmap("tab20")
    colors = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq_labels)}
    colors["rest"] = (0.8, 0.8, 0.8)  # force rest light grey (even if absent it's harmless)

    # --- figure layout: N subplots ---
    fig, axes = plt.subplots(n_ch, 1, sharex=True, figsize=figsize, constrained_layout=True)
    if n_ch == 1:
        axes = [axes]

    # --- plot channels ---
    p_lo, p_hi = robust_percentiles

    for ci, ax in enumerate(axes):
        y = emg_df[ch_cols[ci]].to_numpy()

        segments = []
        seg_colors = []
        for k in range(len(bounds) - 1):
            s, e = bounds[k], bounds[k + 1]
            if e - s < 2:
                continue
            pts = np.column_stack([t[s:e], y[s:e]])
            segments.append(pts)
            seg_colors.append(colors.get(labels[s], (0.3, 0.3, 0.3)))

        lc = LineCollection(segments, colors=seg_colors, linewidths=0.9)
        ax.add_collection(lc)

        # robust y scaling
        y_low, y_high = np.percentile(y, [p_lo, p_hi])
        if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
            # fallback if signal is constant or bad
            y_low, y_high = np.nanmin(y), np.nanmax(y)
            if not np.isfinite(y_low) or y_low == y_high:
                y_low, y_high = -1.0, 1.0

        margin = margin_frac * (y_high - y_low)
        ax.set_ylim(y_low - margin, y_high + margin)

        ax.set_xlim(t[0], t[-1])
        ax.grid(True, alpha=0.25)

        # label: use base name like Ch_0 -> show Ch1, etc.
        base = bases[ci]
        idx_match = re.match(rf"^{re.escape(channel_prefix)}(\d+)$", base)
        ch_name = f"Ch{int(idx_match.group(1)) + 1}" if idx_match else base
        ax.set_ylabel(ch_name, rotation=0, labelpad=25, va="center")

    axes[-1].set_xlabel("Time [s]")

    # --- shared legend (outside, top-center) ---
    handles = [Line2D([0], [0], color=colors[lab], lw=3, label=lab) for lab in uniq_labels]
    ncol = min(len(handles), max_legend_cols)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        frameon=True,
        bbox_to_anchor=(0.5, legend_y),
    )

    if title:
        plt.suptitle(title)

    # full-screen (best effort)
    try:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    except Exception:
        pass
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_label_waveform(emg_df, use_label_str=True):
    """
    Plot Label_int as a square wave, shading each contiguous label segment.

    Parameters
    ----------
    emg_df : pd.DataFrame
        Must contain columns: Label_int and Label_str

    use_label_str : bool
        If True, legend shows Label_str.
        If False, legend shows Label_int.
    """

    labels = emg_df["Label_int"].to_numpy()
    label_names = emg_df["Label_str"].to_numpy()

    n = len(labels)

    plt.figure(figsize=(20, 6))

    # ---- Find label transitions ----
    start = 0
    curr_label = labels[0]

    legend_entries = {}

    for i in range(1, n):

        # Transition detected
        if labels[i] != curr_label:

            end = i

            # Shade region
            plt.axvspan(start, end, alpha=0.3)

            # Store legend label
            if use_label_str:
                legend_entries[curr_label] = label_names[start]
            else:
                legend_entries[curr_label] = str(curr_label)

            # Start new segment
            start = i
            curr_label = labels[i]

    # Shade last segment
    plt.axvspan(start, n, alpha=0.3)
    legend_entries[curr_label] = label_names[start] if use_label_str else str(curr_label)

    # ---- Plot square wave ----
    plt.step(np.arange(n), labels, where="post", linewidth=2)

    # ---- Build legend ----
    patches = [Patch(label=f"{k} → {v}") for k, v in legend_entries.items()]

    plt.legend(handles=patches, title="Labels", loc="upper right")

    plt.title("Label Waveform (Square Wave with Colored Segments)")
    plt.xlabel("Sample Index")
    plt.ylabel("Label_int")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":

    # Adjust here the path
    # Adjust here with your path
    h5_file = r"\..\data_raw_and_filt\S01\vocalized\sess_1_batch_1.h5"
    df = pd.read_hdf(h5_file)

    plot_emg_color_by_label(df, fs=500, use_filtered=True)
