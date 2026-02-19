# # SPDX-FileCopyrightText: 2026 ETH Zurich
# # SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(PROJECT_ROOT))
from utils.I_data_preparation.experimental_config import FS


###### PLOT IN THE PAPER: subject 4
subject_to_consider = "S04"
main_data_dire_folder = Path("/scratch2/gspacone/DATA_DIR_SILENT")
save_fig_path = Path("/home/gspacone/Desktop/Silentwear/artifacts/figures")


def find_all_processed_h5(main_data_dire_folder, subject):
    "Function to return all h5 files contained in the processed directory"
    # Find all .bio files under the subject raw folder
    h5_files = sorted((main_data_dire_folder / "data_raw_and_filt" / subject).rglob("*.h5"))
    # print("Found files for current user:")
    # print(h5_files)
    return h5_files


def plot_words_grid_all_channels(
    h5_files,
    unique_words,
    word_title,
    check_word_bounderies,
    FS,
    cols=None,
    key="emg",
    conditions=("vocalized", "silent"),
    example_idx=4,
    filt_suffix="_filt",
    figsize=(50, 20),
    save_path=None,
    # layout (increase left margin so labels fit)
    L=0.16,
    R=0.995,
    B=0.05,
    T=0.8,
    wspace=0.03,
    hspace=0.0,
    header_h=0.05,
    # montage scaling
    spacing_factor=10,
    margin_factor=2,
    alpha=1.0,
    lw=1.0,
    # styling
    grid_alpha=0.15,
    alt_bg_alpha=0.04,
    channel_colors=None,
    # ticks/labels
    show_time_ticks_bottom_only=True,
    show_time_xlabel_bottom_only=True,
    ylabels_fontsize=14,
    ylabels_pad=8,
    short_channel_labels=True,
    # header text
    word_fontsize=30,
    row_fontsize=30,
    # outer box
    outer_box=True,
    outer_box_lw=2.0,
    outer_box_color="white",
):
    """
    2x8 grid: one word example per column, two rows = vocalized/silent.
    Each cell plots ALL channels stacked.

    KEY FIX:
      Because sharey=True, DO NOT clear y-ticks on other columns.
      Hide them with tick_params(labelleft=False) instead.
    """

    def _get_sorted_channel_cols(df, filt_suffix="_filt"):
        cols = [c for c in df.columns if c.startswith("Ch_") and c.endswith(filt_suffix)]

        def ch_num(c):
            m = re.search(r"Ch_(\d+)", c)
            return int(m.group(1)) if m else 10**9

        return sorted(cols, key=ch_num)

    def _short_labels(cols):
        out = []
        for c in cols:
            m = re.search(r"Ch_(\d+)", c)
            out.append(f"CH{int(m.group(1))}" if m else c)
        return out

    def _plot_stacked_channels_in_cell(ax, df_seg, ch_cols, fs, spacing, ylims, alpha=1.0, lw=1.0):
        if df_seg is None or df_seg.empty:
            ax.set_visible(False)
            return

        n = len(df_seg)
        t = np.arange(n) / fs

        sigs = np.vstack([df_seg[c].to_numpy() for c in ch_cols])
        sigs = sigs - np.median(sigs, axis=1, keepdims=True)

        n_ch = sigs.shape[0]
        offsets = np.arange(n_ch)[::-1] * spacing

        # one color per channel (matplotlib cycle)
        for k in range(n_ch):
            if channel_colors is not None:
                ax.plot(t, sigs[k] + offsets[k], alpha=alpha, linewidth=lw, color=channel_colors[k])
            else:
                ax.plot(t, sigs[k] + offsets[k], alpha=alpha, linewidth=lw)

        ax.set_ylim(*ylims)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(2, 8, figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(left=L, right=R, bottom=B, top=T, wspace=wspace, hspace=hspace)

    offsets_by_row = {}
    ylabels_by_row = {}

    already_plotted = {c: False for c in conditions}

    # ---------------- plotting ----------------
    for h5_file in h5_files:
        for condition in conditions:
            if h5_file.parent.name != condition or already_plotted[condition]:
                continue

            axs_row = 0 if condition == "vocalized" else 1

            emg_df = pd.read_hdf(h5_file, key=key)
            if cols is None:
                ch_cols = _get_sorted_channel_cols(emg_df, filt_suffix=filt_suffix)
            else:
                ch_cols = cols
            if len(ch_cols) == 0:
                raise ValueError(f"No channel columns like 'Ch_*{filt_suffix}' in {h5_file}")

            # global spacing per condition/file
            X = emg_df[ch_cols].to_numpy()
            X = X - np.nanmedian(X, axis=0, keepdims=True)
            amp_ref = np.nanpercentile(np.abs(X), 95)

            spacing = spacing_factor * amp_ref if amp_ref > 0 else 1.0
            n_ch = len(ch_cols)
            ylims = (-margin_factor * spacing, (n_ch - 1) * spacing + margin_factor * spacing)

            offsets_by_row[axs_row] = np.arange(n_ch)[::-1] * spacing
            ylabels_by_row[axs_row] = _short_labels(ch_cols) if short_channel_labels else ch_cols

            axs_word_cnt = 0
            for word_cnt, word in enumerate(unique_words):
                if word == "rest":
                    continue

                ax = axs[axs_row, axs_word_cnt]

                emg_word = emg_df[emg_df["Label_str"] == word]
                idx_word_start, idx_word_stop = check_word_bounderies(emg_word)

                if len(idx_word_start) <= example_idx or len(idx_word_stop) <= example_idx:
                    ax.set_visible(False)
                    axs_word_cnt += 1
                    continue

                seg = emg_word.loc[idx_word_start[example_idx] : idx_word_stop[example_idx]]

                _plot_stacked_channels_in_cell(
                    ax=ax,
                    df_seg=seg,
                    ch_cols=ch_cols,
                    fs=FS,
                    spacing=spacing,
                    ylims=ylims,
                    alpha=alpha,
                    lw=lw,
                )

                ax.grid(True, alpha=grid_alpha)
                ax.set_xlim(0, 1.1)
                axs_word_cnt += 1

            already_plotted[condition] = True

    # ---------------- styling ----------------
    n_rows, n_cols = axs.shape

    # 1) Set y-ticks/labels ONCE per row (shared y-axis!)
    for i in range(n_rows):
        if i in offsets_by_row:
            axs[i, 0].set_yticks(offsets_by_row[i])
            ######## CHANGE HERE TO SHOW NAMES
            axs[i, 0].set_yticklabels(ylabels_by_row[i], fontsize=ylabels_fontsize, color="white")
            # axs[i, 0].set_yticklabels(ylabels_by_row[i], fontsize=ylabels_fontsize, color="black")
            axs[i, 0].tick_params(axis="y", length=0, pad=ylabels_pad, labelleft=True)

    # 2) For other columns: HIDE y labels without clearing ticks (sharey!)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            if not ax.get_visible():
                continue

            for sp in ax.spines.values():
                sp.set_visible(False)

            # x ticks bottom row only
            if show_time_ticks_bottom_only:
                if i == n_rows - 1:
                    ax.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=20)
                    if show_time_xlabel_bottom_only:
                        ax.set_xlabel("Time [s]", fontsize=20)
                else:
                    ax.tick_params(axis="x", bottom=False, labelbottom=False)

            # y label visibility control
            if j != 0:
                ax.tick_params(axis="y", left=False, labelleft=False)  # <-- key fix
            else:
                ax.tick_params(axis="y", left=False)  # keep labels; no tick marks

    # alternating background by column
    for j in range(n_cols):
        if j % 2 == 0:
            for i in range(n_rows):
                if axs[i, j].get_visible():
                    axs[i, j].set_facecolor((0, 0, 0, alt_bg_alpha))

    # outer box
    if outer_box:
        rect = patches.Rectangle(
            (L - 0.002, B - 0.002),
            (R - L) + 0.004,
            (T - B) + 0.004,
            transform=fig.transFigure,
            fill=False,
            linewidth=outer_box_lw,
            edgecolor=outer_box_color,
            zorder=1000,
            clip_on=False,
        )
        fig.add_artist(rect)

    # column titles (words) inside box
    col_centers = [
        0.5 * (axs[0, j].get_position().x0 + axs[0, j].get_position().x1) for j in range(n_cols)
    ]
    col_labels = []
    for word_cnt, w in enumerate(unique_words):
        if w == "rest":
            continue
        col_labels.append(word_title[word_cnt])

    y_words = T - 0.5 * header_h
    for cx, lab in zip(col_centers, col_labels):
        fig.text(cx, y_words, lab, ha="center", va="center", fontsize=word_fontsize)

    # row titles
    row_names = ["Vocalized", "Silent"]
    for i, lab in enumerate(row_names):
        bb = axs[i, 0].get_position()
        cy = 0.5 * (bb.y0 + bb.y1)
        fig.text(L - 0.03, cy, lab, ha="left", va="center", fontsize=row_fontsize, rotation=90)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            bbox_inches="tight",
            pad_inches=0.02,
            transparent=True,
        )
    return fig, axs


def check_word_bounderies(emg_word):
    idx_word = emg_word.index
    # identify start and stops
    idx_prev = idx_word[0]
    idxs_stop = []
    idxs_start = []
    idxs_start.append(idx_word[0])
    for idx_curr in idx_word[1:]:

        if idx_curr != idx_prev + 1:
            # new repetition
            idxs_stop.append(idx_prev)
            idxs_start.append(idx_curr)
        idx_prev = idx_curr
    idxs_stop.append(idx_word[-1])
    idxs_start = np.array(idxs_start)
    idxs_stop = np.array(idxs_stop)

    return idxs_start, idxs_stop


if __name__ == "__main__":

    neckband_ch_order = [0, 1, 2, 5, 3, 4, 7, 6, 8, 15, 9, 14, 10, 13]
    cols = []
    for id in neckband_ch_order:
        cols.append(f"Ch_{id}_filt")

    channel_colors = [
        # Blue pair
        "#1f77b4",  # CH0
        "#6baed6",  # CH1
        # Orange pair
        "#ff7f0e",  # CH2
        "#ffbb78",  # CH3
        # Green pair
        "#2ca02c",  # CH4
        "#98df8a",  # CH5
        # Red pair
        "#d62728",  # CH6
        "#ff9896",  # CH7
        # Purple pair
        "#9467bd",  # CH8
        "#c5b0d5",  # CH9
        # Brown pair
        "#8c564b",  # CH10
        "#c49c94",  # CH11
        # Teal pair
        "#17becf",  # CH12
        "#9edae5",  # CH13
    ]

    unique_words = ["rest", "up", "down", "left", "right", "start", "stop", "forward", "backward"]
    word_title = ["REST", "UP", "DOWN", "LEFT", "RIGHT", "START", "STOP", "FORWARD", "BACKWARD"]

    h5_files = find_all_processed_h5(main_data_dire_folder, subject_to_consider)
    fig, axs = plot_words_grid_all_channels(
        h5_files=h5_files,
        unique_words=unique_words,
        cols=cols,
        example_idx=2,
        word_title=word_title,
        channel_colors=channel_colors,
        check_word_bounderies=check_word_bounderies,
        FS=FS,
        save_path=save_fig_path / "taper.svg",
    )
