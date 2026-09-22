# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Script to visualize windowed features for each text unit and condition 
in a grid layout, with consistent scaling across conditions.
"""


import sys
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "extras"))

from utils.I_data_preparation.experimental_config import FS, get_active_labels
from extras.taper_fig import plot_stacked_channels_in_cell
from extras.fig_config import channel_colors, neckband_ch_order


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------


wins_root = Path("path/to/wins_root")  # Update this to the actual path where the WIN_{window_ms} folders are located
out_dir_base = Path("./windowing_check/figures") # Base directory for saving plots
subject_ids = ["S01"]                  # Update this to the actual subject ID you want to process (e.g., "S01", "S02", etc.)
window_ms = 1400                       # Update this to the desired window size in milliseconds (e.g., 400, 800, etc.) 
conditions = None                      # Set to None to include all conditions, or specify a list of conditions to include (e.g., ["vocalized", "silent"])  
target_session = 1                     # Set to None to include all sessions, or specify a session number to filter (e.g., 1, 2, etc.)  
target_batch = 1                       # Set to None to include all batches, or specify a batch number to filter (e.g., 1, 2, etc.)  
label_mode = "word"                    # "word" | "sentence"
process_all = False                    # Set to True to ignore target_session and target_batch and process all available data for the subject and window size
output_ext = "png"
exclude_words = {"rest"}
amp_ref = None                         # Set to None to derive the vertical scale from the plotted data, or fix it with --amp_ref (see parse_amp_ref)
amp_percentile = 92                    # Percentile of |x| the derived vertical scale is built from (see build_spacing_map)

ordered_cols = [f"Ch_{i}_filt" for i in neckband_ch_order]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _concat_windows(series):
    """Concatenates a pandas Series of arrays (e.g., windowed features) into a single 1D array"""
    arrays = [np.asarray(v) for v in series if v is not None]
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays)


def parse_amp_ref(tokens):
    """
    Parses the --amp_ref tokens into {"default": float|None, "per_cond": {cond: float}}.
    """
    if not tokens:
        return None
    spec = {"default": None, "per_cond": {}}
    for token in tokens:
        if "=" in token:
            cond, _, value = token.partition("=")
            spec["per_cond"][cond.strip()] = float(value)
        else:
            spec["default"] = float(token)
    return spec


def build_spacing_map(df_batch, conditions_list, ch_cols, amp_ref_spec=None, percentile=amp_percentile, spacing_factor=10, margin_factor=2):
    """
    Builds the per-condition vertical scale used by every cell of the figure.
    """
    global_spacing_map = {}

    for cond in conditions_list:
        fixed = None
        if amp_ref_spec is not None:
            fixed = amp_ref_spec["per_cond"].get(cond, amp_ref_spec["default"])

        if fixed is not None:
            amp_ref_cond = float(fixed)
            print(f"  amp_ref[{cond}] = {amp_ref_cond:.4f} (fixed)")
        else:
            df_cond = df_batch[df_batch["condition"] == cond]
            all_signals = []
            for col in ch_cols:
                flat_signal = _concat_windows(df_cond[col])
                if len(flat_signal) > 0:
                    all_signals.append(flat_signal)

            if all_signals:
                amp_ref_cond = float(np.nanpercentile(np.abs(np.concatenate(all_signals)), percentile))
            else:
                amp_ref_cond = 0.0
            print(f"  amp_ref[{cond}] = {amp_ref_cond:.4f} (derived, p{percentile:g}; pass --amp_ref {cond}={amp_ref_cond:.4f} to reuse it)")

        spacing = spacing_factor * amp_ref_cond if amp_ref_cond > 0 else 1.0
        n_ch = len(ch_cols)
        ylims = (-margin_factor * spacing, (n_ch - 1) * spacing + margin_factor * spacing)

        global_spacing_map[cond] = {"spacing": spacing, "ylims": ylims}

    return global_spacing_map


def find_wins_h5(wins_root_dir, subject, win_ms, conditions_list=None, target_sessions=None, target_batches=None):
    """Finds all .h5 files for a given subject and window size, optionally filtering by conditions, multiple sessions, and multiple batches."""
    base_dir = Path(wins_root_dir) / subject
    if not base_dir.exists():
        print(f"Warning: Wins root does not exist: {base_dir}")
        return [], conditions_list or []

    if conditions_list is None or len(conditions_list) == 0:
        conditions_list = sorted(
            [p.name for p in base_dir.iterdir() if p.is_dir() and (p / f"WIN_{win_ms}").exists()]
        )
        if "vocalized" in conditions_list and "silent" in conditions_list:
            ordered = ["vocalized", "silent"]
            conditions_list = ordered + [col for col in conditions_list if col not in ordered]

    if len(conditions_list) == 0:
        print(f"Warning: No condition folders found with WIN_{win_ms} under {base_dir}")
        return [], []

    h5_files = []
    for cond in conditions_list:
        win_dir = base_dir / cond / f"WIN_{win_ms}"
        if win_dir.exists():
            h5_files.extend(sorted(win_dir.glob("*.h5")))

    if target_sessions is not None and len(target_sessions) > 0:
        h5_files = [f for f in h5_files if any(f.name.find(f"sess_{ts}_") != -1 for ts in target_sessions)]
        
    if target_batches is not None and len(target_batches) > 0:
        h5_files = [f for f in h5_files if any(f.name.find(f"batch_{tb}.") != -1 for tb in target_batches)]

    return h5_files, conditions_list


def load_wins_df(h5_files, key="wins_feats"):
    """
    Loads windowed features from a list of .h5 files into a single DataFrame, adding source file and condition metadata.
    The function concatenates all the DataFrames from the .h5 files into one large DataFrame.
    """
    frames = []
    for file in h5_files:
        try:
            df = pd.read_hdf(file, key=key)
            df["source_file"] = file.name
            df["condition"] = file.parent.parent.name
            frames.append(df)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def _normalize_axes(axs, n_rows, n_cols):
    """Ensures that axs is a 2D numpy array of shape (n_rows, n_cols) even if n_rows or n_cols is 1."""
    if isinstance(axs, np.ndarray):
        if axs.ndim == 2:
            return axs
        if axs.ndim == 1:
            if n_rows == 1:
                return axs[np.newaxis, :]
            return axs[:, np.newaxis]
    return np.array([[axs]])


def plot_windows_per_text(
    df,
    text,
    conditions_list,
    ch_cols,
    global_spacing_map,
    save_path,
    alpha=1.0,
    lw=1.0,
    grid_alpha=0.15,
    alt_bg_alpha=0.04,
    row_fontsize=24,
    title_fontsize=28,
    time_xlim_s=None,
    time_xlabel="Time [s]",
    outer_box=True,
    outer_box_lw=2.0,
    outer_box_color="white",
    text_color="black"
):
    """Plots windowed features for a specific text across different conditions in a grid layout."""
    
    counts = [len(df[(df["condition"] == cond) & (df["Label_str"] == text)]) for cond in conditions_list]
    max_reps = max(counts) if len(counts) > 0 else 0
    
    if max_reps == 0:
        return None

    fig_width = max(10, 3.2 * max_reps)
    fig, axs = plt.subplots(len(conditions_list), max_reps, figsize=(fig_width, 10), sharex=True, sharey="row")
    axs = _normalize_axes(axs, len(conditions_list), max_reps)

    L, R, B, T = 0.06, 0.98, 0.1, 0.88
    
    fig.subplots_adjust(left=L, right=R, bottom=B, top=T, wspace=0.20, hspace=0.05)
    
    fig.suptitle(text.upper(), fontsize=title_fontsize, y=0.98, color=text_color)

    for row_idx, cond in enumerate(conditions_list):
        df_cond = df[(df["condition"] == cond) & (df["Label_str"] == text)].copy()
        
        if df_cond.empty:
            for col_idx in range(max_reps):
                axs[row_idx, col_idx].set_visible(False)
            continue

        sort_cols = [col for col in ["session_id", "batch_id", "start_idx"] if col in df_cond.columns]
        if sort_cols:
            df_cond = df_cond.sort_values(sort_cols).reset_index(drop=True)

        spacing = global_spacing_map[cond]["spacing"]
        ylims = global_spacing_map[cond]["ylims"]

        for col_idx in range(max_reps):
            ax = axs[row_idx, col_idx]
            if col_idx % 2 == 0:
                # Black with low alpha creates a light gray background column on white
                ax.set_facecolor((0, 0, 0, alt_bg_alpha))

            if col_idx < len(df_cond):
                row = df_cond.iloc[col_idx]
                seg_df = pd.DataFrame({col: row[col] for col in ch_cols})
                
                plot_stacked_channels_in_cell(
                    ax=ax, df_seg=seg_df, ch_cols=ch_cols, fs=FS,
                    spacing=spacing, ylims=ylims, alpha=alpha, lw=lw,
                    channel_colors=channel_colors,
                )
                
                ax.grid(True, alpha=grid_alpha)
                n_samp = len(np.asarray(row[ch_cols[0]]))
                x_max = time_xlim_s if time_xlim_s is not None else n_samp / FS
                ax.set_xlim(0, x_max)
                
                if row_idx == len(conditions_list) - 1:
                    ax.tick_params(axis="x", bottom=True, labelbottom=True, labelsize=20, colors=text_color)
                    if col_idx == int(max_reps/2):
                        ax.set_xlabel(time_xlabel, fontsize=20, color=text_color)
                else:
                    ax.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                ax.set_visible(False)

        # Remove Y labels and ticks for all columns (including the first one)
        for col_idx in range(max_reps):
            for sp in axs[row_idx, col_idx].spines.values():
                sp.set_visible(False)
            axs[row_idx, col_idx].tick_params(axis="y", left=False, labelleft=False)

        bb = axs[row_idx, 0].get_position()
        cy = 0.5 * (bb.y0 + bb.y1)
        fig.text(L - 0.03, cy, cond.capitalize(), ha="left", va="center", fontsize=row_fontsize, rotation=90, color=text_color)

    if outer_box:
        rect = patches.Rectangle(
            (L - 0.002, B - 0.002), (R - L) + 0.004, (T - B) + 0.004,
            transform=fig.transFigure, fill=False, linewidth=outer_box_lw,
            edgecolor=outer_box_color, zorder=1000, clip_on=False,
        )
        fig.add_artist(rect)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, transparent=False, facecolor="white")
    print(f"Saved plot for text {text.upper()}: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to visualize windowed features.")
    
    parser.add_argument("--wins_root", type=str, default=str(wins_root), help="Path to the wins root directory.")
    parser.add_argument("--out_dir", type=str, default=str(out_dir_base), help="Base directory to save the output figures.")
    parser.add_argument("--subject_ids", nargs="+", default=subject_ids, help="List of subject IDs to process.")
    parser.add_argument("--window_ms", nargs="+", type=int, default=[window_ms], help="List of desired window sizes in milliseconds.")
    parser.add_argument("--conditions", nargs="*", default=conditions, help="List of conditions. Leave empty for all.")
    
    default_sessions = [target_session] if target_session is not None else None
    default_batches = [target_batch] if target_batch is not None else None
    
    parser.add_argument("--target_sessions", nargs="*", type=int, default=default_sessions, help="List of session numbers to filter.")
    parser.add_argument("--target_batches", nargs="*", type=int, default=default_batches, help="List of batch numbers to filter.")
    parser.add_argument("--label_mode", type=str, default=label_mode, choices=["word", "sentence"], help="Label mode: 'word' or 'sentence'.")
    parser.add_argument("--output_ext", type=str, default=output_ext, help="Extension of the saved figure.")
    parser.add_argument("--process_all", action="store_true", default=process_all, help="Ignore targets and process all available data.")
    parser.add_argument(
        "--amp_ref", nargs="*", default=amp_ref,
        help="Fix the vertical scale instead of deriving it from the plotted windows. "
             "Either one number for every condition, or 'condition=number' pairs "
             "(e.g. --amp_ref vocalized=362.0876 silent=366.2839). The number is the "
             "percentile of |x| that the channel spacing is built from, so passing "
             "the value another run printed makes the two figures share one scale.",
    )
    parser.add_argument(
        "--amp_percentile", type=float, default=amp_percentile,
        help="Percentile of |x| the derived vertical scale is built from (default: %(default)g). "
             "Raise it to draw the traces smaller. Ignored for conditions pinned with --amp_ref.",
    )

    args = parser.parse_args()

    amp_ref_spec = parse_amp_ref(args.amp_ref)

    if args.process_all:
        sessions_to_process = [None]
        batches_to_process = [None]
        args.conditions = None
    else:
        sessions_to_process = args.target_sessions if args.target_sessions else [None]
        batches_to_process = args.target_batches if args.target_batches else [None]

    for current_sub in args.subject_ids:
        for current_win in args.window_ms:
            for current_sess in sessions_to_process:
                for current_batch in batches_to_process:
                    
                    sess_str = "all" if current_sess is None else str(current_sess)
                    batch_str = "all" if current_batch is None else str(current_batch)  
                    
                    print(f"Loading data | Subject: {current_sub} | Window: {current_win}ms | Session: {sess_str} | Batch: {batch_str}")
                    
                    h5_files, conditions_list = find_wins_h5(
                        wins_root_dir=args.wins_root, 
                        subject=current_sub, 
                        win_ms=current_win, 
                        conditions_list=args.conditions,
                        target_sessions=[current_sess] if current_sess is not None else None,
                        target_batches=[current_batch] if current_batch is not None else None
                    )
                    
                    if not h5_files:
                        print(f"No h5 files found for {current_sub} (Sess: {sess_str}, Batch: {batch_str}). Skipping...")
                        continue
                        
                    df = load_wins_df(h5_files, key="wins_feats")

                    if df.empty:
                        print(f"No data loaded for {current_sub} (Sess: {sess_str}, Batch: {batch_str}). Skipping...")
                        continue

                    if "session_id" not in df.columns:
                        df["session_id"] = df["source_file"].apply(
                            lambda x: int(m.group(1)) if (m := re.search(r"sess_(\d+)", x)) else None
                        )
                    if "batch_id" not in df.columns:
                        df["batch_id"] = df["source_file"].apply(
                            lambda x: int(m.group(1)) if (m := re.search(r"batch_(\d+)", x)) else None
                        )

                    # Finding unique sessions and batches in the DataFrame to iterate over
                    unique_sessions = df["session_id"].dropna().unique()
                    if len(unique_sessions) == 0:
                        unique_sessions = [None]
                    
                    for act_sess in sorted(unique_sessions, key=lambda x: (x is None, x)):
                        df_sess = df[df["session_id"] == act_sess] if act_sess is not None else df
                        
                        unique_batches = df_sess["batch_id"].dropna().unique()
                        if len(unique_batches) == 0:
                            unique_batches = [None]
                            
                        for act_batch in sorted(unique_batches, key=lambda x: (x is None, x)):
                            df_batch = df_sess[df_sess["batch_id"] == act_batch] if act_batch is not None else df_sess
                            
                            if df_batch.empty:
                                continue
                            
                            actual_sess_str = str(int(act_sess)) if act_sess is not None else "unknown"
                            actual_batch_str = str(int(act_batch)) if act_batch is not None else "unknown"

                            print(f"Generating plots for Session: {actual_sess_str} | Batch: {actual_batch_str}")

                            ch_cols = [col for col in ordered_cols if col in df_batch.columns]
                            df_scale = df_batch[~df_batch["Label_str"].isin(exclude_words)]

                            if df_scale.empty:
                                print(f"Only excluded labels for Session: {actual_sess_str} | Batch: {actual_batch_str}. Skipping...")
                                continue

                            global_spacing_map = build_spacing_map(
                                df_batch=df_scale,
                                conditions_list=conditions_list,
                                ch_cols=ch_cols,
                                amp_ref_spec=amp_ref_spec,
                                percentile=args.amp_percentile,
                            )

                            active_labels = get_active_labels(args.label_mode)
                            ordered_texts = [active_labels[i] for i in sorted(active_labels.keys())]
                            texts = [t for t in ordered_texts if t in df_batch["Label_str"].unique() and t not in exclude_words]

                            current_save_dir = Path(args.out_dir) / current_sub / f"WIN_{current_win}" / f"sess_{actual_sess_str}"
                            current_save_dir.mkdir(parents=True, exist_ok=True)

                            for text in texts:
                                safe_text = text.replace(" ", "_")
                                out_path = current_save_dir / f"{safe_text}_sess_{actual_sess_str}_batch_{actual_batch_str}.{args.output_ext}"
                                
                                plot_windows_per_text(
                                    df=df_batch,
                                    text=text,
                                    conditions_list=conditions_list,
                                    ch_cols=ch_cols,
                                    global_spacing_map=global_spacing_map,
                                    save_path=out_path,
                                    time_xlim_s=current_win / 1000.0,
                                )