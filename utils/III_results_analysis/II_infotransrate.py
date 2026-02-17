"""
itr_window_sweep_analysis.py

Analyze the "Information Transfer Rate" (ITR) ablation study over window size.

This script assumes you trained *inter-session* models for multiple window sizes
(e.g., a sweep like 400 ms, 800 ms, ..., 2000 ms) and saved the runs under a
dedicated experiment folder (e.g., `models/inter_session_win_sweep/`).

What it does
------------
1) Loads all result summaries for a sweep experiment using `load_all_results(...)`
2) Computes ITR per CV fold from the balanced accuracies stored in each run
3) Aggregates ITR statistics (mean/std across folds) per subject and window size
4) Plots, for each condition (silent/vocalized):
   - Balanced accuracy (%) vs window size (left y-axis)
   - ITR (bit/min) vs window size (right y-axis)
   for each subject plus an "Average" block (mean across subjects)

Assumptions / prerequisites
---------------------------
- You ran the training script for multiple window sizes:
  `offline_experiments/intersession_models.py`
- The sweep results are saved under:
  <models_dire>/<dire_for_sweep_experiments>/
- `load_all_results(...)` returns a DataFrame with (at least) these columns:
  - subject
  - condition
  - model_name
  - win_size_ms
  - balanced_acc_vals  (list/array of CV fold accuracies, in [0,1])
  - balanced_acc_mean / balanced_acc_std (optional; used for plotting if present)
"""



import os
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator

# Project-level imports
project_root = Path().resolve()
ARTIFACTS_DIR = Path(os.environ.get('SILENTWEAR_ARTIFACTS_DIR', project_root / 'artifacts'))
.parent.parent
sys.path.insert(0, str(project_root))
from utils.III_results_analysis.utils import load_all_results



############## USER EDITABLE CONFIG ####################
# Paths
# - models_dire: main folder containing experiment subfolders with saved runs
# - dire_for_sweep_experiments: subfolder name for the window-size sweep experiment
# - fig_save_folder: where figures should be saved (if enabled)
models_dire = ARTIFACTS_DIR
dire_for_sweep_experiments = "inter_session_win_sweep"
fig_save_folder = ARTIFACTS_DIR / 'figures'

# Filters
subjects = ["S01", "S02", "S03", "S04"]
conditions = ["silent", "vocalized"]

# Model architecture to analyze (must match `model_name` in the results table)
architecture_to_consider = "speechnet_padded"
############## USER EDITABLE CONFIG END #################


######## Utils #################

def _compute_itr(M=9,T=1400,P=0.8):
    """
    Compute Information Transfer Rate (ITR) in bit/min.

    Parameters
    ----------
    M : int
        Number of classes / targets.
    T : float
        Decision window length in seconds.
    P : float
        Classification accuracy in [0, 1].

    Returns
    -------
    float
        ITR in bit/min.
    """

    a = np.log2(M)
    b = P*np.log2(P)
    c = np.log2((1-P)/(M-1))
    c = (1-P)*c

    ITR = 60*(a+b+c)/T

    return ITR




def _plot_subjects_plus_average_single_box(df_condition, subjects, windows, title,
                                          acc_col_mean="balanced_acc_mean",
                                          acc_col_std="balanced_acc_std",
                                          itr_col_mean="ITRs_means",
                                          itr_col_std="ITRs_stds", 
                                          save_path = None):
    """
    Plot accuracy and ITR vs window size for multiple subjects + an across-subject average.

    The function creates a single wide panel divided into blocks:
    [S01 | S02 | ... | Average], each block containing the same set of window sizes.

    Expected columns in `df_condition`:
    - subject, win_size_ms
    - balanced_acc_mean, balanced_acc_std
    - ITRs_means, ITRs_stds
    """
        
    subjects = list(subjects)
    windows = np.sort(np.asarray(windows))
    nW = len(windows)
    pos_map = {w: j for j, w in enumerate(windows)}

    # ---------- build "All subjects" aggregate (mean across subjects + std across subjects) ----------
    g = df_condition.groupby("win_size_ms")
    agg = g.agg(
        acc_mean=(acc_col_mean, "mean"),
        itr_mean=(itr_col_mean, "mean"),
    ).reset_index()

    # std across subjects of the per-subject MEANS 
    acc_std_across = g[acc_col_mean].apply(lambda v: np.std(v.values)).reset_index(name="acc_std")
    itr_std_across = g[itr_col_mean].apply(lambda v: np.std(v.values)).reset_index(name="itr_std")
    agg = agg.merge(acc_std_across, on="win_size_ms").merge(itr_std_across, on="win_size_ms")

    # order by windows
    agg = agg.set_index("win_size_ms").reindex(windows).reset_index()

    # ---------- plotting setup (one box) ----------
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
    })

    blocks = subjects + ["Average"]
    n_blocks = len(blocks)

    gap = 1     # spacing between blocks (x-index units)
    block = nW + gap

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.6))           #7.2, 2.6
    ax2 = ax.twinx()


    # very light grid (single grid only)
    ax.grid(True, which="major", linewidth=0.35, alpha=0.20)
    ax2.grid(False)

    # ticks/labels across all blocks
    # xticks, xticklabels, centers = [], [], []
    centers = []
    for bi, name in enumerate(blocks):
        start = bi * block

        # alternating band background
        face = "#f4f4f4" if (bi % 2 == 1) else "#ffffff"
        ax.axvspan(start - 0.5, start + nW - 0.5, color=face, zorder=0)

        # separator line between blocks
        # if bi > 0:
        #     ax.axvline(start - 0.5, linewidth=0.6, alpha=0.35)

        # choose data (subject or aggregate)
        if name != "Average":
            df_s = df_condition[df_condition["subject"] == name].sort_values("win_size_ms")

            xvals = df_s["win_size_ms"].to_numpy()
            order = np.argsort(xvals)
            xvals = xvals[order]

            acc = df_s[acc_col_mean].to_numpy()[order] * 100.0
            acc_std = df_s[acc_col_std].to_numpy()[order] * 100.0

            itr = df_s[itr_col_mean].to_numpy()[order]
            itr_std = df_s[itr_col_std].to_numpy()[order]

        else:
            xvals = agg["win_size_ms"].to_numpy()
            acc = agg["acc_mean"].to_numpy() * 100.0
            acc_std = agg["acc_std"].to_numpy() * 100.0
            itr = agg["itr_mean"].to_numpy()
            itr_std = agg["itr_std"].to_numpy()

        # map each win_size to block-local x position
        x_idx = np.array([pos_map.get(w, np.nan) for w in xvals], dtype=float)
        valid = ~np.isnan(x_idx)
        x_plot = start + x_idx[valid]

        # plot accuracy (left)
        ax.errorbar(
            x_plot, acc[valid], yerr=acc_std[valid],
            fmt="o-", capsize=2, markersize=3,
            color="blue", alpha=0.95, linewidth=0.5,
        )

        # plot ITR (right)
        ax2.errorbar(
            x_plot, itr[valid], yerr=itr_std[valid],
            fmt="o-", capsize=2, markersize=3,
            color="red", alpha=0.85, linewidth=0.5
        )

        # # per-block repeated xticks
        # for j, w in enumerate(windows):
        #     xticks.append(start + j)
        #     xticklabels.append(str(int(w)))

        centers.append(start + (nW - 1) / 2)


    label_windows = [400, 800, 1200, 1600, 2000]  # or pick what you want
    grid_ticsk = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]  # or pick what you want
    label_windows = [w for w in windows if w in label_windows]  # keep only existing

    major_xticks, major_xlabels = [], []
    minor_xticks = []

    for bi, name in enumerate(blocks):
        start = bi * block

        # background band
        ax.axvspan(start - 0.5, start + nW - 0.5,
                color=("#f4f4f4" if bi % 2 else "#ffffff"), zorder=0)

        # separator at the boundary (no gap now)
        # if bi > 0:
        #     ax.axvline(start - 0.5, linewidth=0.6, alpha=0.35)

        # minor ticks at every window
        for j, w in enumerate(windows):
            minor_xticks.append(start + j)

        # major ticks only for selected window labels
        for w in label_windows:
            j = np.where(windows == w)[0][0]
            major_xticks.append(start + j)
            major_xlabels.append(str(int(w)))


    # X AXIS SETTINGS
    ax.xaxis.set_major_locator(FixedLocator(major_xticks))
    ax.xaxis.set_minor_locator(FixedLocator(minor_xticks))
    ax.set_xticklabels(major_xlabels, rotation=90, ha="center")
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax.tick_params(axis="x", which="major", length=3, width=0.7)
    ax.tick_params(axis="x", which="major", length=3, width=0.7)
    ax.set_xlim(-0.5, (n_blocks - 1) * block + nW - 0.5)
    ax.set_xlabel("Window size [ms]")

    # LEFT Y AXIS (Accuracy)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylabel("Accuracy (%)", color="blue")
    ax.tick_params(axis='y', which='both', colors='blue')   # <- labels + tick marks blue
    ax.spines['left'].set_color('blue')                     # optional (looks consistent)

    # RIGHT Y AXIS (ITR)
    ax2.set_ylim(0, 200)
    ax2.yaxis.set_major_locator(MultipleLocator(25))
    ax2.set_ylabel("ITR (bit/min)", color="red")
    ax2.tick_params(axis='y', which='both', colors='red')   # <- labels + tick marks red
    ax2.spines['right'].set_color('red')                    # optional

    # Make the *overall box* thicker (all spines that exist)
    for a in (ax, ax2):
        for side in ('left', 'right', 'top', 'bottom'):
            if side in a.spines:
                a.spines[side].set_linewidth(0.1)  # increase as you like (e.g., 2.0)

    # If you want the top/bottom spines to be only drawn once (avoid double-draw look):
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    # block titles above (subjects + All)
    y_top = ax.get_ylim()[1]
    for c, name in zip(centers, blocks):
        ax.text(c, y_top - 10.0, str(name), ha="center", va="bottom", fontsize=10)

    
    #ax.set_title(title, y=1.08)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    return fig, ax, ax2



####################à
if __name__=='__main__':
    # Load all results
    all_results = load_all_results(models_dire/dire_for_sweep_experiments, 
                                   subjects_to_consider = subjects, 
                                   conditions_to_consider = conditions)
    
    # Compute ITR per CV fold from balanced accuracies, then summarize mean/std per run
    new_rows = []
    for idx, row in all_results.iterrows():
        newrow = {}
        balanced_accuracies_values = row["balanced_acc_vals"]
        win_size = row["win_size_ms"]/1000
        itrs = []
        for p in balanced_accuracies_values:
            itrs.append(_compute_itr(M=9, T=win_size, P=p))
        
        newrow["ITRs"] = np.array(itrs)
        newrow["ITRs_means"] = np.mean(np.array(itrs))
        newrow["ITRs_stds"] = np.std(np.array(itrs))

        new_rows.append(newrow)
    all_results = pd.concat((all_results, pd.DataFrame(new_rows)), axis=1)
    
    # Keep only results for the selected model architecture (e.g., speechnet_padded)
    archi_res = all_results[all_results["model_name"] == architecture_to_consider]
    
    # Keep a window subset
    archi_res = archi_res[archi_res["win_size_ms"]<=1400]
    for condition in conditions:
        df_condition = archi_res[archi_res["condition"] == condition]
        windows = np.sort(df_condition["win_size_ms"].unique())

        fig, ax, ax2 = _plot_subjects_plus_average_single_box(
            df_condition=df_condition,
            subjects=subjects,          # (4 subjects) -> plus "All" = 5 blocks
            windows=windows,
            title=f"Condition: {condition}",
            #save_path=None,
            save_path=fig_save_folder/f"itr_{condition}_cut.pdf"
        )
    

        # Display some statistics
        print(f"=======Summary for Condition: {condition} ===========")


        ITRs_per_window = []
        accs_per_window = []
        window_size = []

        unique_subjs = archi_res["subject"].unique()
        unique_windows = archi_res["win_size_ms"].unique()
        
        for window in unique_windows:
            win_itr = []
            win_acc = []

            win_metrics = df_condition[df_condition["win_size_ms"] == window]

            for subj in unique_subjs:
                subj_metrics = win_metrics[win_metrics["subject"] == subj]

                assert len(subj_metrics) == 1, (
                    f"Expected exactly 1 row for subject={subj}, "
                    f"window={window}, got {len(subj_metrics)}"
                )

                win_itr.append(subj_metrics["ITRs_means"].iloc[0])
                win_acc.append(subj_metrics["balanced_acc_mean"].iloc[0])

            ITRs_per_window.append(np.mean(win_itr))
            accs_per_window.append(np.mean(win_acc))
            window_size.append(window)
        
        ITRs_per_window = np.array(ITRs_per_window)
        accs_per_window = np.array(accs_per_window)
        window_size = np.array(window_size)

        # print("Windows")
        # print(window_size)
        # print("Accuracy")
        # print(accs_per_window)
        # print("ITrs")
        # print(ITRs_per_window)
        

        max_acc_loc = np.where(accs_per_window==np.max(accs_per_window))[0]
        max_acc = accs_per_window[max_acc_loc]
        print("Max accuracy value:", max_acc, "Window size of:", window_size[max_acc_loc], "ms")

        max_itr_loc = np.where(ITRs_per_window==np.max(ITRs_per_window))[0]
        max_itr = ITRs_per_window[max_itr_loc]
        print("Max ITR value:", max_itr, "Window size of:", window_size[max_itr_loc], "ms")


        # plateaus at 1000 ms
        one_sec_loc = np.where(window_size == 800)
        one_sec_acc = accs_per_window[one_sec_loc]
        one_sec_itr = ITRs_per_window[one_sec_loc]

        acc_red = ((max_acc - one_sec_acc) / max_acc)*100
        itr_red = ((max_itr - one_sec_itr)/ max_itr) *100
        print("Accuracy value:", one_sec_acc, "ITR:", one_sec_itr , "Window size of:", window_size[one_sec_loc], "ms")

        print("ACC reduction:", acc_red)
        print("ITR reduction", itr_red)
