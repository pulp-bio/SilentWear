"""
I_global_intersession_analysis.py

Summarize and visualize results from:
- Global experiments
- Inter-session experiments

This script:
1) Loads all saved runs for a given experiment type (global or inter_session)
2) Filters runs by:
   - model architecture (model_name)
   - model variant (model_id)
   - condition (silent / vocalized / ...)
   - subject list
   - window size (sanity check)
3) Writes per-subject and across-subject performance summaries to CSV
4) Optionally plots per-subject confusion matrices (mean ± std across CV folds)

Assumptions / prerequisites
---------------------------
- You have already run the training scripts described in README.md:
  - offline_experiments/global_models.py
  - offline_experiments/intersession_models.py
- Each run directory contains a `cv_summary.csv` file with a column named `confusion_matrix`.
- The results directory is structured so that:
  <model_results_dir>/<experiment_to_analyze>/... contains one folder per run.
  The loader `load_all_results(...)` is responsible for scanning and parsing this structure.

Outputs
-------
- CSV summary saved into `res_save_folder`:
  {model_to_select}_{model_id}_{condition}_win{win_size_ms_to_consider}_{experiment_to_analyze}.csv

- If enabled, confusion matrix figure saved into `fig_save_folder`:
  {model_to_select}_{model_id}_{condition}_win{win_size_ms_to_consider}_{experiment_to_analyze}.svg

How to run
----------
Edit the "USER EDITABLE PART" section, then run:

    python utils/III_results_analysis/I_global_intersession_analysis.py
"""

import os
import pandas as pd
from pathlib import Path
import sys
import json
import ast
import numpy as np
import sys
from pathlib import Path
project_root = Path().resolve()
ARTIFACTS_DIR = Path(os.environ.get('SILENTWEAR_ARTIFACTS_DIR', project_root / 'artifacts'))
.parent.parent
sys.path.insert(0, str(project_root))
from utils.III_results_analysis.utils import *


################## USER EDITABLE PART ##########################################################
# Paths
# - model_results_dire: folder containing saved run folders (metadata + CV summaries)
# - res_save_folder: where numeric CSV summaries are written
# - fig_save_folder: where figures are written
model_results_dire = ARTIFACTS_DIR / 'models'
res_save_folder = ARTIFACTS_DIR / 'tables'
if res_save_folder.exists() == False:
    res_save_folder.mkdir(parents=True)
fig_save_folder = ARTIFACTS_DIR / 'figures'
if fig_save_folder.exists() == False:
    fig_save_folder.mkdir(parents=True)

# Experiment selection:
# - "global": pooled training across sessions with CV defined in config
# - "inter_session": train on two sessions, test on the held-out session [or "inter_session_win_sweep"]
# - other values may exist depending on your training pipeline (e.g., "inter_session_win_sweep")
experiment_to_analyze = "inter_session_win_sweep"

# Filters
subjects_to_consider   = ["S01", "S02", "S03", "S04"]
conditions_to_consider = ["silent", "vocalized"]

# Model selection
# model_to_select: must match `model_name` stored in result metadata
# model_id: choose which variant of the model to report (assumes same model_id exists for all subjects)
model_to_select = "speechnet_padded"     # e.g., speechnet_padded, speechnet_base, random_forest
model_id        = "model_6"

# Sanity check: enforce a single window size in the filtered results
win_size_ms_to_consider = 1400

# Plot settings
plot_confusion_matrix = True
############################################################################################



if __name__=='__main__':
     # Load all runs (subjects x conditions) for the requested experiment
    summary_df = load_all_results(model_results_dire/experiment_to_analyze, subjects_to_consider, conditions_to_consider)

     # Filter by model architecture and model variant
    model_df = summary_df[summary_df["model_name"] == model_to_select]
    model_df = model_df[model_df["model_id"] == model_id]

    # Sanity check: the filtered results must contain exactly one window size
    win_size_unique = model_df["win_size_ms"].unique()
    print(win_size_unique)
    if len(win_size_unique)!=1:
        print("Multiple window sizes found!", win_size_unique)
        sys.exit()
    else:
        if win_size_unique!=win_size_ms_to_consider:
            print("Window size in results dataset is different from what requested")
            print(f"Requested: {win_size_ms_to_consider}")
            sys.exit()
    
    print("\n\n================== MODEL REPORT ==================\n")


    for condition in conditions_to_consider:
        # Filter by condition (silent or vocalized)
        model_condition = model_df[model_df["condition"]==condition]

        # Per-subject summary table
        summary_subjects = model_condition[["subject", "balanced_acc_mean", "balanced_acc_std", "balanced_acc_vals"]]
        # Format CV results as "mean±std" in percentage points, computed from fold values
        mean_std_fmt = []
        for idx, row in summary_subjects.iterrows():
            mean = np.round(np.mean(row["balanced_acc_vals"])*100, 1)
            std = np.round(np.std(row["balanced_acc_vals"])*100, 1)
            mean_std_fmt.append(f"{mean}±{std}")
        mean_std_dict = {"mean_std_perc" : mean_std_fmt}
        summary_subjects = pd.concat((summary_subjects, pd.DataFrame(mean_std_dict, index=summary_subjects.index)), axis=1).reset_index(drop=True)

        # Across-subject aggregate (mean and std of per-subject means)
        all_accs = np.array(summary_subjects["balanced_acc_mean"].values)
        summary_subjects.loc[4, "subject"] = 'All'
        summary_subjects.loc[4, "mean_std_perc"] = f"{np.round((np.mean(all_accs)*100),2)}±{np.round((np.std(all_accs)*100),2)}"
        # Save numeric summary
        summary_subjects.to_csv(res_save_folder / f"{model_to_select}_{model_id}_{condition}_win{win_size_ms_to_consider}_{experiment_to_analyze}.csv")
        print(f"Model: {model_to_select} - Window size: {win_size_unique} - condition:{condition}")
        print(summary_subjects)

        print("\n\n")
    

        # Plot per-subject confusion matrix
        if plot_confusion_matrix:
            fig, axs = plt.subplots(
                1,
                len(subjects_to_consider),
                figsize=(20, 6),
                sharey=True,
                constrained_layout=True
            )

            axs = np.atleast_1d(axs)
            for subject_id, subject in enumerate(subjects_to_consider):
                # Extract subject
                curr_subj = model_condition[model_condition["subject"]==subject]
                run_path = curr_subj["run_path"].iloc[0]
                # extract csv summary path
                df = pd.read_csv(f"{run_path}/cv_summary.csv")

                cm_mean, cm_std = mean_std_confusion_matrices(df["confusion_matrix"])

                ax = axs[subject_id]
                disp_lables = list(curr_subj["train_label_map"].iloc[0].values())
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_mean, display_labels=disp_lables)

                # heatmap from mean (no default numbers)
                disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, include_values=False)
            
                mean_sub = np.round(np.mean(curr_subj['balanced_acc_vals'].iloc[0])*100, 1)
                std_sub = np.round(np.std(curr_subj['balanced_acc_vals'].iloc[0])*100, 1)
                title = f"{subject} | {mean_sub}±{std_sub}"
                ax.set_title(title, fontsize=18)  # one title per subject
                ax.tick_params(axis="x", labelrotation=45, labelsize=14)
                ax.set_xticklabels(ax.get_xticklabels(), ha="right")
                ax.tick_params(axis="y", labelsize=14)
                # annotate mean ± std
                for (i, j), m in np.ndenumerate(cm_mean):
                    s = cm_std[i, j]
                    ax.text(j, i, f"{m:.2f}\n±{s:.2f}", ha="center", va="center", fontsize=10)
                #plt.tight_layout()
            #fig.subplots_adjust(top=0.88, bottom=0.12)

            plt.savefig(fig_save_folder/f"{model_to_select}_{model_id}_{condition}_win{win_size_ms_to_consider}_{experiment_to_analyze}.svg", bbox_inches="tight")  # final guarantee)

            # Plot per-subject confusion matrix
        if plot_confusion_matrix:

            n_subj = len(subjects_to_consider)
            ncols = 2
            nrows = 2

            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=(10, 4.5 * nrows),
                sharex=True,
                sharey=True,
                constrained_layout=False
            )

            fig.subplots_adjust(
                left=0.12,
                right=0.98,
                top=0.92,
                bottom=0.10,
                wspace=0.08,
                hspace=0.25
            )

            axs = np.atleast_2d(axs)

            # store first image of each row for colorbar
            row_images = {}

            for subject_id, subject in enumerate(subjects_to_consider):
                row = subject_id // 2
                col = subject_id % 2

                curr_subj = model_condition[model_condition["subject"] == subject]
                run_path = curr_subj["run_path"].iloc[0]

                df = pd.read_csv(f"{run_path}/cv_summary.csv")
                cm_mean, cm_std = mean_std_confusion_matrices(df["confusion_matrix"])

                ax = axs[row, col]

                disp_labels = list(curr_subj["train_label_map"].iloc[0].values())

                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm_mean,
                    display_labels=disp_labels
                )

                disp.plot(
                    ax=ax,
                    cmap=plt.cm.Blues,
                    colorbar=False,
                    include_values=False, 
                )
                im = ax.images[0]
                im.set_clim(0.0, 1.0)

                mean_sub = np.round(np.mean(curr_subj["balanced_acc_vals"].iloc[0]) * 100, 1)
                std_sub  = np.round(np.std(curr_subj["balanced_acc_vals"].iloc[0]) * 100, 1)

                ax.set_title(f"{subject} | {mean_sub}±{std_sub}", fontsize=20)
                ax.tick_params(axis="x", labelrotation=45, labelsize=15)
                ax.set_xticklabels(ax.get_xticklabels(), ha="right")
                ax.tick_params(axis="y", labelsize=15)
                ax.set_xlabel("")
                ax.set_ylabel("")

                # store first image per row
                if row not in row_images:
                    row_images[row] = ax.images[0]

            # turn off unused axes
            for k in range(n_subj, nrows * ncols):
                axs.flatten()[k].axis("off")

            # ---- Add one colorbar per row (on the LEFT) ----
            for row in range(nrows):
                if row in row_images:
                    cbar = fig.colorbar(
                        row_images[row],
                        ax=axs[row, :],
                        location="left",
                        fraction=0.05,
                        pad=0.15
                    )
                    cbar.ax.tick_params(labelsize=15)
                    cbar.set_label("Accuracy", fontsize=15)

            plt.savefig(fig_save_folder/f"{model_to_select}_{model_id}_{condition}_win{win_size_ms_to_consider}_{experiment_to_analyze}_2x2.svg", 
                        bbox_inches="tight", transparent=True) 

