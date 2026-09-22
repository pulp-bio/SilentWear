#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0

"""
inter_session_fold_scatter.py

Analyze Balanced vs Unbalanced Accuracy across folds (sessions)
for inter-session experiments. Generates one combined block plot 
per condition containing all subjects, using twin Y axes.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from general_utils import build_twin_axis_blocks


def get_latest_model_run(win_dir: Path) -> Optional[Path]:
    """Finds the latest model run (e.g., model_6) in a given window directory."""
    if not win_dir.exists():
        return None
        
    candidates = []
    for mr_dir in win_dir.iterdir():
        if not mr_dir.is_dir():
            continue
        m = re.match(r"model_(\d+)$", mr_dir.name)
        if m:
            p = mr_dir / "cv_summary.csv"
            if p.exists():
                candidates.append((int(m.group(1)), p))
                
    if not candidates:
        return None
        
    return sorted(candidates, key=lambda x: x[0])[-1][1]


def plot_condition_blocks(
    condition: str,
    model_name: str,
    subjects: list[str],
    series_bal: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    series_unbal: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    max_folds: int,
    save_path: Path
):
    """Generates the multi-subject block plot for a specific condition."""
    
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.0,
        }
    )

    # Adjust width dynamically based on the number of subjects + Average
    n_blocks = len(subjects) + 1
    fig, ax1 = plt.subplots(1, 1, figsize=(1.8 * n_blocks, 3.2))
    ax2 = ax1.twinx()

    x_values = np.arange(1, max_folds + 1)

    # Define custom styles to match the previous plots
    s1_styles = {"Balanced Acc": {"color": "blue", "marker": "o", "alpha": 0.95}}
    s2_styles = {"Unbalanced Acc": {"color": "red", "marker": "o", "alpha": 0.85}}

    build_twin_axis_blocks(
        ax1=ax1,
        ax2=ax2,
        subjects=subjects,
        x_values=x_values,
        series1=series_bal,
        series2=series_unbal,
        series1_styles=s1_styles,
        series2_styles=s2_styles,
        with_average=False,
        x_label="Fold (Session)",
        y1_label="Balanced Accuracy (%)",
        y2_label="Unbalanced Accuracy (%)",
        y1_lim=(0, 100),
        y2_lim=(0, 100),
        y1_major_step=10,
        y2_major_step=10,
        x_tick_labels=[str(x) for x in x_values],
        gap=1.0,
        block_margin=0.5,
        label_y_offset=10.0, # Drop the label slightly below the upper bound
        label_fontsize=10,
        label_position="bottom" # Place the block labels at the bottom of the plot
    )

    ax1.set_title(f"Inter-session Performance | {condition.capitalize()} | {model_name}", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts_dir", 
        type=Path, 
        default=None,
        help="Root artifacts folder (default: env SILENTWEAR_ARTIFACTS_DIR or ./artifacts)"
    )
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    ap.add_argument("--model_name", type=str, required=True, help="e.g., random_forest")
    ap.add_argument("--model_name_id", type=str, default="w1400ms", help="Window to plot, e.g., w1400ms")
    
    args = ap.parse_args()

    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        env = os.environ.get("SILENTWEAR_ARTIFACTS_DIR", None)
        artifacts_dir = Path(env) if env else Path("./artifacts")

    experiment = "inter_session"
    models_root = artifacts_dir / "models" / experiment
    
    if not models_root.exists():
        print(f"[ERROR] Experiment folder not found: {models_root}")
        return

    figures_dir = artifacts_dir / "figures" / "intersession_blocks"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        series_bal = {"Balanced Acc": {}}
        series_unbal = {"Unbalanced Acc": {}}
        
        max_folds = 0
        valid_subjects = []

        # Gather data for all subjects in the current condition
        for subject in args.subjects:
            win_dir = models_root / subject / condition / args.model_name / args.model_name_id
            if not win_dir.exists():
                continue
                
            csv_path = get_latest_model_run(win_dir)
            if csv_path is None:
                continue
                
            df = pd.read_csv(csv_path)
            if "balanced_accuracy" not in df.columns or "accuracy" not in df.columns:
                continue
            
            # Extract metrics
            bal_acc = df["balanced_accuracy"].astype(float).to_numpy() * 100.0
            unbal_acc = df["accuracy"].astype(float).to_numpy() * 100.0
            
            # Record maximum folds to align axes properly
            n_folds = len(bal_acc)
            if n_folds > max_folds:
                max_folds = n_folds

            # Append to series dictionary for plotting
            series_bal["Balanced Acc"][subject] = {
                "mean": bal_acc, 
                "std": np.zeros_like(bal_acc) # Scatter points have no standard deviation
            }
            series_unbal["Unbalanced Acc"][subject] = {
                "mean": unbal_acc, 
                "std": np.zeros_like(unbal_acc)
            }
            valid_subjects.append(subject)

        if not valid_subjects:
            print(f"[SKIP] No valid subjects found for condition: {condition}")
            continue

        out_filename = figures_dir / f"block_plot_{condition}_{args.model_name}_{args.model_name_id}.png"
        
        plot_condition_blocks(
            condition=condition,
            model_name=args.model_name,
            subjects=valid_subjects,
            series_bal=series_bal,
            series_unbal=series_unbal,
            max_folds=max_folds,
            save_path=out_filename
        )
        print(f"[SAVED] {out_filename}")

if __name__ == "__main__":
    main()