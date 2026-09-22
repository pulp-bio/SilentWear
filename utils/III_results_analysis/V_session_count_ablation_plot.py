#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Plotting Script for Session Count Ablation Studies

Reads the outputs directly from the folder structure generated
by Session_Count_Ablation_Trainer.py.

Expected folder hierarchy

session_count_ablation:
  {args.artifacts_dir}/{n}_sess/models/{exp}/{subject}/{condition}/{net_id}/{window_id}/{model_run}/cv_summary.csv
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, MaxNLocator

from general_utils import resolve_csv_path, model_run_tag, window_from_path, build_save_path


def collect_ablation_data(
    ablation_root: Path,
    experiment_name: str,
    model_run: Optional[str] = None,
) -> pd.DataFrame:
    """Scans the ablation_root for CSV files matching the expected folder structure and collects them into a DataFrame."""
    records: list[dict] = []
    for p in ablation_root.rglob("*"):
        parts = p.parts
        if experiment_name not in parts or not p.is_dir():
            continue

        exp_idx = parts.index(experiment_name)
        if len(parts) != exp_idx + 3:
            continue

        condition_dir = p
        try:
            subject = parts[exp_idx + 1]
            condition = parts[exp_idx + 2]
            n_sess_str = next(seg for seg in parts if seg.endswith("_sess"))
            n_sessions = int(n_sess_str.split("_")[0])

            csv_path = resolve_csv_path(condition_dir, model_run)
            if csv_path is None:
                continue

            df_csv = pd.read_csv(csv_path)
            metric_col = "balanced_accuracy"
            if metric_col not in df_csv.columns:
                continue

            metric_vals = df_csv[metric_col].astype(float).to_numpy()
            if 0 < np.mean(metric_vals) <= 1.0:
                metric_vals *= 100.0

            record: dict = {
                "subject": subject,
                "condition": condition,
                "session_count": n_sessions,
                "window_s": window_from_path(csv_path),
                "metric_mean": float(np.mean(metric_vals)),
                "metric_std": float(np.std(metric_vals)),
                "is_acc": True,
                "model_run_tag": model_run_tag(model_run, csv_path),
            }
            records.append(record)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df_res = pd.DataFrame(records)
    subset = ["subject", "condition", "session_count"]
    df_res = df_res.sort_values(
        "model_run_tag",
        key=lambda s: s.apply(
            lambda x: int(x.split("_")[-1]) if re.match(r"model_\d+", x) else -1
        ),
    )
    return df_res.drop_duplicates(subset=subset, keep="last")


def plot_session_count(
    df: pd.DataFrame,
    experiment_name: str,
    save_path: Path,
    model_run: Optional[str] = None,
):
    """Generates line plots of the metric vs. session count for each condition and window, with one line per subject."""
    conditions = sorted(df["condition"].unique())
    windows = sorted(df["window_s"].unique())
    subjects = sorted(df["subject"].unique())
    is_acc = df["is_acc"].iloc[0]

    tags = df["model_run_tag"].unique()
    mr_tag = model_run if model_run is not None else (tags[0] if len(tags) == 1 else "latest")

    palette = plt.colormaps["tab10"]
    subject_colors = {subj: palette(i % 10) for i, subj in enumerate(subjects)}

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 12})

    for w in windows:
        df_w = df[df["window_s"] == w]
        fig, axes = plt.subplots(
            len(conditions), 1,
            figsize=(8.0, 5.0 * len(conditions)),
            sharex=True,
        )
        if len(conditions) == 1:
            axes = [axes]

        fig.suptitle(
            f"Session-Count Ablation ({experiment_name}) | Window = {w:g} s",
            y=1.02, fontsize=14, fontweight="bold",
        )

        for ax, cond in zip(axes, conditions):
            df_cond = df_w[df_w["condition"] == cond]
            ax.set_title(cond.capitalize(), fontsize=12, fontweight="bold")
            ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.4, color="#cccccc")

            for subj in subjects:
                df_s = df_cond[df_cond["subject"] == subj].sort_values("session_count")
                if df_s.empty:
                    continue
                ax.errorbar(
                    df_s["session_count"].astype(float),
                    df_s["metric_mean"],
                    yerr=df_s["metric_std"],
                    fmt="-o", color=subject_colors[subj], markersize=4, linewidth=1.0,
                    capsize=2.5, elinewidth=0.8, alpha=0.9,
                    label=subj,
                )

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("Accuracy (%)" if is_acc else "Loss")
            if is_acc:
                ax.set_ylim(0, 100)
                ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.legend(title="Subject", loc="lower right", frameon=True, fontsize=8)

        axes[-1].set_xlabel("Sessions considered")

        plt.tight_layout()
        w_ms = int(w * 1000)
        suffix = f"_w{w_ms}ms{save_path.suffix}"
        plt.savefig(build_save_path(save_path, mr_tag, suffix), bbox_inches="tight", dpi=300)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument(
        "--experiment", nargs="+",
        choices=["global", "inter_session"],
        default=["global", "inter_session"],
    )
    ap.add_argument(
        "--model_run", type=str, default=None,
        help="e.g. model_6. If omitted, uses the latest model_<k> per folder.",
    )
    args = ap.parse_args()

    if not args.artifacts_dir.exists():
        print(f"[ERROR] Directory not found: {args.artifacts_dir}")
        return

    for exp_name in args.experiment:
        print(f"\n[PLOT] Scanning for session_count | '{exp_name}'...")

        df_results = collect_ablation_data(args.artifacts_dir, exp_name, model_run=args.model_run)

        if df_results.empty:
            print(f"[SKIP] No data found for '{exp_name}'.")
            continue

        figures_dir = args.artifacts_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        out_fig = figures_dir / f"session_count_{exp_name}_summary.png"
        plot_session_count(df_results, exp_name, out_fig, model_run=args.model_run)

        print(f"[SAVED] Generated plots in {figures_dir}")


if __name__ == "__main__":
    main()