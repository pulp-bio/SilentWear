# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
IV_subject_scaling_analysis.py

Summarize and visualize the subject-scaling experiment
(offline_experiments/VI_subject_scaling_experiment.py).

Input (one row per test-session score):
  <ARTIFACTS_DIR>/models/<experiment>/<condition>/<model_name_id>/results.csv
  (<experiment> = subject_scaling, or subject_scaling_norm_<mode> with --normalize)

Aggregation, per condition
  Unit u   session index identifying a data split: the training session for
           ft_1sess, the test session for zero_shot and ft_2sess (3 units).
  Step 1   per (target, x, bar, u): mean over the N(x) pre-training pools and over
           the test sessions of that split.
  Step 2   per (target, x, bar): mean +- std over the 3 units (population std).
  Step 3   Average: mean +- std of the Step-2 means over the target subjects.

Outputs:
  <ARTIFACTS_DIR>/tables/<experiment>_<condition>_<model_name_id>.csv
  <ARTIFACTS_DIR>/figures/<experiment>_<condition>_<model_name>_<model_name_id>.pdf (+ .png)

Figure style follows III_ft_results.py (paper Figs. 7-8): one figure per condition,
one panel per target subject plus an "Average" panel.

Example:
  python utils/III_results_analysis/IV_subject_scaling_analysis.py \
    --artifacts_dir ./artifacts_rebuttal/seed_42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

BARS = ["zero_shot", "ft_1sess", "ft_2sess"]
BAR_LABELS = {
    "zero_shot": "Zero-shot",
    "ft_1sess": "1 target session",
    "ft_2sess": "2 target sessions",
}
BAR_COLORS = {"zero_shot": "blue", "ft_1sess": "green", "ft_2sess": "orange"}


# ------------------------- aggregation -------------------------
def aggregate(results_path: Path) -> pd.DataFrame:
    """Return per-target and Average mean/std per (n_pretrain_subjects, bar)."""
    r = pd.read_csv(results_path, dtype={"pretrain_subjects": str})
    r = r.drop_duplicates(
        subset=["target", "pretrain_subjects", "bar", "unit", "test_session"], keep="last"
    )

    # Step 1: per unit, mean over pools and test sessions
    units = (
        r.groupby(["target", "n_pretrain_subjects", "bar", "unit"])
        .agg(acc=("balanced_accuracy", "mean"), n_models=("pretrain_subjects", "nunique"))
        .reset_index()
    )
    # Step 2: per target, mean +- std over the 3 units
    per_t = (
        units.groupby(["target", "n_pretrain_subjects", "bar"])
        .agg(
            mean=("acc", "mean"),
            std=("acc", lambda a: float(np.std(a))),
            n_units=("acc", "size"),
            n_models_per_unit=("n_models", "max"),
        )
        .reset_index()
    )
    # Step 3: Average, mean +- std over targets
    avg = (
        per_t.groupby(["n_pretrain_subjects", "bar"])
        .agg(mean=("mean", "mean"), std=("mean", lambda a: float(np.std(a))), n_units=("mean", "size"))
        .reset_index()
    )
    avg["target"] = "Average"

    out = pd.concat([per_t, avg], ignore_index=True)
    out["mean_std_perc"] = [f"{100*m:.1f}±{100*s:.1f}" for m, s in zip(out["mean"], out["std"])]
    return out.sort_values(["target", "bar", "n_pretrain_subjects"]).reset_index(drop=True)


# ------------------------- plotting -------------------------
def plot_subjs_and_avg(summary: pd.DataFrame, subjects, save_path: Path) -> None:
    """1 x (Nsubjects+1) grouped-bar layout, styled as III_ft_results.plot_subjs_and_avgs."""
    panels = list(subjects) + ["Average"]
    n_pretrain = sorted(int(k) for k in summary["n_pretrain_subjects"].unique())  # 0..n-1
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(40, 8), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    fs_ax, fs_label, fs_tick, fs_leg = 30, 30, 30, 30
    x = np.arange(len(n_pretrain))
    width = 0.26

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        face = "#f4f4f4" if (i % 2 == 1) else "#ffffff"
        ax.set_facecolor(face)
        ax.set_axisbelow(True)

        d = summary[summary["target"] == panel]
        for b, bar in enumerate(BARS):
            db = d[d["bar"] == bar].set_index("n_pretrain_subjects")
            mean = np.array([db["mean"].get(k, np.nan) for k in n_pretrain]) * 100
            std = np.array([db["std"].get(k, np.nan) for k in n_pretrain]) * 100
            pos = x + (b - 1) * width
            ax.bar(pos, mean, width=width, color=BAR_COLORS[bar], zorder=2)
            ax.errorbar(
                pos,
                mean,
                yerr=std,
                fmt="none",
                ecolor="black",
                capsize=4,
                linewidth=1.5 if panel == "Average" else 2,
                zorder=3,
            )

        ax.set_title(panel, fontsize=fs_ax, y=0.92)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in n_pretrain])
        ax.set_xlim(-0.6, len(n_pretrain) - 0.4)
        ax.tick_params(axis="both", labelsize=fs_tick)

    # --- legend ---
    handles = [patches.Patch(color=BAR_COLORS[b], label=BAR_LABELS[b]) for b in BARS]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(handles),
        frameon=False,
        fontsize=fs_leg,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=0.8,
        labelspacing=0.2,
        borderaxespad=0.0,
    )

    # --- layout ---
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.13, top=0.86, wspace=0.05)
    fig.supylabel("Accuracy (%)", fontsize=fs_label, x=0.04)
    fig.supxlabel("Number of pre-training subjects", fontsize=fs_label, y=0.03)

    fig.canvas.draw()

    # hide per-axes spines and keep y-ticks only on left edge
    x0s = np.array([ax.get_position().x0 for ax in axes])
    left_x0 = x0s.min()
    for ax in axes:
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        if np.isclose(ax.get_position().x0, left_x0, atol=1e-3):
            ax.tick_params(axis="y", left=True, labelleft=True, pad=6)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    # outer box around plotting area
    x0 = min(ax.get_position().x0 for ax in axes)
    y0 = min(ax.get_position().y0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)
    y1 = max(ax.get_position().y1 for ax in axes)
    outer_box = patches.Rectangle(
        (x0 - 0.001, y0 - 0.002),
        (x1 - x0) + 0.001 + 0.002,
        (y1 - y0) + 0.002 + 0.002,
        transform=fig.transFigure,
        fill=False,
        linewidth=0.2,
        edgecolor="black",
        zorder=10,
    )
    fig.add_artist(outer_box)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


# ------------------------- CLI main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Subject-scaling analysis: tables and figures")
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        required=True,
        help="Artifacts root of one run, e.g. ./artifacts_rebuttal/seed_42",
    )
    parser.add_argument("--model_name", type=str, default="speechnet")
    parser.add_argument("--model_name_id", type=str, default="w1400ms")
    parser.add_argument(
        "--subjects", nargs="+", default=None, help="Panels to plot (default: all targets in results)"
    )
    parser.add_argument("--conditions", nargs="+", default=["vocalized", "silent"])
    parser.add_argument(
        "--experiment",
        default="subject_scaling",
        help="models/<experiment>, e.g. subject_scaling_norm_session_oracle",
    )
    args = parser.parse_args()

    for cond in args.conditions:
        results = (
            args.artifacts_dir / "models" / args.experiment / cond / args.model_name_id / "results.csv"
        )
        if not results.exists():
            print(f"[SKIP] no results for {cond}: {results}")
            continue

        summary = aggregate(results)
        table_path = args.artifacts_dir / "tables" / f"{args.experiment}_{cond}_{args.model_name_id}.csv"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(table_path, index=False)

        fig_path = (
            args.artifacts_dir
            / "figures"
            / f"{args.experiment}_{cond}_{args.model_name}_{args.model_name_id}.pdf"
        )
        subjects = args.subjects or sorted(t for t in summary["target"].unique() if t != "Average")
        plot_subjs_and_avg(summary, subjects, fig_path)

        print(f"\n=== {cond} ===")
        print(
            summary.pivot_table(
                index="target",
                columns=["bar", "n_pretrain_subjects"],
                values="mean_std_perc",
                aggfunc="first",
            ).to_string()
        )
        print(f"[SAVED] {table_path}\n[SAVED] {fig_path}")


if __name__ == "__main__":
    main()
