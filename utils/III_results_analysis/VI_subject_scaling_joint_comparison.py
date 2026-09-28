# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
VI_subject_scaling_joint_comparison.py

Subject scaling with and without joint-modality pre-training
(offline_experiments/VI_subject_scaling_experiment.py, default vs --joint_pretraining).
Both runs are aggregated as in IV_subject_scaling_analysis.py; pre-training pools,
target data (fine-tuning and test) and splits are the same, only the pool data differ:
    mode-specific   pool subjects contribute the evaluated condition only
    joint           pool subjects contribute silent + vocalized data

Inputs:
  <ARTIFACTS_DIR>/models/subject_scaling/<condition>/<model_name_id>/results.csv
  <ARTIFACTS_DIR>/models/subject_scaling_joint/<condition>/<model_name_id>/results.csv
Outputs:
  <ARTIFACTS_DIR>/tables/subject_scaling_joint_vs_mode_specific_<condition>_<model_name_id>.csv
  <ARTIFACTS_DIR>/figures/subject_scaling_joint_vs_mode_specific_<condition>_<model_name>_<model_name_id>.pdf (+ .png)

Example:
  python utils/III_results_analysis/VI_subject_scaling_joint_comparison.py \
    --artifacts_dir ./artifacts_rebuttal/seeds_pooled
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.III_results_analysis.IV_subject_scaling_analysis import (
    BAR_COLORS,
    BAR_LABELS,
    BARS,
    aggregate,
)

VARIANTS = [("mode_specific", "subject_scaling"), ("joint", "subject_scaling_joint")]


def compare(artifacts_dir: Path, cond: str, model_name_id: str) -> pd.DataFrame:
    tabs = []
    for name, experiment in VARIANTS:
        path = artifacts_dir / "models" / experiment / cond / model_name_id / "results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        t = aggregate(path)[["target", "n_pretrain_subjects", "bar", "mean", "std", "mean_std_perc"]]
        tabs.append(t.rename(columns={c: f"{c}_{name}" for c in ("mean", "std", "mean_std_perc")}))
    keys = ["target", "n_pretrain_subjects", "bar"]
    out = tabs[0].merge(tabs[1], on=keys, how="outer", validate="one_to_one")
    out["delta_pp"] = (100 * (out["mean_joint"] - out["mean_mode_specific"])).round(1)
    return out.sort_values(["target", "bar", "n_pretrain_subjects"]).reset_index(drop=True)


def plot(comp: pd.DataFrame, subjects, save_path: Path) -> None:
    """Paper Fig. 7-8 style; per bar type: mode-specific (light, hatched) | joint (solid)."""
    panels = list(subjects) + ["Average"]
    n_pretrain = sorted(int(k) for k in comp["n_pretrain_subjects"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(40, 8), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    fs = 30
    x = np.arange(len(n_pretrain))
    width = 0.13

    for i, (ax, panel) in enumerate(zip(axes, panels)):
        ax.set_facecolor("#f4f4f4" if i % 2 == 1 else "#ffffff")
        ax.set_axisbelow(True)
        d = comp[comp["target"] == panel]
        for b, bar in enumerate(BARS):
            db = d[d["bar"] == bar].set_index("n_pretrain_subjects")
            for v, (name, _) in enumerate(VARIANTS):
                mean = np.array([db[f"mean_{name}"].get(k, np.nan) for k in n_pretrain]) * 100
                std = np.array([db[f"std_{name}"].get(k, np.nan) for k in n_pretrain]) * 100
                pos = x + (b - 1) * 2 * width + (v - 0.5) * width
                if name == "joint":
                    ax.bar(pos, mean, width=width, color=BAR_COLORS[bar], zorder=2)
                else:
                    ax.bar(pos, mean, width=width, color=BAR_COLORS[bar], alpha=0.35,
                           hatch="//", edgecolor="white", linewidth=0, zorder=2)
                ax.errorbar(pos, mean, yerr=std, fmt="none", ecolor="black", capsize=3,
                            linewidth=1.5, zorder=3)
        ax.set_title(panel, fontsize=fs, y=0.92)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in n_pretrain])
        ax.set_xlim(-0.6, len(n_pretrain) - 0.4)
        ax.tick_params(axis="both", labelsize=fs, direction="in")
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        if i > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)

    handles = [patches.Patch(color=BAR_COLORS[b], label=BAR_LABELS[b]) for b in BARS]
    handles += [
        patches.Patch(facecolor="grey", alpha=0.35, hatch="//", edgecolor="white",
                      label="Mode-specific pre-training"),
        patches.Patch(facecolor="grey", label="Joint pre-training"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=len(handles),
               frameon=False, fontsize=fs * 0.8, handlelength=2.0, columnspacing=0.8)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.13, top=0.86, wspace=0.05)
    fig.supylabel("Accuracy (%)", fontsize=fs, x=0.04)
    fig.supxlabel("Number of pre-training subjects", fontsize=fs, y=0.03)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Subject scaling: joint vs mode-specific pre-training")
    ap.add_argument("--artifacts_dir", type=Path, required=True)
    ap.add_argument("--model_name", default="speechnet")
    ap.add_argument("--model_name_id", default="w1400ms")
    ap.add_argument("--conditions", nargs="+", default=["silent"])
    args = ap.parse_args()

    for cond in args.conditions:
        comp = compare(args.artifacts_dir, cond, args.model_name_id)
        tag = f"subject_scaling_joint_vs_mode_specific_{cond}"
        table = args.artifacts_dir / "tables" / f"{tag}_{args.model_name_id}.csv"
        table.parent.mkdir(parents=True, exist_ok=True)
        comp.to_csv(table, index=False)
        subjects = sorted(t for t in comp["target"].unique() if t != "Average")
        fig = args.artifacts_dir / "figures" / f"{tag}_{args.model_name}_{args.model_name_id}.pdf"
        plot(comp, subjects, fig)

        avg = comp[comp["target"] == "Average"]
        print(f"\n=== {cond}: Average (mode-specific -> joint, delta pp) ===")
        for _, r in avg.iterrows():
            print(f"  x={int(r.n_pretrain_subjects)} {r.bar:9s} {r.mean_std_perc_mode_specific:>10s} -> "
                  f"{r.mean_std_perc_joint:>10s}  ({r.delta_pp:+.1f})")
        print(f"[SAVED] {table}\n[SAVED] {fig}")


if __name__ == "__main__":
    main()
