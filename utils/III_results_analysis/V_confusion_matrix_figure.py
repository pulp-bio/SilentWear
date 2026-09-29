# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
V_confusion_matrix_figure.py

Paper Fig. 4 in a single vector figure: per-subject confusion matrices for
  a) Global Evaluation        (Vocalized | Silent)
  b) Inter-Session Evaluations (Vocalized | Silent)
each condition as a 2x2 grid of subjects (S01 S02 / S03 S04).

Same data and style as the per-condition figures of I_global_intersession_analysis.py
(--plot_confusion_matrix): matrix = mean over folds of the row-normalised confusion
matrices, title = "<subject> | mean±std" of the balanced accuracy over folds,
Blues colormap on a fixed 0-1 scale, one "Accuracy" colorbar per row.

Input:
  <ARTIFACTS_DIR>/models/<experiment>/<subject>/<condition>/<model_name>/<model_name_id>/<model_run>/cv_summary.csv
Outputs:
  <ARTIFACTS_DIR>/figures/<model_name>_<model_name_id>_cm_global_inter_session.{svg,pdf,png}

Example:
  python utils/III_results_analysis/V_confusion_matrix_figure.py \
    --artifacts_dir ./artifacts/seeds_pooled
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
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.I_data_preparation.experimental_config import ORIGINAL_LABELS
from utils.III_results_analysis.I_global_intersession_analysis import (
    _latest_model_run,
    _pick_cm_col,
    mean_std_confusion_matrices,
)

BLOCKS = [("global", "a) Global Evaluation"), ("inter_session", "b) Inter-Session Evaluations")]
CONDITIONS = [("vocalized", "Vocalized"), ("silent", "Silent")]


def load_subject(artifacts_dir: Path, experiment, subject, condition, model_name, model_name_id, model_run):
    base = artifacts_dir / "models" / experiment / subject / condition / model_name / model_name_id
    run = model_run if model_run else _latest_model_run(base)
    if run is None:
        raise FileNotFoundError(f"no model runs in {base}")
    df = pd.read_csv(base / run / "cv_summary.csv")
    cm_mean, _ = mean_std_confusion_matrices(df[_pick_cm_col(df)])
    vals = df["balanced_accuracy"].astype(float).to_numpy()
    title = f"{subject} | {np.round(np.mean(vals) * 100, 1)}±{np.round(np.std(vals) * 100, 1)}"
    return cm_mean, title


def plot(artifacts_dir: Path, model_name: str, model_name_id: str, model_run, subjects, out_base: Path):
    labels = list(ORIGINAL_LABELS.values())
    fs_title, fs_tick, fs_head, fs_sub = 20, 15, 24, 22

    # columns: [colorbar] + per block: [cm cm gap cm cm], blocks separated by a wider gap
    cm_w, gap_cond, gap_block, cbar_w, cbar_gap = 1.0, 0.18, 0.45, 0.06, 0.12
    widths = [cbar_w, cbar_gap]
    for b in range(len(BLOCKS)):
        if b > 0:
            widths.append(gap_block)
        for c in range(len(CONDITIONS)):
            if c > 0:
                widths.append(gap_cond)
            widths += [cm_w, cm_w]

    fig = plt.figure(figsize=(40, 10.5))
    gs = GridSpec(2, len(widths), figure=fig, width_ratios=widths, wspace=0.08, hspace=0.28,
                  left=0.03, right=0.995, top=0.84, bottom=0.12)

    # map (block, condition, sub-column) -> gridspec column
    col_of = {}
    col = 2
    for b in range(len(BLOCKS)):
        if b > 0:
            col += 1
        for c in range(len(CONDITIONS)):
            if c > 0:
                col += 1
            col_of[(b, c, 0)], col_of[(b, c, 1)] = col, col + 1
            col += 2

    image = None
    first_ax = {}  # first matrix of each row, to size the colorbars
    for b, (experiment, _) in enumerate(BLOCKS):
        for c, (condition, _) in enumerate(CONDITIONS):
            for k, subject in enumerate(subjects):
                r, sc = divmod(k, 2)
                ax = fig.add_subplot(gs[r, col_of[(b, c, sc)]])
                first_ax.setdefault(r, ax)
                cm, title = load_subject(
                    artifacts_dir, experiment, subject, condition, model_name, model_name_id, model_run
                )
                image = ax.imshow(cm, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.set_title(title, fontsize=fs_title)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                if r == 1:
                    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs_tick)
                else:
                    ax.set_xticklabels([])
                if sc == 0:
                    ax.set_yticklabels(labels, fontsize=fs_tick)
                else:
                    ax.set_yticklabels([])

    # one colorbar per row (left), as tall as the (square) matrices of that row
    fig.canvas.draw()
    for r in range(2):
        cax = fig.add_subplot(gs[r, 0])
        pos, ref = cax.get_position(), first_ax[r].get_position()
        cax.set_position([pos.x0, ref.y0, pos.width, ref.height])
        cbar = fig.colorbar(image, cax=cax)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.tick_params(labelsize=fs_tick)
        cbar.set_label("Accuracy", fontsize=fs_tick)

    # headers: block titles and condition titles
    fig.canvas.draw()

    def span(c0, c1):
        a0 = fig.axes[0].get_gridspec()[0, c0].get_position(fig)
        a1 = fig.axes[0].get_gridspec()[0, c1].get_position(fig)
        return (a0.x0 + a1.x1) / 2

    for b, (_, block_title) in enumerate(BLOCKS):
        x = span(col_of[(b, 0, 0)], col_of[(b, len(CONDITIONS) - 1, 1)])
        fig.text(x, 0.955, block_title, ha="center", va="center", fontsize=fs_head)
        for c, (_, cond_title) in enumerate(CONDITIONS):
            fig.text(span(col_of[(b, c, 0)], col_of[(b, c, 1)]), 0.905, cond_title,
                     ha="center", va="center", fontsize=fs_sub)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_base}.svg/.pdf/.png")


def main():
    parser = argparse.ArgumentParser(description="Paper Fig. 4: combined confusion-matrix figure")
    parser.add_argument("--artifacts_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="speechnet")
    parser.add_argument("--model_name_id", type=str, default="w1400ms")
    parser.add_argument("--model_run", type=str, default=None, help="e.g. model_1 (default: latest)")
    parser.add_argument("--subjects", nargs=4, default=["S01", "S02", "S03", "S04"])
    args = parser.parse_args()

    out = args.artifacts_dir / "figures" / f"{args.model_name}_{args.model_name_id}_cm_global_inter_session"
    plot(args.artifacts_dir, args.model_name, args.model_name_id, args.model_run, args.subjects, out)


if __name__ == "__main__":
    main()
