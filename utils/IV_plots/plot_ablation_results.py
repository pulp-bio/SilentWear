# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Plots for the classification ablation studies
========================================================================

Regenerates the ablation figures used in the thesis in a small, serif-font
style consistent with the
other figures of the report and suitable for a paper as well.

It reads the ``cv_summary.csv`` files produced by the ablation trainers,
aggregates ``balanced_accuracy`` (mean over folds, then mean over subjects),
and emits four vector PDFs plus the aggregated numbers on stdout.

Only the ``3_subjects_6_sessions`` runs are used:

  1. session_count_ablation                        -> ablation_session_count.pdf
  2. data_augmentation_ablation_stride_dim         -> ablation_aug_stride.pdf
  3. data_augmentation_ablation_num_strides        -> ablation_aug_numstrides.pdf
  4. stride_dim (augmented) vs stride_dim (orig.)  -> ablation_aug_quantity_quality.pdf

Usage
-----
    python utils/IV_plots/plot_ablation_results.py \
        --artifacts_root artifacts/artifacts_ablation \
        --out_dir eth-report/report_template/figures
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    }
)

GRAY = "#7f7f7f"
ORANGE = "#E69F00"
BLUE = "#0072B2"
GREEN = "#009E73"
VERMILLION = "#D55E00"

EXPS = ["global", "inter_session"]
EXP_TITLE = {"global": "Global", "inter_session": "Inter-session"}
CONDS = ["vocalized", "silent"]
COND_TITLE = {"vocalized": "Vocalized", "silent": "Silent"}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def fold_mean(csv_path: Path) -> Optional[float]:
    """Mean of the ``balanced_accuracy`` column over the folds of one run."""
    vals: List[float] = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            try:
                vals.append(float(row["balanced_accuracy"]))
            except (KeyError, ValueError):
                pass
    return st.mean(vals) if vals else None


def collect(root: Path, has_variant: bool) -> Dict[Tuple[str, int, str, str], Dict[str, float]]:
    """Walk ``root`` and return {(variant, n_sess, exp, cond): {subject: bacc%}}.

    ``has_variant`` is False for the session-count ablation, whose tree has no
    per-variant level (the model is always the un-augmented baseline).
    """
    out: Dict[Tuple[str, int, str, str], Dict[str, float]] = defaultdict(dict)
    for csv_path in root.rglob("cv_summary.csv"):
        parts = csv_path.parts
        try:
            mi = parts.index("models")
            exp, subj, cond = parts[mi + 1], parts[mi + 2], parts[mi + 3]
            n_sess = int(parts[mi - 1].split("_")[0])
            variant = parts[mi - 2] if has_variant else "baseline"
        except (ValueError, IndexError):
            continue
        m = fold_mean(csv_path)
        if m is not None:
            out[(variant, n_sess, exp, cond)][subj] = m * 100.0
    return out


def curve(
    data: Dict[Tuple[str, int, str, str], Dict[str, float]],
    variant: str,
    exp: str,
    cond: str,
    sessions=range(1, 7),
) -> Tuple[List[int], List[float], List[float]]:
    """Return (xs, mean over subjects, sem over subjects) for one series."""
    xs, means, sems = [], [], []
    for n in sessions:
        subj_vals = list(data.get((variant, n, exp, cond), {}).values())
        if not subj_vals:
            continue
        xs.append(n)
        means.append(st.mean(subj_vals))
        sems.append(st.pstdev(subj_vals) / (len(subj_vals) ** 0.5) if len(subj_vals) > 1 else 0.0)
    return xs, means, sems


# ---------------------------------------------------------------------------
# Plot primitives
# ---------------------------------------------------------------------------


def plot_series(ax, xs, means, sems, color, label, linestyle="-", marker="o"):
    if not xs:
        return
    lo = [m - s for m, s in zip(means, sems)]
    hi = [m + s for m, s in zip(means, sems)]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(xs, means, color=color, label=label, linestyle=linestyle, marker=marker,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.4)


def style_axis(ax, ylim):
    ax.set_ylim(*ylim)
    ax.set_xticks(range(1, 7))
    ax.set_xlim(0.8, 6.2)
    ax.yaxis.set_major_locator(MultipleLocator(10))  # horizontal gridline every 10%
    ax.grid(True, axis="y")
    ax.tick_params(length=2.5)


def common_ylim(all_means, pad=3.0, floor=0.0, ceil=100.0):
    # Snap the limits to multiples of 10 so the every-10% gridlines meet the frame.
    lo = max(floor, math.floor((min(all_means) - pad) / 10.0) * 10)
    hi = min(ceil, math.ceil((max(all_means) + pad) / 10.0) * 10)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Figure 1: session-count ablation (protocol adequacy)
# ---------------------------------------------------------------------------


def fig_session_count(data, out_path: Path):
    series = [("vocalized", BLUE), ("silent", VERMILLION)]
    all_m = []
    for exp in EXPS:
        for cond, _ in series:
            all_m += curve(data, "baseline", exp, cond)[1]
    ylim = common_ylim(all_m)

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), sharey=True)
    for ax, exp in zip(axes, EXPS):
        for cond, color in series:
            xs, m, s = curve(data, "baseline", exp, cond)
            plot_series(ax, xs, m, s, color, COND_TITLE[cond])
        ax.set_title(EXP_TITLE[exp])
        ax.set_xlabel("Enrolment sessions")
        style_axis(ax, ylim)
    axes[0].set_ylabel("Balanced accuracy (%)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Figures 2 & 3: augmentation hyper-parameter sweeps (2x2 small multiples)
# ---------------------------------------------------------------------------


def fig_sweep(data, variants, out_path: Path, title=None):
    """variants: list of (key, label, color)."""
    all_m = []
    for key, _, _ in variants:
        for exp in EXPS:
            for cond in CONDS:
                all_m += curve(data, key, exp, cond)[1]
    ylim = common_ylim(all_m)

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 2.9), sharex=True, sharey=True)
    for r, exp in enumerate(EXPS):
        for c, cond in enumerate(CONDS):
            ax = axes[r][c]
            for key, label, color in variants:
                xs, m, s = curve(data, key, exp, cond)
                plot_series(ax, xs, m, s, color, label)
            style_axis(ax, ylim)
            if r == 0:
                ax.set_title(COND_TITLE[cond])
            if c == 0:
                ax.set_ylabel(f"{EXP_TITLE[exp]}\nBal. acc. (%)")
            if r == 1:
                ax.set_xlabel("Enrolment sessions")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(variants),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: quantity vs quality (augmented solid vs original-size dashed)
# ---------------------------------------------------------------------------


def fig_quantity_quality(data_aug, data_orig, out_path: Path):
    strides = [("stride10_n2", "10 ms", ORANGE), ("stride20_n2", "20 ms", BLUE)]
    all_m = []
    for key, _, _ in strides:
        for exp in EXPS:
            for cond in CONDS:
                all_m += curve(data_aug, key, exp, cond)[1]
                all_m += curve(data_orig, key, exp, cond)[1]
                all_m += curve(data_aug, "baseline", exp, cond)[1]
    ylim = common_ylim(all_m)

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 3.2), sharex=True, sharey=True)
    for r, exp in enumerate(EXPS):
        for c, cond in enumerate(CONDS):
            ax = axes[r][c]
            xs, m, s = curve(data_aug, "baseline", exp, cond)
            plot_series(ax, xs, m, s, GRAY, "No augmentation")
            for key, label, color in strides:
                xs, m, s = curve(data_aug, key, exp, cond)
                plot_series(ax, xs, m, s, color, f"{label}, augmented", linestyle="-")
                xo, mo, so = curve(data_orig, key, exp, cond)
                plot_series(ax, xo, mo, so, color, f"{label}, original size",
                            linestyle="--", marker="s")
            style_axis(ax, ylim)
            if r == 0:
                ax.set_title(COND_TITLE[cond])
            if c == 0:
                ax.set_ylabel(f"{EXP_TITLE[exp]}\nBal. acc. (%)")
            if r == 1:
                ax.set_xlabel("Enrolment sessions")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Text summary (the numbers that go into the tables)
# ---------------------------------------------------------------------------


def print_table(name, data, variants, at_session=6):
    print(f"\n===== {name} | balanced acc (%), mean over subjects @ {at_session} sessions =====")
    header = f"{'variant':16s} " + " ".join(f"{EXP_TITLE[e][:5]:>6s}/{c[:3]}" for e in EXPS for c in CONDS)
    print(header)
    for key, label, _ in variants:
        cells = []
        for e in EXPS:
            for c in CONDS:
                subj = data.get((key, at_session, e, c), {})
                cells.append(f"{st.mean(subj.values()):11.1f}" if subj else f"{'--':>11s}")
        print(f"{label:16s} " + " ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", type=Path, default=Path("artifacts/artifacts_ablation"))
    ap.add_argument("--out_dir", type=Path, default=Path("eth-report/report_template/figures"))
    args = ap.parse_args()

    base = args.artifacts_root / "artifacts_ablation_3_subjects_6_sessions"
    base_orig = args.artifacts_root / "artifacts_ablation_3_subjects_6_sessions_original_size"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d_session = collect(base / "session_count_ablation", has_variant=False)
    d_stride = collect(base / "data_augmentation_ablation_stride_dim", has_variant=True)
    d_numstr = collect(base / "data_augmentation_ablation_num_strides", has_variant=True)
    d_stride_orig = collect(base_orig / "data_augmentation_ablation_stride_dim", has_variant=True)

    stride_variants = [
        ("baseline", "No augmentation", GRAY),
        ("stride10_n2", "10 ms", ORANGE),
        ("stride20_n2", "20 ms", BLUE),
        ("stride50_n2", "50 ms", GREEN),
        ("stride100_n2", "100 ms", VERMILLION),
    ]
    numstr_variants = [
        ("baseline", "No augmentation", GRAY),
        ("stride10_n2", "2 shifts", ORANGE),
        ("stride10_n5", "5 shifts", BLUE),
        ("stride10_n10", "10 shifts", GREEN),
    ]

    fig_session_count(d_session, args.out_dir / "ablation_session_count.pdf")
    fig_sweep(d_stride, stride_variants, args.out_dir / "ablation_aug_stride.pdf")
    fig_sweep(d_numstr, numstr_variants, args.out_dir / "ablation_aug_numstrides.pdf")
    fig_quantity_quality(d_stride, d_stride_orig, args.out_dir / "ablation_aug_quantity_quality.pdf")

    print_table("session_count (baseline)", d_session, [("baseline", "baseline", GRAY)])
    print_table("stride sweep (augmented)", d_stride, stride_variants)
    print_table("num-strides sweep (augmented)", d_numstr, numstr_variants)
    print_table("stride sweep (original size)", d_stride_orig, stride_variants)


if __name__ == "__main__":
    main()
