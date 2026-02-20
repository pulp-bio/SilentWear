#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
itr_window_sweep_analysis.py

Analyze Information Transfer Rate (ITR) vs window size for inter-session experiments.

Artifact dir layout
  <ARTIFACTS_DIR>/models/inter_session/<subject>/<condition>/<model_name>/<model_name_id>/model_<k>/
    - cv_summary.csv
    - run_cfg.json

Outputs:
  <ARTIFACTS_DIR>/figures/itr_<model_name>_<condition>_<windows_tag>.pdf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator


# ----------------------------- helpers -----------------------------


@dataclass
class RunRef:
    subject: str
    condition: str
    model_name: str
    model_name_id: str
    model_run: str
    run_path: Path
    cv_summary_csv: Path


def _expand_windows_s(vals: List[float], step: float) -> List[float]:
    """
    If [] -> default 0.4..1.4 step 0.2
    If [single] -> that
    If [start end] -> expand inclusive with step
    If [a b c ...] -> explicit list
    """
    if len(vals) == 0:
        start, end = 0.4, 1.4
    elif len(vals) == 1:
        return [float(vals[0])]
    elif len(vals) == 2:
        start, end = float(vals[0]), float(vals[1])
    else:
        return [float(v) for v in vals]

    if step <= 0:
        raise ValueError("--window_step_s must be > 0")

    if end < start:
        start, end = end, start

    out = []
    x = start
    while x <= end + 1e-9:
        out.append(round(x, 3))
        x += step

    if abs(out[-1] - end) > 1e-6:
        out.append(round(end, 3))

    return out


def _model_name_id_from_window_s(window_s: float) -> str:
    return f"w{int(round(window_s * 1000))}ms"


def _latest_model_run(folder: Path) -> Optional[str]:
    if not folder.exists():
        return None
    candidates = []
    for p in folder.iterdir():
        if p.is_dir() and p.name.startswith("model_"):
            try:
                k = int(p.name.split("_")[-1])
                candidates.append((k, p.name))
            except Exception:
                continue
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1][1]


def _find_runs(
    artifacts_dir: Path,
    subjects: List[str],
    conditions: List[str],
    model_name: str,
    model_name_ids: List[str],
    model_run: Optional[str],
    experiment: str = "inter_session",
) -> List[RunRef]:
    """
    Scan:
      artifacts/models/inter_session/<sub>/<cond>/<model_name>/<model_name_id>/model_<k>/
    """
    out: List[RunRef] = []
    root = artifacts_dir / "models" / experiment

    for sub in subjects:
        for cond in conditions:
            for mid in model_name_ids:
                base = root / sub / cond / model_name / mid
                if not base.exists():
                    continue

                mr = model_run if model_run else _latest_model_run(base)
                if mr is None:
                    continue

                run_path = base / mr
                cv = run_path / "cv_summary.csv"
                if not cv.exists():
                    continue

                out.append(
                    RunRef(
                        subject=sub,
                        condition=cond,
                        model_name=model_name,
                        model_name_id=mid,
                        model_run=mr,
                        run_path=run_path,
                        cv_summary_csv=cv,
                    )
                )
    return out


def _compute_itr(M: int, T: float, P: float) -> float:
    """
    ITR in bit/min. M classes, T seconds per decision, P accuracy in [0,1].
    """
    P = float(P)
    if P <= 0.0:
        # limit: P log P -> 0; (1-P) log((1-P)/(M-1)) -> log(1/(M-1))
        a = np.log2(M)
        c = np.log2(1.0 / (M - 1))
        return float(60.0 * (a + c) / T)

    if P >= 1.0:
        # perfect classifier: b=1*log2(1)=0, c=(0)*...=0
        return float(60.0 * np.log2(M) / T)

    a = np.log2(M)
    b = P * np.log2(P)
    c = (1.0 - P) * np.log2((1.0 - P) / (M - 1))
    return float(60.0 * (a + b + c) / T)


def _plot_subjects_plus_average_single_box(
    df_condition: pd.DataFrame,
    subjects: List[str],
    windows_ms: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
):
    """
    df_condition expected columns:
      - subject
      - win_size_ms
      - acc_mean, acc_std
      - itr_mean, itr_std
    """
    subjects = list(subjects)
    windows_ms = np.sort(np.asarray(windows_ms))
    nW = len(windows_ms)
    pos_map = {w: j for j, w in enumerate(windows_ms)}

    # aggregate across subjects (mean of subject-means; std across subjects)
    g = df_condition.groupby("win_size_ms")
    agg = g.agg(
        acc_mean=("acc_mean", "mean"),
        itr_mean=("itr_mean", "mean"),
    ).reset_index()

    acc_std_across = g["acc_mean"].apply(lambda v: np.std(v.values)).reset_index(name="acc_std")
    itr_std_across = g["itr_mean"].apply(lambda v: np.std(v.values)).reset_index(name="itr_std")
    agg = agg.merge(acc_std_across, on="win_size_ms").merge(itr_std_across, on="win_size_ms")
    agg = agg.set_index("win_size_ms").reindex(windows_ms).reset_index()

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

    blocks = subjects + ["Average"]
    n_blocks = len(blocks)

    gap = 1
    block = nW + gap

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.6))
    ax2 = ax.twinx()

    ax.grid(True, which="major", linewidth=0.35, alpha=0.20)
    ax2.grid(False)

    centers = []

    for bi, name in enumerate(blocks):
        start = bi * block

        face = "#f4f4f4" if (bi % 2 == 1) else "#ffffff"
        ax.axvspan(start - 0.5, start + nW - 0.5, color=face, zorder=0)

        if name != "Average":
            df_s = df_condition[df_condition["subject"] == name].sort_values("win_size_ms")
            xvals = df_s["win_size_ms"].to_numpy()
            acc = df_s["acc_mean"].to_numpy()
            acc_std = df_s["acc_std"].to_numpy()
            itr = df_s["itr_mean"].to_numpy()
            itr_std = df_s["itr_std"].to_numpy()
        else:
            xvals = agg["win_size_ms"].to_numpy()
            acc = agg["acc_mean"].to_numpy()
            acc_std = agg["acc_std"].to_numpy()
            itr = agg["itr_mean"].to_numpy()
            itr_std = agg["itr_std"].to_numpy()

        x_idx = np.array([pos_map.get(w, np.nan) for w in xvals], dtype=float)
        valid = ~np.isnan(x_idx)
        x_plot = start + x_idx[valid]

        # accuracy (%)
        ax.errorbar(
            x_plot,
            acc[valid],
            yerr=acc_std[valid],
            fmt="o-",
            capsize=2,
            markersize=3,
            color="blue",
            alpha=0.95,
            linewidth=0.5,
        )

        # ITR
        ax2.errorbar(
            x_plot,
            itr[valid],
            yerr=itr_std[valid],
            fmt="o-",
            capsize=2,
            markersize=3,
            color="red",
            alpha=0.85,
            linewidth=0.5,
        )

        centers.append(start + (nW - 1) / 2)

    # major labels at a subset of window sizes
    label_windows = [400, 800, 1200, 1600, 2000]
    label_windows = [w for w in windows_ms if w in label_windows]

    major_xticks, major_xlabels = [], []
    minor_xticks = []

    for bi in range(n_blocks):
        start = bi * block
        for j, w in enumerate(windows_ms):
            minor_xticks.append(start + j)
        for w in label_windows:
            j = np.where(windows_ms == w)[0][0]
            major_xticks.append(start + j)
            major_xlabels.append(str(int(w)))

    ax.xaxis.set_major_locator(FixedLocator(major_xticks))
    ax.xaxis.set_minor_locator(FixedLocator(minor_xticks))
    ax.set_xticklabels(major_xlabels, rotation=90, ha="center")
    ax.tick_params(axis="x", which="minor", length=2, width=0.5)
    ax.tick_params(axis="x", which="major", length=3, width=0.7)
    ax.set_xlim(-0.5, (n_blocks - 1) * block + nW - 0.5)
    ax.set_xlabel("Window size [ms]")

    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylabel("Accuracy (%)", color="blue")
    ax.tick_params(axis="y", which="both", colors="blue")
    ax.spines["left"].set_color("blue")

    # ITR axis scale: auto based on data (safer than hardcoding 200)
    itr_max = float(np.nanmax(df_condition["itr_mean"].to_numpy()))
    # itr_ylim = max(50.0, np.ceil((itr_max + 10.0) / 25.0) * 25.0)
    ax2.set_ylim(0, 200)
    ax2.yaxis.set_major_locator(MultipleLocator(25))
    ax2.set_ylabel("ITR (bit/min)", color="red")
    ax2.tick_params(axis="y", which="both", colors="red")
    ax2.spines["right"].set_color("red")

    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    y_top = ax.get_ylim()[1]
    for c, name in zip(centers, blocks):
        ax.text(c, y_top - 10.0, str(name), ha="center", va="bottom", fontsize=10)

    # ax.set_title(title, y=1.08)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)

    return fig


# ----------------------------- main -----------------------------


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--artifacts_dir",
        type=Path,
        default=None,
        help="Root artifacts folder (default: env SILENTWEAR_ARTIFACTS_DIR or ./artifacts)",
    )
    ap.add_argument(
        "--experiment",
        type=str,
        default="inter_session",
        choices=["inter_session"],
        help="ITR is computed from inter_session runs",
    )

    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])

    ap.add_argument("--model_name", type=str, required=True, help="e.g., speechnet")
    ap.add_argument(
        "--model_run",
        type=str,
        default=None,
        help="e.g., model_6; if omitted, uses latest model_<k> per folder.",
    )

    # windows selection
    ap.add_argument(
        "--model_name_id",
        type=str,
        default=None,
        help="e.g., w1400ms. If set, ignores --windows_s expansion.",
    )
    ap.add_argument(
        "--windows_s",
        nargs="*",
        type=float,
        default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        help="If model_name_id not set: either <start end> or explicit list. Default: 0.4..1.4 step 0.2",
    )
    ap.add_argument("--window_step_s", type=float, default=0.2)
    ap.add_argument("--max_window_ms", type=int, default=None, help="Optional cutoff (e.g., 1400)")

    # ITR settings
    ap.add_argument("--num_classes", type=int, default=9)

    # outputs
    ap.add_argument("--figures_dir", type=Path, default=None)

    args = ap.parse_args()

    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        env = os.environ.get("SILENTWEAR_ARTIFACTS_DIR", None)
        artifacts_dir = Path(env) if env else Path("./artifacts")

    figures_dir = args.figures_dir if args.figures_dir else (artifacts_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # model_name_ids
    if args.model_name_id:
        model_name_ids = [args.model_name_id]
        windows_tag = args.model_name_id
    else:
        windows_s = _expand_windows_s(args.windows_s, args.window_step_s)
        model_name_ids = [_model_name_id_from_window_s(w) for w in windows_s]
        windows_tag = f"{model_name_ids[0]}_to_{model_name_ids[-1]}"

    runs = _find_runs(
        artifacts_dir=artifacts_dir,
        subjects=args.subjects,
        conditions=args.conditions,
        model_name=args.model_name,
        model_name_ids=model_name_ids,
        model_run=args.model_run,
        experiment=args.experiment,
    )
    if len(runs) == 0:
        raise SystemExit(
            f"No runs found for model={args.model_name}, model_name_ids={model_name_ids}. "
            f"Check artifacts_dir={artifacts_dir}"
        )

    # Build a tidy table: one row per (sub, cond, window)
    records = []
    for r in runs:
        df = pd.read_csv(r.cv_summary_csv)
        bal_col = "balanced_accuracy"
        bal_vals = df[bal_col].astype(float).to_numpy()  # folds, [0..1]

        # infer win_ms from model_name_id: w1400ms -> 1400
        try:
            win_ms = int(r.model_name_id.replace("w", "").replace("ms", ""))
        except Exception:
            # fallback: if run_cfg has it; otherwise skip
            continue

        if args.max_window_ms is not None and win_ms > int(args.max_window_ms):
            continue

        T_sec = win_ms / 1000.0
        itrs = np.array(
            [_compute_itr(M=args.num_classes, T=T_sec, P=p) for p in bal_vals], dtype=float
        )

        records.append(
            {
                "subject": r.subject,
                "condition": r.condition,
                "win_size_ms": win_ms,
                "acc_vals": bal_vals,
                "acc_mean": float(np.mean(bal_vals) * 100.0),
                "acc_std": float(np.std(bal_vals) * 100.0),
                "itr_vals": itrs,
                "itr_mean": float(np.mean(itrs)),
                "itr_std": float(np.std(itrs)),
                "run_path": str(r.run_path),
                "model_run": r.model_run,
                "model_name_id": r.model_name_id,
            }
        )

    res = pd.DataFrame(records)
    if res.empty:
        raise SystemExit(
            "No usable runs found after filtering (maybe max_window_ms cut everything)."
        )

    # Ensure we have one row per (sub, cond, win)
    # If duplicates exist (e.g., multiple model_runs), keep the latest or chosen
    res = res.sort_values(["subject", "condition", "win_size_ms", "model_run"]).drop_duplicates(
        subset=["subject", "condition", "win_size_ms"], keep="last"
    )

    # Plot per condition
    for cond in args.conditions:
        dfc = res[res["condition"] == cond].copy()
        if dfc.empty:
            continue

        windows_ms = np.sort(dfc["win_size_ms"].unique())

        out_fig = figures_dir / f"itr_{args.model_name}_{cond}_{windows_tag}.pdf"
        _plot_subjects_plus_average_single_box(
            df_condition=dfc,
            subjects=args.subjects,
            windows_ms=windows_ms,
            title=f"{args.model_name} | {cond}",
            save_path=out_fig,
        )
        print(f"[SAVED] {out_fig}")

        # Print max stats (average across subjects)
        unique_windows = windows_ms
        avg_acc = []
        avg_itr = []

        for w in unique_windows:
            win_metrics = dfc[dfc["win_size_ms"] == w]
            # average across subjects (per-subject means)
            avg_acc.append(float(np.mean(win_metrics["acc_mean"].to_numpy())))
            avg_itr.append(float(np.mean(win_metrics["itr_mean"].to_numpy())))

        avg_acc = np.asarray(avg_acc)
        avg_itr = np.asarray(avg_itr)

        w_best_acc = unique_windows[int(np.argmax(avg_acc))]
        w_best_itr = unique_windows[int(np.argmax(avg_itr))]
        best_acc = np.max(avg_acc)
        print(f"\n======= Summary for condition={cond} =======")
        print(f"Best avg Accuracy: {best_acc:.2f}% at window={w_best_acc} ms")
        print(f"Best avg ITR:      {np.max(avg_itr):.2f} bit/min at window={w_best_itr} ms")

        # Find also variations at best itr
        acc_at_best_itr = avg_acc[int(np.argmax(avg_itr))]
        print(f"Accuracy at best ITR:      {acc_at_best_itr:.2f}%")
        reduction = (((best_acc - acc_at_best_itr) / best_acc)) * 100
        print(f"Accuracy reduction:        {reduction:.2f}%")


if __name__ == "__main__":
    main()
