#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Summarize and visualize results from:
- Global experiments
- Inter-session experiments

Supports both metric modes, auto-detected from the cv_summary.csv columns:
- classification runs (accuracy / balanced_accuracy / confusion_matrix)
- closed-set recognition runs (free-char CTC; wer / balanced_wer / cer / balanced_cer);
  block-scatter plots show WER (sentences) or CER (words) instead of accuracy, and
  confusion-matrix plots are skipped (predictions are reference/hypothesis strings).

New artifacts layout (paper wrapper compatible):
  <ARTIFACTS_DIR>/models/<experiment>/<subject>/<condition>/<model_name>/<model_name_id>/model_<k>/
  or when pooled:
  <ARTIFACTS_DIR>/models/<experiment>/all_subjects/<condition>/<model_name>/<model_name_id>/model_<k>/

Outputs:
  <ARTIFACTS_DIR>/tables/{model}_{model_run or latest}_{condition}_{model_name_id}_{experiment}.csv
  <ARTIFACTS_DIR>/figures/{model}_{model_run or latest}_{condition}_{model_name_id}_{experiment}_cm.svg

Examples:
Global @ 1400ms (Subject-Specific):
  python utils/III_results_analysis/I_global_intersession_analysis.py \
    --artifacts_dir ./artifacts \
    --experiment global \
    --model_name random_forest \
    --model_name_id w1400ms

Global @ 1400ms (Pooled All Subjects):
  python utils/III_results_analysis/I_global_intersession_analysis.py \
    --artifacts_dir ./artifacts \
    --experiment global \
    --model_name random_forest \
    --model_name_id w1400ms \
    --pool_subjects
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import ConfusionMatrixDisplay
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.I_data_preparation.experimental_config import get_active_labels
from utils.III_results_analysis.general_utils import build_twin_axis_blocks

CM_LABEL_MODE = "both"   # "text", "code", or "both"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class RunRef:
    subject: str
    condition: str
    model_name: str
    model_name_id: str
    model_run: str  # e.g. model_6
    run_path: Path  # .../model_6
    cv_summary_csv: Path
    run_cfg_json: Path


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
    """
    Return latest model_<k> in folder, based on max k.
    """
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
    experiment: str,
    subjects: List[str],
    conditions: List[str],
    model_name: str,
    model_name_ids: List[str],
    model_run: Optional[str],
) -> List[RunRef]:
    """
    Scan:
      artifacts/models/<experiment>/<subject>/<condition>/<model_name>/<model_name_id>/<model_run>/
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
                cfg = run_path / "run_cfg.json"
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
                        run_cfg_json=cfg,
                    )
                )
    return out


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------


def _parse_cm_cell(x) -> np.ndarray:
    """
    confusion matrix cell might be:
      - JSON string of list-of-lists
      - python literal string
      - already list
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        return arr
    if pd.isna(x):
        raise ValueError("NaN confusion matrix entry")

    s = str(x).strip()
    try:
        obj = json.loads(s)
    except Exception:
        import ast

        obj = ast.literal_eval(s)
    return np.asarray(obj, dtype=float)


def mean_std_confusion_matrices(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    mats = [_parse_cm_cell(v) for v in series.values]
    stack = np.stack(mats, axis=0)  # [fold, i, j]
    return stack.mean(axis=0), stack.std(axis=0)


# ---------------------------------------------------------------------------
# Task, metrics and labels
# ---------------------------------------------------------------------------


def _read_label_mode_from_run_cfg(run_cfg_path: Path) -> str:
    """Read label_mode from a saved run_cfg.json, defaulting to 'word'."""
    if run_cfg_path.exists():
        try:
            cfg = json.loads(run_cfg_path.read_text())
            return cfg.get("experimental_settings", {}).get("label_mode", "word")
        except Exception:
            pass
    return "word"


# Metric columns written by compute_wer_metrics for closed-set recognition runs
RECOGNITION_METRICS = [
    "wer", "balanced_wer", "cer", "balanced_cer", "vocab_wer", "balanced_vocab_wer",
]
CLASSIFICATION_METRICS = ["balanced_accuracy", "unbalanced_accuracy"]


def _detect_metrics_mode(df: pd.DataFrame) -> Optional[str]:
    """Detect which task a cv_summary.csv belongs to from its metric columns.

    Returns 'classification' (accuracy/confusion-matrix metrics), 'recognition'
    (WER/CER metrics from the free-character CTC decoder), or None if neither
    set of columns is present.
    """
    if "balanced_accuracy" in df.columns:
        return "classification"
    if "wer" in df.columns:
        return "recognition"
    return None


def _make_cm_labels(n_classes: int, mode: str, label_mode: str = "text") -> list[str]:
    """Build tick labels for confusion matrix axes."""
    active_labels = get_active_labels(label_mode)
    labels = []
    for i in range(n_classes):
        text = active_labels.get(i, str(i))
        if mode == "code":
            labels.append(str(i))
        elif mode == "text":
            labels.append(text)
        else:
            labels.append(f"{i} {text}")
    return labels

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--artifacts_dir",
        type=Path,
        default=None,
        help="Root artifacts folder (default: env SILENTWEAR_ARTIFACTS_DIR or ./artifacts)",
    )
    ap.add_argument("--experiment", type=str, choices=["global", "inter_session"], required=True)

    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    
    ap.add_argument(
        "--pool_subjects",
        action="store_true",
        help="Look for pooled 'all_subjects' models instead of individual subject folders.",
    )

    ap.add_argument(
        "--model_name", type=str, required=True, help="e.g., speechnet or random_forest"
    )
    ap.add_argument(
        "--model_run",
        type=str,
        default=None,
        help="e.g., model_6. If omitted, uses latest model_<k> per folder.",
    )

    # Window selection
    ap.add_argument(
        "--model_name_id",
        type=str,
        default=None,
        help="e.g., w1400ms. If provided, overrides --windows_s expansion.",
    )
    ap.add_argument(
        "--windows_s",
        nargs="*",
        type=float,
        default=[],
        help="If model_name_id not set: either <start end> or explicit list. Default: 0.4..1.4 step 0.2",
    )
    ap.add_argument("--window_step_s", type=float, default=0.2)

    # Outputs
    ap.add_argument("--tables_dir", type=Path, default=None)
    ap.add_argument("--figures_dir", type=Path, default=None)
    
    # Plots
    ap.add_argument("--plot_confusion_matrix", action="store_true")
    ap.add_argument("--plot_block_scatter", action="store_true", help="Plot accuracy block plots per condition across session folds")
    ap.add_argument(
        "--transparent", action="store_true", help="Save figures with transparent background"
    )

    args = ap.parse_args()

    if args.pool_subjects:
        args.subjects = ["all_subjects"]

    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        env = os.environ.get("SILENTWEAR_ARTIFACTS_DIR", None)
        artifacts_dir = Path(env) if env else Path("./artifacts")

    tables_dir = args.tables_dir if args.tables_dir else (artifacts_dir / "tables")
    figures_dir = args.figures_dir if args.figures_dir else (artifacts_dir / "figures")
    cm_figures_dir = figures_dir / "confusion_matrices"
    block_figures_dir = figures_dir / "intersession_block_scatters"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm_figures_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_block_scatter:
        block_figures_dir.mkdir(parents=True, exist_ok=True)

    # Determine model_name_ids (windows)
    if args.model_name_id:
        model_name_ids = [args.model_name_id]
    else:
        windows = _expand_windows_s(args.windows_s, args.window_step_s)
        model_name_ids = [_model_name_id_from_window_s(w) for w in windows]

    # Scan runs
    runs = _find_runs(
        artifacts_dir=artifacts_dir,
        experiment=args.experiment,
        subjects=args.subjects,
        conditions=args.conditions,
        model_name=args.model_name,
        model_name_ids=model_name_ids,
        model_run=args.model_run,
    )

    if len(runs) == 0:
        print(
            f"[WARN] No runs found for experiment={args.experiment}, model={args.model_name}, "
            f"subjects={args.subjects}, model_name_ids={model_name_ids}. Check artifacts_dir={artifacts_dir}"
        )
        return

    # Group runs by (model_name_id, condition)
    by_mid_cond: Dict[Tuple[str, str], List[RunRef]] = defaultdict(list)
    for r in runs:
        by_mid_cond[(r.model_name_id, r.condition)].append(r)

    model_run_tag = args.model_run if args.model_run else "latest"

    # For each window + condition, build per-subject summary and save CSV
    for (mid, cond), run_list in sorted(by_mid_cond.items(), key=lambda x: (x[0][0], x[0][1])):
        print("\n" + "=" * 110)
        print(
            f"Experiment: {args.experiment} | Model: {args.model_name} | model_name_id: {mid} | Condition: {cond}"
        )
        print("=" * 110)

        rows = []
        row_modes = []
        # Keep deterministic subject ordering
        for sub in args.subjects:
            rr: List[RunRef] = [r for r in run_list if r.subject == sub]
            if len(rr) == 0:
                continue
            if len(rr) > 1:
                # if multiple matches, pick the one with the largest model_k or first
                rr = sorted(
                    rr,
                    key=lambda x: (
                        int(x.model_run.split("_")[-1]) if x.model_run.startswith("model_") else -1
                    ),
                )
            r = rr[-1]

            df = pd.read_csv(r.cv_summary_csv)
            metrics_mode = _detect_metrics_mode(df)
            if metrics_mode is None:
                print(f"[WARN] {r.cv_summary_csv} has no known metric columns. Skipping {sub}.")
                continue

            row = {
                "subject": sub,
                "condition": cond,
                "model_name": args.model_name,
                "model_name_id": mid,
                "model_run": r.model_run,
                "run_path": str(r.run_path),
            }
            if metrics_mode == "classification":
                for df_col, out_col in [("balanced_accuracy", "balanced_accuracy"), ("accuracy", "unbalanced_accuracy")]:
                    if df_col in df.columns:
                        vals = df[df_col].astype(float).to_numpy()
                        row.update(
                            {
                                f"{out_col}_mean": float(np.mean(vals)),
                                f"{out_col}_std": float(np.std(vals)),
                                f"{out_col}_vals": json.dumps(vals.tolist()),
                            }
                        )
            else:
                for col in [c for c in RECOGNITION_METRICS if c in df.columns]:
                    vals = df[col].astype(float).to_numpy()
                    row.update(
                        {
                            f"{col}_mean": float(np.mean(vals)),
                            f"{col}_std": float(np.std(vals)),
                            f"{col}_vals": json.dumps(vals.tolist()),
                        }
                    )
            rows.append(row)
            row_modes.append(metrics_mode)

        if len(rows) == 0:
            print(f"[WARN] No valid subject data found for {mid} / {cond}")
            continue

        if len(set(row_modes)) > 1:
            print(
                f"[WARN] Mixed classification/recognition runs under {mid} / {cond}; "
                f"keeping only '{row_modes[0]}' rows."
            )
            rows = [row for row, m in zip(rows, row_modes) if m == row_modes[0]]
        group_mode = row_modes[0]

        summary_subjects = pd.DataFrame(rows)

        metric_group_cols = CLASSIFICATION_METRICS if group_mode == "classification" else RECOGNITION_METRICS
        metric_cols = [c for c in metric_group_cols if f"{c}_vals" in summary_subjects.columns]

        for col in metric_cols:
            mean_std_fmt = []
            for _, row in summary_subjects.iterrows():
                vals = np.asarray(json.loads(row[f"{col}_vals"]), dtype=float)
                mean = np.round(np.mean(vals) * 100, 1)
                std = np.round(np.std(vals) * 100, 1)
                mean_std_fmt.append(f"{mean}±{std}")
            summary_subjects[f"{col}_mean_std_perc"] = mean_std_fmt

        # Add All row (mean/std of per-subject means)
        all_row = {
            "subject": "All",
            "condition": cond,
            "model_name": args.model_name,
            "model_name_id": mid,
            "model_run": model_run_tag,
            "run_path": "",
        }
        for col in metric_cols:
            all_means = summary_subjects[f"{col}_mean"].to_numpy(dtype=float)
            all_row[f"{col}_mean"] = float(np.mean(all_means))
            all_row[f"{col}_std"] = float(np.std(all_means))
            all_row[f"{col}_vals"] = ""
            all_row[f"{col}_mean_std_perc"] = (
                f"{np.round(np.mean(all_means)*100, 2)}±{np.round(np.std(all_means)*100, 2)}"
            )

        # Do not append "All" row when evaluating a single pooled model (all_subjects)
        if not args.pool_subjects and len(summary_subjects) > 1:
            summary_subjects = pd.concat([summary_subjects, pd.DataFrame([all_row])], ignore_index=True)

        # Save CSV
        out_csv = (
            tables_dir / f"{args.model_name}_{model_run_tag}_{cond}_{mid}_{args.experiment}.csv"
        )
        summary_subjects.to_csv(out_csv, index=False)

        print_cols = [
            "subject",
            # "model_run"
            ] + [
            f"{c}_mean_std_perc"
            for c in metric_group_cols
            if f"{c}_mean_std_perc" in summary_subjects.columns
        ]
        # to_string avoids pandas truncating the recognition metric columns
        print(summary_subjects[print_cols].to_string())
        print(f"\n[SAVED] {out_csv}")

    # Combined session fold scatter block plot layout
    if args.plot_block_scatter:
        if args.experiment != "inter_session":
            print("[WARN] --plot_block_scatter is only supported for 'inter_session' experiments. Skipping.")
        else:
            by_mid_only: Dict[str, Dict[str, List[RunRef]]] = defaultdict(lambda: defaultdict(list))
            for r in runs:
                by_mid_only[r.model_name_id][r.condition].append(r)

            for mid, runs_by_cond in sorted(by_mid_only.items()):
                for cond in args.conditions:
                    run_list_cond: List[RunRef] = runs_by_cond.get(cond, [])
                    if not run_list_cond:
                        continue

                    y1_by_sub: Dict[str, np.ndarray] = {}
                    y2_by_sub: Dict[str, np.ndarray] = {}
                    max_folds = 0
                    valid_subjects = []
                    label_mode_detected = "word"
                    group_mode: Optional[str] = None
                    recog_metric_key = "wer"   # 'wer' (sentence) or 'cer' (word)

                    for sub in args.subjects:
                        rr: List[RunRef] = [r for r in run_list_cond if r.subject == sub]
                        if not rr:
                            continue
                        r = sorted(rr, key=lambda x: (int(x.model_run.split("_")[-1]) if x.model_run.startswith("model_") else -1))[-1]

                        df = pd.read_csv(r.cv_summary_csv)
                        metrics_mode = _detect_metrics_mode(df)
                        if metrics_mode == "classification":
                            if "accuracy" not in df.columns:
                                continue
                            y1 = df["balanced_accuracy"].astype(float).to_numpy() * 100.0
                            y2 = df["accuracy"].astype(float).to_numpy() * 100.0
                        elif metrics_mode == "recognition":
                            # WER degenerates to ~exact-match on single-word refs,
                            # so plot CER for word-mode runs and WER for sentences.
                            lm = _read_label_mode_from_run_cfg(r.run_cfg_json)
                            recog_metric_key = "cer" if lm == "word" else "wer"
                            bal_col, unbal_col = f"balanced_{recog_metric_key}", recog_metric_key
                            if bal_col not in df.columns or unbal_col not in df.columns:
                                continue
                            y1 = df[bal_col].astype(float).to_numpy() * 100.0
                            y2 = df[unbal_col].astype(float).to_numpy() * 100.0
                        else:
                            continue

                        if group_mode is None:
                            group_mode = metrics_mode
                        elif metrics_mode != group_mode:
                            print(f"[WARN] Mixed metric modes in block scatter for {mid} / {cond}; skipping {sub}.")
                            continue

                        label_mode_detected = _read_label_mode_from_run_cfg(r.run_cfg_json)

                        if len(y1) > max_folds:
                            max_folds = len(y1)

                        y1_by_sub[sub] = y1
                        y2_by_sub[sub] = y2
                        valid_subjects.append(sub)

                    if not valid_subjects:
                        continue

                    if group_mode == "recognition":
                        mlabel = recog_metric_key.upper()   # "WER" or "CER"
                        s1_name, s2_name = f"Balanced {mlabel}", f"Unbalanced {mlabel}"
                        y1_label, y2_label = f"Balanced {mlabel} (%)", f"Unbalanced {mlabel} (%)"
                        max_val = max(float(np.max(v)) for d in (y1_by_sub, y2_by_sub) for v in d.values())
                        y_max = max(100.0, math.ceil(max_val / 10.0) * 10.0)
                    else:
                        s1_name, s2_name = "Balanced Acc", "Unbalanced Acc"
                        y1_label, y2_label = "Balanced Accuracy (%)", "Unbalanced Accuracy (%)"
                        y_max = 100.0

                    series_bal: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
                        s1_name: {sub: {"mean": v, "std": np.zeros_like(v)} for sub, v in y1_by_sub.items()}
                    }
                    series_unbal: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
                        s2_name: {sub: {"mean": v, "std": np.zeros_like(v)} for sub, v in y2_by_sub.items()}
                    }

                    # Dynamic figure instantiation
                    n_blocks = len(valid_subjects) + (0 if args.pool_subjects else 1)
                    fig, ax1 = plt.subplots(1, 1, figsize=(1.8 * max(1, n_blocks), 3.2))
                    ax2 = ax1.twinx()
                    x_values = np.arange(1, max_folds + 1)

                    s1_styles = {s1_name: {"color": "blue", "marker": "o", "alpha": 0.95}}
                    s2_styles = {s2_name: {"color": "red", "marker": "o", "alpha": 0.85}}

                    build_twin_axis_blocks(
                        ax1=ax1, ax2=ax2, subjects=valid_subjects, x_values=x_values,
                        series1=series_bal, series2=series_unbal, series1_styles=s1_styles, series2_styles=s2_styles,
                        with_average=(not args.pool_subjects and len(valid_subjects) > 1), x_label="Fold (Session)", y1_label=y1_label, y2_label=y2_label,
                        y1_lim=(0, y_max), y2_lim=(0, y_max), y1_major_step=10, y2_major_step=10,
                        x_tick_labels=[str(x) for x in x_values], gap=1.0, block_margin=0.5,
                        label_y_offset=10.0, label_fontsize=10, label_position="bottom"
                    )
                    # Title dynamically appends data target configuration type
                    mode_title = "Sentences" if label_mode_detected == "sentence" else "Words"
                    ax1.set_title(f"Inter-session Performance ({mode_title}) | {cond.capitalize()} | {args.model_name}", pad=20)
                    
                    out_filename = block_figures_dir / f"block_plot_{cond}_{args.model_name}_{mid}.png"
                    plt.tight_layout()
                    fig.savefig(out_filename, bbox_inches="tight", dpi=300, transparent=args.transparent)
                    plt.close(fig)
                    print(f"[SAVED BLOCK PLOT] {out_filename}")

    # Confusion matrices: one figure per mid, all conditions side by side
    if args.plot_confusion_matrix:
        by_mid: Dict[str, Dict[str, List[RunRef]]] = defaultdict(lambda: defaultdict(list))
        for r in runs:
            by_mid[r.model_name_id][r.condition].append(r)

        for mid, runs_by_cond in sorted(by_mid.items()):
            target_conds = [c for c in args.conditions if c in runs_by_cond]
            nconds = len(target_conds)
            if not target_conds:
                continue

            # Recognition runs have no confusion matrix (predictions are strings)
            probe_df = pd.read_csv(runs_by_cond[target_conds[0]][0].cv_summary_csv)
            if "confusion_matrix" not in probe_df.columns:
                print(
                    f"[WARN] No confusion_matrix column for model_name_id={mid} "
                    "(recognition run: WER/CER metrics, string predictions). "
                    "Skipping confusion-matrix plot."
                )
                continue

            # Subjects that appear in at least one condition, in args order
            active_subj_set = {r.subject for rl in runs_by_cond.values() for r in rl}
            target_subjs = [s for s in args.subjects if s in active_subj_set]
            n_subjs = len(target_subjs)
            if n_subjs == 0:
                continue

            ncols = min(2, n_subjs)
            nrows = math.ceil(n_subjs / max(1, ncols))

            # Load summary CSVs (already saved above) for title annotations
            summary_dict: Dict[str, pd.DataFrame] = {}
            for cond in target_conds:
                csv_path = tables_dir / f"{args.model_name}_{model_run_tag}_{cond}_{mid}_{args.experiment}.csv"
                if csv_path.exists():
                    summary_dict[cond] = pd.read_csv(csv_path)
            
            label_mode = _read_label_mode_from_run_cfg(runs_by_cond[target_conds[0]][0].run_cfg_json)            
            if label_mode == "sentence":
                fig = plt.figure(figsize=(5.5 * ncols * nconds, 7 * nrows))
            else:
                fig = plt.figure(figsize=(5 * ncols * nconds, 7 * nrows))
                
            exp_title = args.experiment.replace("_", " ").title()
            fig.suptitle(f"{exp_title} Evaluation", fontsize=24, y=1.02)

            pad_left = 0.08
            pad_right = 0.98
            pad_bottom = 0.18

            width_colorbar = 0.08
            
            if label_mode == "sentence":
                wspace_colorbar = 0.4
                wspace_between_conds = 0.4
                fontsize = 5
            else:
                wspace_colorbar = 0.2
                wspace_between_conds = 0.2
                fontsize = 7

            width_ratios = [width_colorbar, wspace_colorbar]
            for i in range(nconds):
                width_ratios.extend([1] * ncols)
                if i < nconds - 1:
                    width_ratios.append(wspace_between_conds)

            total_gs_cols = len(width_ratios)

            gs = gridspec.GridSpec(
                nrows, total_gs_cols,
                width_ratios=width_ratios,
                wspace=0.1, hspace=0.25,
                left=pad_left, right=pad_right, top=0.88, bottom=pad_bottom
            )

            total_ratio_sum = sum(width_ratios)
            for cond_idx, cond in enumerate(target_conds):
                start_ratio = width_colorbar + wspace_colorbar + cond_idx * (ncols * 1 + wspace_between_conds)
                center_ratio = start_ratio + (ncols / 2.0)
                center_x = pad_left + (center_ratio / total_ratio_sum) * (pad_right - pad_left)
                fig.text(center_x, 0.94, cond.capitalize(), ha="center", fontsize=20)

            for row in range(nrows):
                cax = fig.add_subplot(gs[row, 0])
                last_im = None

                for cond_idx, cond in enumerate(target_conds):
                    run_list_cond: List[RunRef] = runs_by_cond.get(cond, [])
                    for col_rel in range(ncols):
                        subj_idx = row * ncols + col_rel
                        if subj_idx >= n_subjs:
                            continue

                        sub = target_subjs[subj_idx]
                        col_abs = 2 + cond_idx * (ncols + 1) + col_rel
                        ax = fig.add_subplot(gs[row, col_abs])

                        rr: List[RunRef] = [r for r in run_list_cond if r.subject == sub]
                        if len(rr) == 0:
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title(sub, fontsize=14, pad=10)
                            continue

                        r = sorted(rr, key=lambda x: (int(x.model_run.split("_")[-1]) if x.model_run.startswith("model_") else -1))[-1]

                        df = pd.read_csv(r.cv_summary_csv)

                        cm_mean, cm_std = mean_std_confusion_matrices(df["confusion_matrix"])
                        n_classes = int(cm_mean.shape[0])
                        label_mode = _read_label_mode_from_run_cfg(r.run_cfg_json)
                        text_labels = _make_cm_labels(n_classes, CM_LABEL_MODE, label_mode=label_mode)

                        disp = ConfusionMatrixDisplay(confusion_matrix=cm_mean, display_labels=text_labels)
                        disp.plot(ax=ax, cmap="Blues", colorbar=False, include_values=True, values_format=".1f", text_kw={"fontsize": fontsize})

                        last_im = ax.images[0]
                        last_im.set_clim(0.0, 1.0)

                        bal_vals = df["balanced_accuracy"].to_numpy(dtype=float)
                        unbal_vals = df["accuracy"].to_numpy(dtype=float)
                        
                        bal_mean = np.mean(bal_vals) * 100
                        bal_std = np.std(bal_vals) * 100
                        std_mean = np.mean(unbal_vals) * 100
                        std_std = np.std(unbal_vals) * 100
                        
                        title = f"{sub} \n Bal: {bal_mean:.1f}±{bal_std:.1f}% \n Unbal: {std_mean:.1f}±{std_std:.1f}%"

                        ax.set_title(title, fontsize=16, pad=8)

                        if col_rel == 0:
                            ax.tick_params(axis="y", labelsize=10)
                            ax.set_yticklabels(text_labels, fontsize=10)
                        else:
                            ax.set_yticklabels([])
                        ax.set_ylabel("")

                        last_subj_idx = n_subjs - 1
                        last_row = last_subj_idx // ncols
                        last_col_rel = last_subj_idx % ncols

                        is_bottom = False
                        if row == last_row and col_rel <= last_col_rel:
                            is_bottom = True
                        elif row == last_row - 1 and col_rel > last_col_rel:
                            is_bottom = True

                        if is_bottom:
                            ax.tick_params(axis="x", labelrotation=45, labelsize=9)
                            ax.set_xticklabels(text_labels, ha="right", fontsize=9)
                        else:
                            ax.set_xticklabels([])
                        ax.set_xlabel("")

                if last_im is not None:
                    cbar = fig.colorbar(last_im, cax=cax)
                    cbar.ax.yaxis.set_ticks_position('left')
                    cbar.ax.yaxis.set_label_position('left')
                    cbar.set_label("Accuracy", fontsize=15, labelpad=10)

            out_fig_svg = (
                cm_figures_dir / f"{args.model_name}_{model_run_tag}_{mid}_{args.experiment}_cm.svg"
            )
            out_fig_png = out_fig_svg.with_suffix(".png")
            fig.savefig(out_fig_svg, bbox_inches="tight", transparent=args.transparent)
            fig.savefig(out_fig_png, bbox_inches="tight", dpi=300, transparent=args.transparent)
            plt.close(fig)
            print(f"[SAVED CM] {out_fig_svg}")
            print(f"[SAVED CM] {out_fig_png}")


if __name__ == "__main__":
    main()