#!/usr/bin/env python3
"""
Summarize and visualize results from:
- Global experiments
- Inter-session experiments

New artifacts layout (paper wrapper compatible):
  <ARTIFACTS_DIR>/models/<experiment>/<subject>/<condition>/<model_name>/<model_name_id>/model_<k>/
    - cv_summary.csv
    - run_cfg.json

Outputs:
  <ARTIFACTS_DIR>/tables/{model}_{model_run or latest}_{condition}_{model_name_id}_{experiment}.csv
  <ARTIFACTS_DIR>/figures/{model}_{model_run or latest}_{condition}_{model_name_id}_{experiment}_cm.svg

Examples:


Global @ 1400ms:
  python utils/III_results_analysis/I_global_intersession_analysis.py \
    --artifacts_dir ./artifacts \
    --experiment global \
    --model_name random_forest \
    --model_name_id w1400ms
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]   
sys.path.insert(0, str(PROJECT_ROOT))
from utils.I_data_preparation.experimental_config import ORIGINAL_LABELS


# ----------------------------- helpers -----------------------------

@dataclass
class RunRef:
    subject: str
    condition: str
    model_name: str
    model_name_id: str
    model_run: str              # e.g. model_6
    run_path: Path              # .../model_6
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


def _pick_bal_acc_col(df: pd.DataFrame) -> str:
    """
    Try a few likely names from your trainers.
    """
    candidates = [
        "balanced_accuracy",
        "balanced_acc",
        "balanced_acc_test",
        "balanced_accuracy_test",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find balanced accuracy column. Available: {list(df.columns)}")


def _pick_cm_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "confusion_matrix",
        "confusion_matrix_test",
        "cm",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


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


def _infer_display_labels(run_cfg_path: Path, fallback_n: int) -> List[str]:
    """
    Best-effort label extraction:
    - If run_cfg has base_cfg with label mapping, use it.
    - Else fallback to class indices.
    """
    if run_cfg_path.exists():
        try:
            cfg = json.loads(run_cfg_path.read_text())
            base = cfg.get("base_cfg", {})
            # common patterns if you stored it somewhere:
            for key in ["train_label_map", "label_map", "labels_map", "train_labels_map"]:
                m = base.get(key, None)
                if isinstance(m, dict) and len(m) > 0:
                    # dict values are display labels
                    return [str(v) for v in m.values()]
            # sometimes stored at top-level
            for key in ["train_label_map", "label_map"]:
                m = cfg.get(key, None)
                if isinstance(m, dict) and len(m) > 0:
                    return [str(v) for v in m.values()]
        except Exception:
            pass

    return [str(i) for i in range(fallback_n)]


# ----------------------------- main analysis -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--artifacts_dir", type=Path, default=None,
                    help="Root artifacts folder (default: env SILENTWEAR_ARTIFACTS_DIR or ./artifacts)")
    ap.add_argument("--experiment", type=str, choices=["global", "inter_session"], required=True)

    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])

    ap.add_argument("--model_name", type=str, required=True, help="e.g., speechnet or random_forest")
    ap.add_argument("--model_run", type=str, default=None,
                    help="e.g., model_6. If omitted, uses latest model_<k> per folder.")

    # Window selection
    ap.add_argument("--model_name_id", type=str, default=None,
                    help="e.g., w1400ms. If provided, overrides --windows_s expansion.")
    ap.add_argument("--windows_s", nargs="*", type=float, default=[],
                    help="If model_name_id not set: either <start end> or explicit list. Default: 0.4..1.4 step 0.2")
    ap.add_argument("--window_step_s", type=float, default=0.2)

    # Outputs
    ap.add_argument("--tables_dir", type=Path, default=None)
    ap.add_argument("--figures_dir", type=Path, default=None)

    ap.add_argument("--plot_confusion_matrix", action="store_true")
    ap.add_argument("--transparent", action="store_true", help="Save figures with transparent background")

    args = ap.parse_args()

    artifacts_dir = args.artifacts_dir
    if artifacts_dir is None:
        env = os.environ.get("SILENTWEAR_ARTIFACTS_DIR", None)
        artifacts_dir = Path(env) if env else Path("./artifacts")

    tables_dir = args.tables_dir if args.tables_dir else (artifacts_dir / "tables")
    figures_dir = args.figures_dir if args.figures_dir else (artifacts_dir / "figures")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
        raise SystemExit(
            f"No runs found for experiment={args.experiment}, model={args.model_name}, "
            f"model_name_ids={model_name_ids}. Check artifacts_dir={artifacts_dir}"
        )

    # Group runs by (model_name_id, condition)
    by_mid_cond: Dict[Tuple[str, str], List[RunRef]] = {}
    for r in runs:
        by_mid_cond.setdefault((r.model_name_id, r.condition), []).append(r)

    # For each window + condition, build per-subject summary + (optional) confusion matrices
    for (mid, cond), run_list in sorted(by_mid_cond.items(), key=lambda x: (x[0][0], x[0][1])):
        print("\n" + "=" * 90)
        print(f"Experiment: {args.experiment} | Model: {args.model_name} | model_name_id: {mid} | Condition: {cond}")
        print("=" * 90)

        rows = []
        # Keep deterministic subject ordering
        for sub in args.subjects:
            rr = [r for r in run_list if r.subject == sub]
            if len(rr) == 0:
                continue
            if len(rr) > 1:
                # if multiple matches, pick the one with the largest model_k or first
                rr = sorted(rr, key=lambda x: int(x.model_run.split("_")[-1]) if x.model_run.startswith("model_") else -1)
            r = rr[-1]

            df = pd.read_csv(r.cv_summary_csv)
            bal_col = _pick_bal_acc_col(df)
            bal_vals = df[bal_col].astype(float).to_numpy()

            rows.append({
                "subject": sub,
                "condition": cond,
                "model_name": args.model_name,
                "model_name_id": mid,
                "model_run": r.model_run,
                "run_path": str(r.run_path),
                "balanced_acc_mean": float(np.mean(bal_vals)),
                "balanced_acc_std": float(np.std(bal_vals)),
                "balanced_acc_vals": json.dumps(bal_vals.tolist()),
            })

        if len(rows) == 0:
            print(f"[WARN] No subjects found for {mid} / {cond}")
            continue

        summary_subjects = pd.DataFrame(rows)

        # Pretty mean±std (%)
        mean_std_fmt = []
        for _, row in summary_subjects.iterrows():
            vals = np.asarray(json.loads(row["balanced_acc_vals"]), dtype=float)
            mean = np.round(np.mean(vals) * 100, 1)
            std = np.round(np.std(vals) * 100, 1)
            mean_std_fmt.append(f"{mean}±{std}")
        summary_subjects["mean_std_perc"] = mean_std_fmt

        # Add All row (mean/std of per-subject means)
        all_means = summary_subjects["balanced_acc_mean"].to_numpy(dtype=float)
        all_row = {
            "subject": "All",
            "condition": cond,
            "model_name": args.model_name,
            "model_name_id": mid,
            "model_run": (args.model_run if args.model_run else "latest"),
            "run_path": "",
            "balanced_acc_mean": float(np.mean(all_means)),
            "balanced_acc_std": float(np.std(all_means)),
            "balanced_acc_vals": "",
            "mean_std_perc": f"{np.round(np.mean(all_means)*100, 2)}±{np.round(np.std(all_means)*100, 2)}",
        }
        summary_subjects = pd.concat([summary_subjects, pd.DataFrame([all_row])], ignore_index=True)

        # Save CSV
        model_run_tag = args.model_run if args.model_run else "latest"
        out_csv = tables_dir / f"{args.model_name}_{model_run_tag}_{cond}_{mid}_{args.experiment}.csv"
        summary_subjects.to_csv(out_csv, index=False)
        print(summary_subjects[["subject", "mean_std_perc"]])
        print(f"[SAVED] {out_csv}")

        # Confusion matrices (2x2)
        if args.plot_confusion_matrix:
            n_subj = len(args.subjects)
            nrows, ncols = 2, 2
            fig, axs = plt.subplots(
                nrows, ncols,
                figsize=(10, 4.5 * nrows),
                sharex=True, sharey=True,
                constrained_layout=False
            )
            fig.subplots_adjust(
                left=0.12, right=0.98, top=0.92, bottom=0.10,
                wspace=0.08, hspace=0.25
            )
            axs = np.atleast_2d(axs)

            row_images = {}

            for idx, sub in enumerate(args.subjects):
                row = idx // 2
                col = idx % 2
                ax = axs[row, col]

                rr = [r for r in run_list if r.subject == sub]
                if len(rr) == 0:
                    ax.axis("off")
                    continue
                r = rr[-1]

                df = pd.read_csv(r.cv_summary_csv)
                cm_col = _pick_cm_col(df)
                if cm_col is None:
                    ax.set_title(f"{sub} | (no CM in cv_summary.csv)", fontsize=16)
                    ax.axis("off")
                    continue

                cm_mean, cm_std = mean_std_confusion_matrices(df[cm_col])


                disp_labels = list(ORIGINAL_LABELS.values())
                print(disp_labels)

                disp = ConfusionMatrixDisplay(confusion_matrix=cm_mean, display_labels=disp_labels)
                disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False, include_values=False)

                # Force consistent scale
                im = ax.images[0]
                im.set_clim(0.0, 1.0)

                # title: subject | mean±std balanced acc
                subj_row = summary_subjects[summary_subjects["subject"] == sub]
                if len(subj_row) > 0:
                    title = f"{sub} | {subj_row['mean_std_perc'].iloc[0]}"
                else:
                    title = sub
                ax.set_title(title, fontsize=20)
                ax.tick_params(axis="x", labelrotation=45, labelsize=15)
                ax.set_xticklabels(ax.get_xticklabels(), ha="right")
                ax.tick_params(axis="y", labelsize=15)
                ax.set_xlabel("")
                ax.set_ylabel("")

                # store image for row colorbar
                if row not in row_images:
                    row_images[row] = ax.images[0]

                # # annotate mean±std per cell
                # for (i, j), m in np.ndenumerate(cm_mean):
                #     s = cm_std[i, j]
                #     ax.text(j, i, f"{m:.2f}\n±{s:.2f}", ha="center", va="center", fontsize=10)

            # turn off unused axes if fewer than 4 subjects
            for k in range(n_subj, nrows * ncols):
                axs.flatten()[k].axis("off")

            # one colorbar per row (left)
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

            out_fig = figures_dir / f"{args.model_name}_{model_run_tag}_{cond}_{mid}_{args.experiment}_cm_2x2.svg"
            plt.savefig(out_fig, bbox_inches="tight", transparent=args.transparent)
            plt.close(fig)
            print(f"[SAVED] {out_fig}")


if __name__ == "__main__":
    main()
