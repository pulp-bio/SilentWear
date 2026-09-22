#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Plotting Script for Data Augmentation Ablation Studies

Reads the outputs directly from the folder structure generated
by Data_Augmentation_Ablation_Trainer.py.

Expected folder hierarchy

data_augmentation_ablation:
  {args.artifacts_dir}/{run_label}/{n}_sess/models/{exp}/{subject}/{condition}/{net_id}/{window_id}/{model_run}/cv_summary.csv

If multiple subjects are present, each subject is plotted as a separate
horizontal block. An "Average" block is appended automatically when more 
than one subject is available. With a single subject no average block is 
shown and no subject label appears in the figure title.
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from general_utils import (
    resolve_csv_path,
    model_run_tag,
    window_from_path,
    build_save_path,
    build_multi_subject_blocks,
)

_VARIANT_STYLES_DEFAULT: Dict[str, Dict[str, Any]] = {
    "No augmentation": {"color": "#b3b3b3", "marker": "o"},
    "2 strides":       {"color": "#377eb8", "marker": "o"},
    "5 strides":       {"color": "#ff7f00", "marker": "o"},
    "10 strides":      {"color": "#4daf4a", "marker": "o"},
}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _parse_run_label(run_label: Optional[str]) -> dict:
    """Parses the run_label to extract augmentation parameters."""
    if run_label is None:
        return {"variant_base": "Unknown", "stride_ms": np.nan, "num_strides": np.nan}
    if run_label.startswith("baseline"):
        return {"variant_base": "No augmentation", "stride_ms": np.nan, "num_strides": np.nan}
    
    match = re.match(r"stride(?P<stride_ms>\d+)_n(?P<num_strides>\d+)", run_label)
    if match:
        return {
            "variant_base": "Augmented",
            "stride_ms": float(match.group("stride_ms")),
            "num_strides": float(match.group("num_strides"))
        }
    return {"variant_base": run_label, "stride_ms": np.nan, "num_strides": np.nan}


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

            # run_label is 3 levels above experiment_name in the augmentation tree
            run_label = parts[exp_idx - 3]
            parsed_label = _parse_run_label(run_label)

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
            record.update(parsed_label)
            records.append(record)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df_res = pd.DataFrame(records)
    
    unique_strides = df_res["stride_ms"].dropna().unique()
    unique_nums = df_res["num_strides"].dropna().unique()

    def make_variant(row):
        if row["variant_base"] != "Augmented":
            return row["variant_base"]
        
        # Constant stride, variable Num strides
        if len(unique_strides) == 1 and len(unique_nums) > 1:
            return f"{int(row['num_strides'])} strides"
        # Constant num strides, variable Stride
        elif len(unique_nums) == 1 and len(unique_strides) > 1:
            return f"{int(row['stride_ms'])} ms stride"
        # Both variable or both constant (but not "No augmentation")
        else:
            return f"{int(row['num_strides'])} str, {int(row['stride_ms'])} ms"

    df_res["variant"] = df_res.apply(make_variant, axis=1)

    subset = ["subject", "condition", "session_count", "variant"]
    df_res = df_res.sort_values(
        "model_run_tag",
        key=lambda s: s.apply(
            lambda x: int(x.split("_")[-1]) if re.match(r"model_\d+", x) else -1
        ),
    )
    return df_res.drop_duplicates(subset=subset, keep="last")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _build_variant_styles(variant_order: List[str]) -> Dict[str, Dict[str, Any]]:
    """Assigns a plotting style (color, marker) to each variant based on the provided order."""
    palette = plt.colormaps["tab10"]
    styles: Dict[str, Dict[str, Any]] = {}
    for i, v in enumerate(variant_order):
        styles[v] = _VARIANT_STYLES_DEFAULT.get(
            v, {"color": palette(i % 10), "marker": "o"}
        )
    return styles


def _sorted_variants(variants) -> List[str]:
    """Sorts variants with a preference for "No augmentation" first, then by number of strides if applicable, then alphabetically."""
    preferred = ["No augmentation"]
    order = [v for v in preferred if v in variants]
    others = sorted(
        [v for v in variants if v not in order],
        key=lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 999,
    )
    return order + others


def plot_data_augmentation(
    df: pd.DataFrame,
    experiment_name: str,
    save_path: Path,
    model_run: Optional[str] = None,
):
    """
    For each (condition, window_size) pair produce one figure.

    The horizontal axis is divided into one block per subject; if multiple
    subjects are present an additional "Average" block is appended on the
    right.  Within each block the x-axis represents session_count values and
    each augmentation variant is a separate coloured line.
    """
    conditions   = sorted(df["condition"].unique())
    windows      = sorted(df["window_s"].unique())
    subjects     = sorted(df["subject"].unique())
    is_acc       = bool(df["is_acc"].iloc[0])
    multi_subject = len(subjects) > 1

    tags   = df["model_run_tag"].unique()
    mr_tag = model_run if model_run is not None else (tags[0] if len(tags) == 1 else "latest")

    variant_order  = _sorted_variants(df["variant"].unique())
    variant_styles = _build_variant_styles(variant_order)

    unique_strides = df["stride_ms"].dropna().unique()
    unique_nums = df["num_strides"].dropna().unique() 
    title_suffix = ""
    file_suffix = ""
    if len(unique_strides) == 1 and len(unique_nums) > 1:
        title_suffix = f" | Stride Size = {int(unique_strides[0])} ms"
        file_suffix = f"_stride{int(unique_strides[0])}ms"
    elif len(unique_nums) == 1 and len(unique_strides) > 1:
        title_suffix = f" | Num Strides = {int(unique_nums[0])}"
        file_suffix = f"_n{int(unique_nums[0])}strides"

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
    })

    n_rows = len(conditions)

    for w in windows:
        df_w = df[df["window_s"] == w]

        # Collect all session counts present for this window (shared x-axis)
        session_counts = np.array(sorted(df_w["session_count"].unique()), dtype=float)
        nX = len(session_counts)

        n_blocks = len(subjects) + (1 if multi_subject else 0)
        fig_w = max(6.0, 1.6 * nX * n_blocks + 1.0)
        fig_h = 7.0 * n_rows

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(fig_w, fig_h),
            sharex=True, sharey=True,
            squeeze=False,
        )

        title = f"Data Augmentation Ablation ({experiment_name}){title_suffix} | Window = {w:g} s"
        fig.suptitle(title, y=1.02, fontsize=13, fontweight="bold")

        for row_idx, cond in enumerate(conditions):
            df_cond = df_w[df_w["condition"] == cond]
            ax = axes[row_idx][0]

            ax.set_title(cond.capitalize(), fontsize=11, fontweight="bold")
            ax.set_ylabel("Accuracy (%)" if is_acc else "Metric")

            # Build the `series` dict expected by build_multi_subject_blocks:
            #   series[variant_name][subject_name] = {"mean": array, "std": array}
            series: Dict[str, Any] = {}
            for variant in variant_order:
                subj_data: Dict[str, Dict[str, np.ndarray]] = {}
                for subj in subjects:
                    df_sv = (
                        df_cond[
                            (df_cond["variant"] == variant)
                            & (df_cond["subject"] == subj)
                        ]
                        .sort_values("session_count")
                    )
                    if df_sv.empty:
                        continue
                    means = np.full(nX, np.nan)
                    stds  = np.full(nX, np.nan)
                    sc_to_idx = {float(sc): i for i, sc in enumerate(session_counts)}
                    for _, row in df_sv.iterrows():
                        idx = sc_to_idx.get(float(row["session_count"]))
                        if idx is not None:
                            means[idx] = row["metric_mean"]
                            stds[idx]  = row["metric_std"]
                    subj_data[subj] = {"mean": means, "std": stds}
                if subj_data:
                    series[variant] = subj_data

            x_tick_labels = [str(int(sc)) for sc in session_counts]

            build_multi_subject_blocks(
                ax=ax,
                subjects=subjects,
                x_values=session_counts,
                series=series,
                with_average=multi_subject,
                series_styles=variant_styles,
                x_label="Sessions considered" if row_idx == n_rows - 1 else "",
                y_label="Accuracy (%)" if is_acc else "Metric",
                y_lim=(0, 100) if is_acc else None,
                y_major_step=10.0 if is_acc else None,
                x_tick_labels=x_tick_labels,
                gap=0.0,
                block_margin=0.5,
                label_y_offset=8.0,
                label_fontsize=9,
                x_tick_rotation=0,
            )

        # Single shared legend above the first subplot
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            axes[0][0].legend(
                handles, labels,
                loc="lower center",
                ncol=len(handles),
                frameon=False,
                bbox_to_anchor=(0.5, 1.18),
                fontsize=9,
            )

        plt.tight_layout()
        w_ms   = int(w * 1000)
        suffix = f"{file_suffix}_w{w_ms}ms{save_path.suffix}"
        plt.savefig(
            build_save_path(save_path, mr_tag, suffix),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", type=Path, required=True)
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

    # Check if the user directly passed a specific ablation folder 
    if args.artifacts_dir.name.startswith("data_augmentation_ablation"):
        ablation_roots = [args.artifacts_dir]
    else:
        possible_names = [
            "data_augmentation_ablation",
            "data_augmentation_ablation_stride_dim",
            "data_augmentation_ablation_num_strides"
        ]
        ablation_roots = [args.artifacts_dir / name for name in possible_names if (args.artifacts_dir / name).exists() and (args.artifacts_dir / name).is_dir()]
        
        if not ablation_roots:
            print(f"[ERROR] No ablation directory (e.g., data_augmentation_ablation) found in {args.artifacts_dir}")
            return

    for ablation_root in ablation_roots:
        print(f"\n{'='*60}")
        print(f"[PROCESSING DIRECTORY] {ablation_root.name}")
        print(f"{'='*60}")

        for exp_name in args.experiment:
            print(f"\n[PLOT] Scanning for data_augmentation | '{exp_name}'...")

            df_results = collect_ablation_data(ablation_root, exp_name, model_run=args.model_run)

            if df_results.empty:
                print(f"[SKIP] No data found for '{exp_name}' in {ablation_root.name}.")
                continue

            # Plots will be saved inside the specific analyzed folder
            figures_dir = ablation_root / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            out_fig = figures_dir / f"data_augmentation_{exp_name}_summary.png"
            plot_data_augmentation(df_results, exp_name, out_fig, model_run=args.model_run)

            print(f"[SAVED] Generated plots in {figures_dir}")


if __name__ == "__main__":
    main()