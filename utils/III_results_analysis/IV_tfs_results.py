#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
IV_tfs_results.py

Print a compact train-from-scratch summary table (average across subjects), plus
an overall average accuracy across tested batches.

Example:
    python utils/III_results_analysis/IV_tfs_results.py \
    --artifacts_dir ./artifacts \
    --model_name emg_transformer \
    --model_base_id w1400ms \
    --bs_id bs_config_1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _pick_acc_col(df: pd.DataFrame) -> str:
    candidates = [
        "zero_shot_balanced_acc",
        "balanced_acc",
        "balanced_accuracy",
        "balanced_acc_test",
        "balanced_accuracy_test",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No accuracy column found. Available columns: {list(df.columns)}")


def _load_subject_curve(summary_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(summary_csv)

    if "num_prev_ft_rounds" not in df.columns:
        raise KeyError(f"Missing 'num_prev_ft_rounds' in {summary_csv}")

    acc_col = _pick_acc_col(df)

    # Keep compatibility with files that contain zero_shot_test_batch.
    if "zero_shot_test_batch" in df.columns:
        step1 = df.groupby(["num_prev_ft_rounds", "zero_shot_test_batch"], as_index=False)[[acc_col]].mean()
        grouped = step1.groupby("num_prev_ft_rounds", as_index=False)[[acc_col]].mean()
    else:
        grouped = df.groupby("num_prev_ft_rounds", as_index=False)[[acc_col]].mean()

    grouped = grouped.sort_values(by="num_prev_ft_rounds")

    rounds = grouped["num_prev_ft_rounds"].to_numpy(dtype=int)
    acc_perc = grouped[acc_col].to_numpy(dtype=float) * 100.0
    return rounds, acc_perc


def _align_common_rounds(curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    all_round_sets = [set(x.tolist()) for x, _ in curves.values()]
    common = sorted(set.intersection(*all_round_sets))
    if not common:
        raise ValueError("No common num_prev_ft_rounds across the selected subjects.")
    return np.asarray(common, dtype=int)


def _build_tfs_table(
    artifacts_dir: Path,
    model_name: str,
    model_base_id: str,
    bs_id: str,
    subjects: List[str],
    condition: str,
) -> Tuple[pd.DataFrame, float, List[str]]:
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    missing_subjects: List[str] = []

    for subject in subjects:
        summary_csv = (
            artifacts_dir
            / "models"
            / "train_from_scratch"
            / subject
            / condition
            / model_name
            / model_base_id
            / bs_id
            / "train_from_scratch_summary.csv"
        )

        if not summary_csv.exists():
            missing_subjects.append(subject)
            continue

        curves[subject] = _load_subject_curve(summary_csv)

    if not curves:
        raise FileNotFoundError(
            "No train_from_scratch_summary.csv found for the selected args. "
            f"condition={condition}, model={model_name}, model_base_id={model_base_id}, bs_id={bs_id}"
        )

    common_rounds = _align_common_rounds(curves)

    aligned_acc = []
    for subject in sorted(curves.keys()):
        rounds, acc = curves[subject]
        idx = np.array([np.where(rounds == r)[0][0] for r in common_rounds], dtype=int)
        aligned_acc.append(acc[idx])

    acc_stack = np.vstack(aligned_acc)
    means = acc_stack.mean(axis=0)
    stds = acc_stack.std(axis=0)

    table = pd.DataFrame(
        {
            "last_tested_batch": common_rounds + 1,
            "train_from_scratch_mean_acc": means,
            "train_from_scratch_std_acc": stds,
        }
    )
    table["train_from_scratch_mean_std"] = [
        f"{m:.2f} +/- {s:.2f}" for m, s in zip(means, stds)
    ]

    avg_accuracy = float(np.mean(means))
    return table, avg_accuracy, missing_subjects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print train-from-scratch summary table (average across subjects)."
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Root artifacts dir (default: env SILENTWEAR_ARTIFACTS_DIR or ./artifacts)",
    )
    parser.add_argument("--model_name", type=str, default="speechnet")
    parser.add_argument("--model_base_id", type=str, default="w1400ms")
    parser.add_argument("--bs_id", type=str, default="bs_config_0")
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    parser.add_argument("--conditions", nargs="+", default=["vocalized", "silent"])

    args = parser.parse_args()

    if args.artifacts_dir is not None:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = Path(os.environ.get("SILENTWEAR_ARTIFACTS_DIR", "./artifacts"))

    tables_dir = artifacts_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        table, avg_accuracy, missing = _build_tfs_table(
            artifacts_dir=artifacts_dir,
            model_name=args.model_name,
            model_base_id=args.model_base_id,
            bs_id=args.bs_id,
            subjects=args.subjects,
            condition=condition,
        )

        print(f"\nCondition: {condition}")
        if missing:
            print(f"[WARN] Missing subjects: {', '.join(missing)}")

        print("Train from scratch summary table (Average across subjects)")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"Average accuracy: {avg_accuracy:.2f}\n")

        out_csv = (
            tables_dir
            / f"tfs_summary_{condition}_{args.model_name}_{args.model_base_id}_{args.bs_id}.csv"
        )
        table.to_csv(out_csv, index=False)
        print(f"[SAVED] {out_csv}")


if __name__ == "__main__":
    main()
