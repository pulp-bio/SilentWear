#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Percentile analysis of the windows.
=================================================================================

Why this script exists
----------------------
The per-subject input normalization of ``offline_experiments/general_utils.py``
needs a scale per subject and per channel.
Transforms need a bound, and this script measures where to put it from 
the data rather than assuming a value. Point it at a windowed dataset and it
answers three questions:

1. **How dominant are the artefacts?**  The ratio between the largest absolute
   sample of a channel and its high percentiles.
2. **How rare are they?**  The fraction of samples far above the 99th percentile,
   which is the clip budget the threshold has to cover.
3. **Which estimator is reproducible?**  The coefficient of variation of each
   candidate scale across the sessions of one subject. A normalization constant
   that moves between sessions defeats the inter-session protocol.

It then recommends a percentile for ``experiment.normalization_percentile``
(min-max) and the equivalent clip in sigma units for
``experiment.normalization_clip_sigma`` (z-score).

Usage
-----
    python utils/II_feature_extraction/amplitude_percentile_analysis.py \\
        --data_dir data_sentences/data_sentences_5_subjects_6_sessions

    # one condition, a subset of subjects, and a CSV of the per-channel table
    python utils/II_feature_extraction/amplitude_percentile_analysis.py \\
        --data_dir data_sentences/data_sentences_5_subjects_6_sessions \\
        --conditions vocalized --subjects S01 S03 --out amplitude_stats.csv

The dataset is the one produced by ``win_feature_extraction_main.py``: the script
reads the windows under ``<data_dir>/<win_and_feats>/<subject>/<condition>/
WIN_<ms>/*.h5`` and uses only the filtered channel columns, the same columns the
models consume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------


CANDIDATE_PERCENTILES = [100.0, 99.9, 99.5, 99.0, 97.5, 95.0, 90.0]
ARTEFACT_FACTOR = 10.0
REQUIRED_MARGIN = 5.0


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def discover_window_files(
    data_dir: Path,
    win_and_feats: str,
    subjects: Optional[List[str]],
    conditions: Optional[List[str]],
) -> Dict[str, Dict[str, List[Path]]]:
    """Build a map of ``{subject: {condition: [h5 files]}}`` for the requested selection."""
    root = data_dir / win_and_feats
    if not root.is_dir():
        raise FileNotFoundError(
            f"No windowed data under '{root}'. Pass --win_and_feats if the "
            "dataset uses a different sub-directory name."
        )

    found: Dict[str, Dict[str, List[Path]]] = {}
    for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if subjects and sub_dir.name not in subjects:
            continue
        for cond_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
            if conditions and cond_dir.name not in conditions:
                continue
            files = sorted(cond_dir.rglob("*.h5"))
            if files:
                found.setdefault(sub_dir.name, {})[cond_dir.name] = files
    if not found:
        raise FileNotFoundError(f"No .h5 windows matched the selection under '{root}'.")
    return found


def channel_columns(df: pd.DataFrame) -> List[str]:
    """Extract the filtered channel columns, i.e. what the deep models actually read."""
    return [c for c in df.columns if c.startswith("Ch_") and c.endswith("_filt")]


def collect_stats(files_by_subject, percentiles: List[float]) -> pd.DataFrame:
    """Collect one row per subject/condition/session/channel with its amplitude scales."""
    rows = []
    for subject, per_cond in files_by_subject.items():
        for condition, files in per_cond.items():
            for path in files:
                frame = pd.read_hdf(path)
                if not isinstance(frame, pd.DataFrame):
                    print(f"  [skip] expected DataFrame in {path.name}")
                    continue
                cols = channel_columns(frame)
                if not cols:
                    print(f"  [skip] no filtered channel columns in {path.name}")
                    continue
                for col in cols:
                    windows = np.stack(
                        [np.asarray(v, dtype=np.float32).ravel() for v in frame[col].to_numpy()]
                    )
                    magnitude = np.abs(windows)
                    std = float(windows.std())
                    p99 = float(np.percentile(magnitude, 99.0))
                    row = {
                        "subject": subject,
                        "condition": condition,
                        "session": path.stem,
                        "channel": col,
                        "n_samples": int(magnitude.size),
                        "std": std,
                        "max": float(magnitude.max()),
                    }
                    for p in percentiles:
                        row[f"p{p:g}"] = float(np.percentile(magnitude, p))
                    row["artefact_frac_%"] = float(
                        (magnitude > ARTEFACT_FACTOR * p99).mean() * 100.0
                    )
                    row["win_with_artefact_%"] = float(
                        (magnitude.max(axis=1) > ARTEFACT_FACTOR * p99).mean() * 100.0
                    )
                    row["p99_in_sigma"] = p99 / std if std > 0 else np.nan
                    row["max_in_sigma"] = row["max"] / std if std > 0 else np.nan
                    rows.append(row)
                print(f"  {subject}/{condition}/{path.stem}: {len(cols)} channels", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def report_dominance(df: pd.DataFrame, percentiles: List[float]) -> None:
    print("\n" + "=" * 78)
    print("1. HOW DOMINANT ARE THE ARTEFACTS")
    print("=" * 78)
    print("\nRatio between the largest absolute sample of a channel and its percentiles.")
    print("A ratio of 1000 means true min-max scaling would squeeze 99% of the signal")
    print("into a thousandth of the output range.\n")

    ratios = pd.DataFrame({"subject": df["subject"]})
    for p in percentiles:
        if p < 100.0:
            ratios[f"max/p{p:g}"] = df["max"] / df[f"p{p:g}"]
    print("  median over channels, per subject:")
    print(ratios.groupby("subject").median().round(1).to_string())

    worst = df.assign(ratio=df["max"] / df["p99"]).nlargest(5, "ratio")
    print("\n  the five worst channels overall:")
    print(
        worst[["subject", "condition", "session", "channel", "p99", "max", "ratio"]]
        .round(1)
        .to_string(index=False)
    )


def report_contamination(df: pd.DataFrame) -> float:
    print("\n" + "=" * 78)
    print("2. HOW RARE ARE THEY")
    print("=" * 78)
    print(f"\nSamples above {ARTEFACT_FACTOR:g}x the channel's own 99th percentile.")
    print("This is the budget the clip must cover; everything else is genuine EMG.\n")

    per_subject = df.groupby("subject")
    summary = pd.DataFrame(
        {
            "typical channel (median)": per_subject["artefact_frac_%"].median(),
            "worst channel (max)": per_subject["artefact_frac_%"].max(),
            "windows hit (mean %)": per_subject["win_with_artefact_%"].mean(),
        }
    )
    print(summary.round(4).to_string())

    worst_channel = float(df["artefact_frac_%"].max())
    print(f"\n  worst channel anywhere in the dataset: {worst_channel:.4f}% of samples")
    return worst_channel


def report_reproducibility(df: pd.DataFrame, percentiles: List[float]) -> pd.Series:
    print("\n" + "=" * 78)
    print("3. WHICH ESTIMATOR IS REPRODUCIBLE ACROSS SESSIONS")
    print("=" * 78)
    print("\nCoefficient of variation of each scale over the sessions of one subject,")
    print("median over channels. A constant that moves between sessions defeats the")
    print("inter-session protocol, so lower is better.\n")

    estimators = ["std"] + [f"p{p:g}" for p in percentiles if p < 100.0] + ["max"]
    n_sessions = df.groupby(["subject", "condition"])["session"].nunique()
    if int(n_sessions.max()) < 2:
        print("  only one session per subject: skipped.")
        return pd.Series(dtype=float)

    cv = (
        df.groupby(["subject", "condition", "channel"])[estimators]
        .agg(lambda s: s.std() / s.mean() if s.mean() else np.nan)
        .groupby("subject")
        .median()
    )
    print(cv.round(3).to_string())
    mean_cv = cv.mean()
    print("\n  mean over subjects:")
    print(mean_cv.round(3).to_string())
    return mean_cv


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


def recommend(df: pd.DataFrame, worst_contamination: float, mean_cv: pd.Series,
              percentiles: List[float]) -> None:
    print("\n" + "=" * 78)
    print("4. RECOMMENDATION")
    print("=" * 78)
    print(
        f"\nA percentile p clips (100 - p)% of the samples. That budget must cover the "
        f"worst\nchannel's contamination ({worst_contamination:.4f}%) by at least "
        f"{REQUIRED_MARGIN:g}x, so the scale is set\nby EMG and not by the edge of the "
        "artefact population; among the percentiles\nthat qualify, the least clipping one "
        "preserves the most genuine dynamics.\n"
    )

    table = []
    for p in sorted((x for x in percentiles if x < 100.0), reverse=True):
        budget = 100.0 - p
        margin = budget / worst_contamination if worst_contamination > 0 else np.inf
        table.append(
            {
                "percentile": p,
                "clips %": budget,
                "margin over worst": margin,
                "covers artefacts": "yes" if margin >= REQUIRED_MARGIN else "NO",
                "session CV": float(mean_cv.get(f"p{p:g}", np.nan)),
            }
        )
    report = pd.DataFrame(table)
    print(report.round(3).to_string(index=False))

    qualifying = report[report["covers artefacts"] == "yes"]
    if qualifying.empty:
        print(
            "\n  No candidate covers the contamination with margin. The recording is "
            "\n  unusually contaminated: inspect the worst channels above before "
            "\n  normalizing, they may be broken rather than artefacted."
        )
        return

    chosen = float(qualifying.iloc[0]["percentile"])
    sigma = float(df["p99_in_sigma"].median())
    max_sigma = float(df["max_in_sigma"].max())
    if not np.isclose(chosen, 99.0):
        col = f"p{chosen:g}"
        sigma = float((df[col] / df["std"]).median()) if col in df else sigma

    print(f"\n  -> min-max:  normalization_percentile: {chosen:g}")
    print(f"     clips {100.0 - chosen:g}% of samples, "
          f"{qualifying.iloc[0]['margin over worst']:.1f}x the worst contamination.")
    print(f"\n  -> z-score:  normalization_clip_sigma: {sigma:.1f}")
    print(f"     the same percentile expressed in standard deviations. Without it the "
          f"largest\n     sample in this dataset enters the network at {max_sigma:.0f} sigma; "
          "the z-score rescales\n     but does not bound, so it needs the clip as much as "
          "min-max needs the percentile.")
    print("\n  Both keys live under 'experiment:' in the base config, next to")
    print("  data_normalization and normalization_kind.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Percentile analysis of window amplitudes, to choose a robust "
                    "normalization scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Dataset root, e.g. data_sentences/data_sentences_5_subjects_6_sessions")
    parser.add_argument("--win_and_feats", default="wins_and_features",
                        help="Sub-directory holding the windowed data.")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subject IDs to include (default: all found).")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Conditions to include, e.g. vocalized silent (default: all).")
    parser.add_argument("--percentiles", nargs="*", type=float, default=CANDIDATE_PERCENTILES,
                        help="Candidate percentiles to evaluate.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional CSV path for the full per-channel table.")
    args = parser.parse_args()

    percentiles = sorted(set(args.percentiles), reverse=True)

    print(f"Scanning '{args.data_dir}' ...")
    files_by_subject = discover_window_files(
        args.data_dir, args.win_and_feats, args.subjects, args.conditions
    )
    total = sum(len(f) for c in files_by_subject.values() for f in c.values())
    print(f"{len(files_by_subject)} subject(s), {total} recording(s).\n")

    df = collect_stats(files_by_subject, percentiles)
    if df.empty:
        print("No channel statistics collected.")
        return 1

    report_dominance(df, percentiles)
    worst_contamination = report_contamination(df)
    mean_cv = report_reproducibility(df, percentiles)
    recommend(df, worst_contamination, mean_cv, percentiles)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nPer-channel table -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
