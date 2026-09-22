#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Plot the detected windows against the trigger?
===============================================================

This script shows where each window was placed, compared with the trigger the
detector never read. The figures show how far the detected onset sits from the 
cue and whether that distance is consistent.

Three outputs, all written under ``--out_dir``:

``overview_<subject>_<condition>_sess<N>.png``
    A stretch of the recording with the trigger boxes shaded and the detected
    windows drawn on top, over the activity trace the detector thresholds. This
    is the figure to look at first: it shows at a glance whether windows land
    inside their box, how late, and what the false alarms look like.

``alignment_<subject>.png``
    Distribution of onset minus cue over all trials of the subject, per
    condition, with the median marked. A tight unimodal distribution at a
    positive lag is the expected shape.

``summary.csv``
    Per recording: number of trials, detection rate, median and spread of the
    lag, and how many windows would fall outside their cue box.

Usage
-----
    python utils/IV_plots/onset_vs_trigger_fig.py \\
        --data_dir data_sentences/data_sentences_5_subjects_6_sessions

    # one subject, and a longer stretch in the overview
    python utils/IV_plots/onset_vs_trigger_fig.py \\
        --data_dir data_sentences/data_sentences_5_subjects_6_sessions \\
        --subjects S01 --overview_s 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.I_data_preparation.experimental_config import RAW_AND_FILTERED_DIRNAME
from utils.I_data_preparation.onset_detection import (
    OnsetConfig,
    compute_activity,
    detect_events,
    filtered_channel_columns,
    label_boxes,
    match_events_to_boxes,
)


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------


SPEECH_C = "#1b6ca8"
DETECT_C = "#d1495b"
FA_C = "#8d6e63"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def discover(data_dir: Path, processed: str, subjects, conditions) -> List[Path]:
    root = data_dir / processed
    if not root.is_dir():
        raise FileNotFoundError(f"No processed recordings under '{root}'.")
    files: List[Path] = []
    for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if subjects and sub_dir.name not in subjects:
            continue
        for cond_dir in sorted(p for p in sub_dir.iterdir() if p.is_dir()):
            if conditions and cond_dir.name not in conditions:
                continue
            files.extend(sorted(cond_dir.glob("*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 recordings matched the selection under '{root}'.")
    return files


def analyse(path: Path, cfg: OnsetConfig):
    """Detected events, trigger boxes and their pairing for one recording."""
    df = pd.DataFrame(pd.read_hdf(path, key="emg")).reset_index(drop=True)
    X = df[filtered_channel_columns(df)].to_numpy(dtype=float)
    activity = compute_activity(X, cfg)
    events = detect_events(activity, cfg)
    boxes = label_boxes(df)
    matches = match_events_to_boxes(events, boxes, cfg.match_tolerance_s, cfg.fs)

    lags, claimed = [], set()
    for ev, bi in zip(events, matches):
        if bi is None or bi in claimed:
            continue
        claimed.add(bi)
        lags.append((ev.onset - boxes[bi][0]) / cfg.fs)
    return activity, events, boxes, matches, np.asarray(lags), len(df)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_overview(activity, events, boxes, matches, window_s, cfg, path, out, span_s):
    """Trigger boxes, detected windows and the activity trace on one time axis."""
    n_show = min(len(activity), int(span_s * cfg.fs))
    # Start at the first cue box so the figure is never a stretch of empty rest.
    start = boxes[0][0] - int(1.0 * cfg.fs) if boxes else 0
    start = max(0, min(start, len(activity) - n_show))
    stop = start + n_show
    t = np.arange(start, stop) / cfg.fs

    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.plot(t, activity[start:stop], lw=0.8, color="#333333", label="activity $a[n]$")
    ax.axhline(cfg.t_high, color=DETECT_C, ls="--", lw=1.0, label=r"$\tau_{high}$")
    ax.axhline(cfg.t_low, color=DETECT_C, ls=":", lw=1.0, label=r"$\tau_{low}$")

    for bs, be, _ in boxes:
        if be < start or bs > stop:
            continue
        ax.axvspan(bs / cfg.fs, be / cfg.fs, color=SPEECH_C, alpha=0.14, lw=0)

    if start < cfg.warmup_s * cfg.fs:
        ax.axvspan(start / cfg.fs, cfg.warmup_s, color="#9e9e9e", alpha=0.25, lw=0)
        ax.text(cfg.warmup_s / 2, 0.5, "baseline\nwarm-up", ha="center", va="bottom",
                fontsize=8, color="#555555")

    win = int(window_s * cfg.fs)
    top = float(np.nanpercentile(activity[start:stop], 99.5)) * 1.15
    for ev, bi in zip(events, matches):
        if ev.offset < start or ev.onset > stop:
            continue
        matched = bi is not None
        colour = DETECT_C if matched else FA_C
        ax.axvline(ev.onset / cfg.fs, color=colour, lw=1.1, alpha=0.75)
        ax.hlines(top, ev.onset / cfg.fs, (ev.onset + win) / cfg.fs,
                  color=colour, lw=3.2, alpha=0.85 if matched else 0.5)

    ax.set_xlim(start / cfg.fs, stop / cfg.fs)
    ax.set_ylim(0, top * 1.12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized activity")
    ax.set_title(
        f"{path.parents[1].name} / {path.parent.name} / {path.stem}   —   "
        f"shaded = trigger box (never read by the detector), "
        f"red = detected onset + {window_s:g} s window, brown = unmatched",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_alignment(lags_by_cond, subject, window_s, out):
    """Histogram of onset minus cue, per condition."""
    conds = [c for c, v in lags_by_cond.items() if len(v)]
    if not conds:
        return
    fig, axes = plt.subplots(1, len(conds), figsize=(5.5 * len(conds), 3.6), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        lags = lags_by_cond[cond]
        ax.hist(lags, bins=40, color=SPEECH_C, alpha=0.8)
        med = float(np.median(lags))
        ax.axvline(0, color="#333333", lw=1.2, label="cue")
        ax.axvline(med, color=DETECT_C, lw=1.6, label=f"median {med:+.3f} s")
        ax.set_xlabel("Detected onset − cue [s]")
        ax.set_ylabel("Trials")
        ax.set_title(f"{subject} / {cond}  (n={len(lags)})", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"Window start vs. cue ({window_s:g} s windows)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--processed", default=RAW_AND_FILTERED_DIRNAME)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--window_s", type=float, default=2.4,
                    help="Window length drawn on the overview; match the extraction.")
    ap.add_argument("--overview_s", type=float, default=30.0,
                    help="Seconds of recording shown in the overview figure.")
    ap.add_argument("--overview_session", type=int, default=1,
                    help="Only recordings of this session get an overview figure.")
    ap.add_argument("--out_dir", type=Path, default=Path("./windowing_check/onset_vs_trigger"))
    ap.add_argument("--t_high", type=float, default=None)
    ap.add_argument("--fmt", default="png", choices=("png", "pdf"),
                    help="Figure format. Use pdf for the thesis figures.")
    args = ap.parse_args()

    cfg = OnsetConfig(**({"t_high": args.t_high} if args.t_high else {}))
    files = discover(args.data_dir, args.processed, args.subjects, args.conditions)
    print(f"Analysing {len(files)} recordings | window {args.window_s:g} s | "
          f"t_high={cfg.t_high} t_low={cfg.t_low:.2f}\n")

    rows = []
    lags_by_subject: dict = {}
    for path in files:
        subject, condition = path.parents[1].name, path.parent.name
        activity, events, boxes, matches, lags, n = analyse(path, cfg)
        if not boxes:
            continue

        box_len = float(np.median([be - bs for bs, be, _ in boxes])) / cfg.fs
        outside = float(np.mean(lags + args.window_s > box_len)) if len(lags) else float("nan")

        rows.append({
            "subject": subject, "condition": condition, "recording": path.stem,
            "trials": len(boxes), "detected": len(lags),
            "detection_rate": len(lags) / len(boxes),
            "lag_median_s": float(np.median(lags)) if len(lags) else np.nan,
            "lag_p10_s": float(np.percentile(lags, 10)) if len(lags) else np.nan,
            "lag_p90_s": float(np.percentile(lags, 90)) if len(lags) else np.nan,
            "lag_iqr_s": float(np.subtract(*np.percentile(lags, [75, 25]))) if len(lags) else np.nan,
            "window_past_box_frac": outside,
        })
        lags_by_subject.setdefault(subject, {}).setdefault(condition, []).extend(lags.tolist())

        if f"sess_{args.overview_session}_" in path.stem:
            plot_overview(
                activity, events, boxes, matches, args.window_s, cfg, path,
                args.out_dir / f"overview_{subject}_{condition}_{path.stem}.{args.fmt}",
                args.overview_s,
            )
            print(f"  overview: {subject}/{condition}/{path.stem}")

    for subject, by_cond in lags_by_subject.items():
        plot_alignment({c: np.asarray(v) for c, v in by_cond.items()},
                       subject, args.window_s,
                       args.out_dir / f"alignment_{subject}.{args.fmt}")
        print(f"  alignment: {subject}")

    table = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "summary.csv", index=False)

    print("\n" + "=" * 78)
    print("Detected onset relative to the cue (positive = after the cue)")
    print("=" * 78)
    print(f"{'subject':<9} {'cond':<10} {'det':>6} {'lag med':>9} {'p10..p90':>16} {'IQR':>7} {'win past box':>13}")
    for (sub, cond), g in table.groupby(["subject", "condition"]):
        print(f"{sub:<9} {cond:<10} {g['detection_rate'].mean():>5.0%} "
              f"{g['lag_median_s'].median():>+9.3f} "
              f"{g['lag_p10_s'].median():>+7.3f}..{g['lag_p90_s'].median():>+6.3f} "
              f"{g['lag_iqr_s'].median():>7.3f} {g['window_past_box_frac'].mean():>12.0%}")
    print("-" * 78)
    print(f"{'ALL':<9} {'':<10} {table['detection_rate'].mean():>5.0%} "
          f"{table['lag_median_s'].median():>+9.3f} "
          f"{table['lag_p10_s'].median():>+7.3f}..{table['lag_p90_s'].median():>+6.3f} "
          f"{table['lag_iqr_s'].median():>7.3f} {table['window_past_box_frac'].mean():>12.0%}")
    print(f"\nFigures and summary.csv written to {args.out_dir}")


if __name__ == "__main__":
    main()
