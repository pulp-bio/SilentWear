#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Report for the trigger-free onset detector
===================================================================

Scores ``utils/I_data_preparation/onset_detection.py`` against the trigger boxes
of the processed recordings. The detector never sees the trigger; it is used here
only as ground truth, so the numbers say how much would be lost by cutting the
windows without it.

Read the columns as follows:

* ``det`` — how many cue boxes received at least one onset, as a percentage and
  as the plain count. Whatever is missing is the set of utterances that would
  simply not appear in the onset-aligned training set.
* ``FA`` — how many detected events overlap no cue box at all. Swallows and
  movements, mostly. An utterance split in two is not counted here, so this is
  spurious events only, not detection mistakes.
* ``dur p95`` — how long the detected utterances last, from onset to offset.
  "p95" means 95% of them are shorter than this, so a window of that length
  fits all but the longest 5%. It is the number to pick ``window_size_s`` from:
  choose less and you cut the tail off more than 5% of the utterances.

The closing block reads the same events as a **binary speech-against-rest
classifier**, pooling the 2x2 table over the recordings before deriving the
rates. Two units are reported, because the two answer different questions:

* **segment** — one trigger segment is one decision. A cue box is a positive and
  a rest span a negative, which is the granularity the windowing stage works at.
  A false positive is a rest span the detector wakes up in.
* **sample** — one sample is one decision, so the table also charges the part of
  a box the detector leaves uncovered. Recall is bounded above here: a box opens
  at the cue and the articulation starts a few hundred milliseconds later, and
  the samples in between are counted as misses although nothing is said in them.

Usage
-----
    python utils/I_data_preparation/onset_detection_report.py \\
        --data_dir data_sentences

    # one subject, all its sessions, and a CSV of the per-recording table
    python utils/I_data_preparation/onset_detection_report.py \\
        --data_dir data_sentences \\
        --subjects S01 --out onset_report.csv

    # try a different operating point before committing to an extraction
    python utils/I_data_preparation/onset_detection_report.py \\
        --data_dir data_sentences \\
        --t_high 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.I_data_preparation.experimental_config import RAW_AND_FILTERED_DIRNAME
from utils.I_data_preparation.onset_detection import (
    OnsetConfig,
    binary_rates_from_counts,
    detect_events_in_dataframe,
    label_boxes,
    score_against_trigger,
)


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


def print_binary_block(table: pd.DataFrame) -> None:
    """Print the detector evaluation as a binary speech-against-rest classifier.

    The 2x2 tables are pooled over the recordings and the rates derived from the
    pooled counts, so a short recording does not weigh as much as a long one.
    """
    print("\n" + "=" * 72)
    print("The detector as a binary speech / rest classifier")
    print("=" * 72)

    for prefix, unit, note in (
        ("seg", "segment", "one cue box or one rest span is one decision"),
        ("smp", "sample", "one sample is one decision, over the whole recording"),
    ):
        tp, fp = int(table[f"{prefix}_tp"].sum()), int(table[f"{prefix}_fp"].sum())
        fn, tn = int(table[f"{prefix}_fn"].sum()), int(table[f"{prefix}_tn"].sum())
        m = binary_rates_from_counts(tp, fp, fn, tn)
        n = tp + fp + fn + tn

        print(f"\n-- per {unit} ({note}) --")
        print(f"{'':>14}{'pred speech':>14}{'pred rest':>12}")
        print(f"{'true speech':>14}{tp:>14,}{fn:>12,}")
        print(f"{'true rest':>14}{fp:>14,}{tn:>12,}")
        print(f"   {n:,} decisions, {(tp + fn) / n:.1%} of them speech")
        print(
            f"   recall (sensitivity) {m['recall']:.1%} | "
            f"specificity {m['specificity']:.1%} | precision {m['precision']:.1%}"
        )
        print(
            f"   F1 {m['f1']:.3f} | balanced accuracy {m['balanced_accuracy']:.1%} | "
            f"accuracy {m['accuracy']:.1%} | MCC {m['mcc']:.3f}"
        )
        print(f"   false positive rate {m['false_positive_rate']:.2%}")
        worst_recall = table[f"{prefix}_recall"].min()
        best_recall = table[f"{prefix}_recall"].max()
        print(
            f"   per-recording recall spans {worst_recall:.1%} to {best_recall:.1%}, "
            f"specificity {table[f'{prefix}_specificity'].min():.1%} to "
            f"{table[f'{prefix}_specificity'].max():.1%}"
        )

    box_s = table["box_duration_median_s"].median()
    spoken_s = table["duration_median_s"].median()
    print(
        f"\nA cue box lasts {box_s:.2f} s and the articulation inside it {spoken_s:.2f} s "
        f"(median detected event), so a detector covering the speech exactly would score "
        f"about {spoken_s / box_s:.0%} sample recall. The rest spans between two boxes "
        f"last {table['rest_span_median_s'].median():.2f} s."
    )
    print(
        f"The detector flags {table['flagged_frac'].mean():.1%} of the recording as "
        f"speech, against the {table['smp_tp'].sum() + table['smp_fn'].sum():,} samples "
        f"({(table['smp_tp'].sum() + table['smp_fn'].sum()) / (table['smp_tp'].sum() + table['smp_fp'].sum() + table['smp_fn'].sum() + table['smp_tn'].sum()):.1%}) "
        f"the cue boxes cover. False positives cost "
        f"{table['false_alarms_per_min'].mean():.2f} spurious events per minute."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--processed", default=RAW_AND_FILTERED_DIRNAME)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--t_high", type=float, default=None, help="Override OnsetConfig.t_high.")
    ap.add_argument("--t_low_ratio", type=float, default=None)
    ap.add_argument("--topk", type=int, default=None,
                    help="Override OnsetConfig.topk, the channels averaged per sample.")
    ap.add_argument("--out", type=Path, default=None, help="Write the per-recording table to CSV.")
    args = ap.parse_args()

    overrides = {}
    if args.t_high is not None:
        overrides["t_high"] = args.t_high
    if args.t_low_ratio is not None:
        overrides["t_low_ratio"] = args.t_low_ratio
    if args.topk is not None:
        overrides["topk"] = args.topk
    cfg = OnsetConfig(**overrides)

    files = discover(args.data_dir, args.processed, args.subjects, args.conditions)
    print(f"Scoring {len(files)} recordings | t_high={cfg.t_high} t_low={cfg.t_low:.2f} "
          f"topk={cfg.topk} min_dur={cfg.min_duration_s}s\n")

    header = f"{'recording':<34} {'det':>18} {'FA':>6} {'dur p95':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for path in files:
        df = pd.read_hdf(path, key="emg")
        df = pd.DataFrame(df).reset_index(drop=True)
        events = detect_events_in_dataframe(df, cfg)
        boxes = label_boxes(df)
        stats = score_against_trigger(events, boxes, len(df), fs=cfg.fs)

        subject, condition = path.parents[1].name, path.parent.name
        name = f"{subject}/{condition}/{path.stem}"
        rows.append({"subject": subject, "condition": condition, "recording": path.stem, **stats})
        det = f"{stats['detection_rate']:.0%} ({stats['n_detected']}/{stats['n_boxes']})"
        print(f"{name:<34} {det:>18} {stats['false_alarms']:>6} {stats['duration_p95_s']:>9.2f}")

    table = pd.DataFrame(rows)
    print("-" * len(header))
    n_det, n_box = int(table["n_detected"].sum()), int(table["n_boxes"].sum())
    total = f"{n_det / n_box:.0%} ({n_det}/{n_box})" if n_box else "n/a"
    print(
        f"{'TOTAL':<34} {total:>18} {int(table['false_alarms'].sum()):>6} "
        f"{table['duration_p95_s'].mean():>9.2f}"
    )
    print(f"\n{n_box - n_det} of {n_box} utterances were not detected "
          f"({(n_box - n_det) / n_box:.1%})." if n_box else "")
    print(
        f"\nA 2.0 s cue-anchored window truncates "
        f"{table['truncated_at_2000ms_frac'].mean():.1%} of the detected trials, "
        f"a 2.4 s one {table['truncated_at_2400ms_frac'].mean():.1%}; "
        f"the span from cue to offset has a median of "
        f"{table['span_from_cue_median_s'].mean():.2f} s "
        f"and a 90th percentile of {table['span_from_cue_p90_s'].mean():.2f} s."
    )
    print(
        f"Events matching a cue box against those matching none: "
        f"peak activity {table['peak_matched_median'].mean():.2f} against "
        f"{table['peak_unmatched_median'].mean():.2f}, duration "
        f"{table['duration_matched_median_s'].mean():.2f} s against "
        f"{table['duration_unmatched_median_s'].mean():.2f} s "
        f"({int(table['n_unmatched_events'].sum())} unmatched events in total)."
    )

    print_binary_block(table)

    worst = table.loc[table["detection_rate"].idxmin()]
    print(
        f"\nWorst recording: {worst['subject']}/{worst['condition']}/{worst['recording']} "
        f"at {worst['detection_rate']:.0%} detection."
    )
    print(
        f"Suggested window_size_s to cover the 95th percentile of utterances: "
        f"{np.ceil(table['duration_p95_s'].mean() * 10) / 10:.1f} s "
        f"(per-recording p95 ranges {table['duration_p95_s'].min():.2f}-"
        f"{table['duration_p95_s'].max():.2f} s)."
    )

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nPer-recording table written to {args.out}")


if __name__ == "__main__":
    main()
