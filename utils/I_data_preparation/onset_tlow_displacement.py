#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Displacement report for the lower threshold
===================================================================

Measures where the lower threshold (``t_low``) puts the onset in the trigger-free
onset detector. The detector carries two thresholds: ``t_high`` confirms that an
event is physiological, while ``t_low`` decides where its onset is placed through a
backward search from the confirmation sample. This script evaluates how much earlier
the onset lands by varying ``t_low``.

The script runs the detector once per ``t_low_ratio``, keeps the first event of every
cue box, and reports the displacement.

Read the metrics as follows:

* ``displacement`` — how much the onset moved with respect to the adopted ratio.
  A negative displacement means the onset moved earlier, which means the window opens
  closer to the true start of the articulation.
* ``cue boxes covered`` — how many cue boxes are covered by both the tested ratio
  and the adopted one. Only the cue boxes covered by *both* configurations enter a
  comparison, so a ratio that loses utterances is not credited with moving the onsets
  it dropped.

Usage
-----
    # the whole sentence corpus, adopted ratio against the three alternatives
    python utils/I_data_preparation/onset_tlow_displacement.py \\
        --data_dir data_sentences

    # one subject, and the per-recording table on disk
    python utils/I_data_preparation/onset_tlow_displacement.py \\
        --data_dir data_sentences --subjects S01 \\
        --out onset_tlow_displacement.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.I_data_preparation.experimental_config import RAW_AND_FILTERED_DIRNAME
from utils.I_data_preparation.onset_detection import (
    OnsetConfig,
    detect_events_in_dataframe,
    label_boxes,
    match_events_to_boxes,
)
from utils.I_data_preparation.onset_detection_report import discover


def first_onset_per_box(df: pd.DataFrame, cfg: OnsetConfig, boxes) -> dict:
    """{box index: onset sample of the first event assigned to it}."""
    events = detect_events_in_dataframe(df, cfg)
    matches = match_events_to_boxes(events, boxes, cfg.match_tolerance_s, cfg.fs)
    out = {}
    for ev, m in zip(events, matches):
        if m is not None and m not in out:
            out[m] = ev.onset
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--processed", default=RAW_AND_FILTERED_DIRNAME)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--reference_ratio", type=float, default=OnsetConfig.t_low_ratio,
                    help="The adopted ratio, against which the others are measured.")
    ap.add_argument("--ratios", nargs="*", type=float, default=[0.3, 0.5, 0.7],
                    help="Alternatives to compare against the reference.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = discover(args.data_dir, args.processed, args.subjects, args.conditions)
    ref_cfg = OnsetConfig(t_low_ratio=args.reference_ratio)
    print(f"{len(files)} recordings | reference t_low_ratio={args.reference_ratio} "
          f"(t_low={ref_cfg.t_low:.2f}) | alternatives {args.ratios}\n")

    rows = []
    for path in files:
        df = pd.DataFrame(pd.read_hdf(path, key="emg")).reset_index(drop=True)
        boxes = label_boxes(df)
        if not boxes:
            continue
        ref = first_onset_per_box(df, ref_cfg, boxes)
        subject, condition = path.parents[1].name, path.parent.name
        for r in args.ratios:
            alt = first_onset_per_box(df, OnsetConfig(t_low_ratio=r), boxes)
            shared = sorted(set(ref) & set(alt))
            if not shared:
                continue
            d_ms = np.array([(alt[b] - ref[b]) / ref_cfg.fs * 1000.0 for b in shared])
            rows.append({
                "subject": subject, "condition": condition, "recording": path.stem,
                "t_low_ratio": r, "n_shared_boxes": len(shared),
                "n_boxes": len(boxes),
                "displacement_median_ms": float(np.median(d_ms)),
                "displacement_p10_ms": float(np.percentile(d_ms, 10)),
                "displacement_p90_ms": float(np.percentile(d_ms, 90)),
                "frac_earlier": float(np.mean(d_ms < 0)),
            })
        print(f"  {subject}/{condition}/{path.stem}: done")

    table = pd.DataFrame(rows)
    if table.empty:
        print("\nNo comparable cue box found.")
        return

    print(f"\n{'ratio':>6} {'t_low':>6} {'median':>9} {'p10':>8} {'p90':>8} "
          f"{'earlier':>8} {'boxes':>7}")
    print("-" * 56)
    for r, g in table.groupby("t_low_ratio"):
        print(f"{r:>6.2f} {OnsetConfig(t_low_ratio=r).t_low:>6.2f} "
              f"{g['displacement_median_ms'].mean():>8.1f}ms "
              f"{g['displacement_p10_ms'].mean():>7.1f} "
              f"{g['displacement_p90_ms'].mean():>7.1f} "
              f"{g['frac_earlier'].mean():>7.0%} "
              f"{int(g['n_shared_boxes'].sum()):>7d}")
    print("\nNegative is earlier than the adopted ratio. 'boxes' is how many cue "
          "boxes both configurations covered, which is what the medians average over.")

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nPer-recording table: {args.out}")


if __name__ == "__main__":
    main()
