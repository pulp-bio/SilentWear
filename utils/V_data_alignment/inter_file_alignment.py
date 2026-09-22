# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Synchronize signals from two independent .bio files without altering raw data.
Assumes both files contain the same trigger sequence (same codes, same order).

Usage:
    python inter_file_alignment.py file_a.bio file_b.bio output_dir [--debug]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.V_data_alignment.align_bio_signals import align_bio_signals
from utils.V_data_alignment.write_bio_file import write_bio_file


def extract_trigger_times(signals: dict) -> np.ndarray:
    """Extract the absolute software timestamps of trigger rising edges."""
    trigger = signals["trigger"]["data"].reshape(-1)
    timestamp = signals["timestamp"]["data"].reshape(-1)

    prev = np.concatenate([[0], trigger[:-1]])
    onset_idx = np.where((trigger != 0) & (prev == 0))[0]

    return timestamp[onset_idx]


def synchronize_bio_files(signals_a: dict, signals_b: dict, debug: bool = False) -> dict:
    """
    Aligns File B to File A using matched trigger timestamps.
    The clock drift is interpolated between consecutive trigger pairs.
    """
    times_a = extract_trigger_times(signals_a)
    times_b = extract_trigger_times(signals_b)

    if len(times_a) != len(times_b):
        raise ValueError(
            f"Trigger count mismatch: File A has {len(times_a)}, File B has {len(times_b)}."
        )

    # Local clock offset at each trigger pair: how much B is ahead/behind A
    local_offsets = times_a - times_b

    if debug:
        print(f"[DEBUG][INTER] Trigger counts: A={len(times_a)}, B={len(times_b)}")
        if len(times_a) > 0:
            print(
                "[DEBUG][INTER] Offset computed at rising edges before alignment (A-B): "
                f"min={local_offsets.min():.3f}s, "
                f"max={local_offsets.max():.3f}s, "
                f"mean={local_offsets.mean():.3f}s"
            )

    signals_b_aligned = {}
    for name, sig in signals_b.items():
        aligned_sig = {"fs": sig["fs"], "data": sig["data"].copy()}

        if name == "timestamp":
            ts_b = aligned_sig["data"].reshape(-1)

            # Interpolate the offset across all samples; clamp outside range
            interpolated_offsets = np.interp(ts_b, times_b, local_offsets)
            aligned_sig["data"] = (ts_b + interpolated_offsets).reshape(-1, 1)

        signals_b_aligned[name] = aligned_sig

    if debug and len(times_a) > 0:
        times_b_aligned = extract_trigger_times(signals_b_aligned)
        offsets_after = times_a - times_b_aligned

        print(
            "[DEBUG][INTER] Residual offset computed at rising edges after alignment (A-B_aligned): "
            f"min={offsets_after.min():.3f}s, "
            f"max={offsets_after.max():.3f}s, "
            f"mean={offsets_after.mean():.3f}s, "
        )

        ts_a = signals_a["timestamp"]["data"].reshape(-1)
        ts_b_aligned = signals_b_aligned["timestamp"]["data"].reshape(-1)
        aligned_non_decreasing = bool(np.all(np.diff(ts_b_aligned) >= 0)) if ts_b_aligned.size > 1 else True
        print(
            "[DEBUG][INTER] Software timestamp delta: "
            f"start={ts_a[0] - ts_b_aligned[0]:.3f}s, "
            f"end={ts_a[-1] - ts_b_aligned[-1]:.3f}s"
        )
        print(f"[DEBUG][INTER] Aligned timestamp non-decreasing: {aligned_non_decreasing}")

    return signals_b_aligned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize two .bio files based on trigger markers."
    )
    parser.add_argument("file_a", help="Reference .bio file (its time grid is kept).")
    parser.add_argument("file_b", help="Secondary .bio file (its timestamps are mapped to file A).")
    parser.add_argument("output_dir", help="Directory where the aligned files are written.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics for intra/inter alignment checks.",
    )
    args = parser.parse_args()

    file_a = Path(args.file_a)
    file_b = Path(args.file_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nStep 1: intra-file alignment (fixing hardware packet losses)")
    signals_a = align_bio_signals(str(file_a))
    signals_b = align_bio_signals(str(file_b))

    print("Step 2: inter-file synchronization (mapping clocks)")
    signals_b_aligned = synchronize_bio_files(signals_a, signals_b, debug=args.debug)

    print("Step 3: saving files")
    out_a = output_dir / f"{file_a.stem}_inter_aligned{file_a.suffix}"
    out_b = output_dir / f"{file_b.stem}_inter_aligned{file_b.suffix}"

    write_bio_file(str(out_a), signals_a)
    print(f"\n[SAVED]: {out_a}")

    write_bio_file(str(out_b), signals_b_aligned)
    print(f"[SAVED]: {out_b}\n")


if __name__ == "__main__":
    main()