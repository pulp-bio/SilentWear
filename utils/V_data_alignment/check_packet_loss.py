# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Packet-loss diagnostics for .bio files based on counter signals.

Usage
-----
    python check_packet_loss.py <file.bio> [<file.bio> ...]
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.V_data_alignment.read_bio_file import read_bio_file


def compute_modulus(signal: np.ndarray, dtype: np.dtype) -> int:
    """Infer modulus from dtype and observed values."""
    dt = np.dtype(dtype)

    if dt.itemsize == 1:
        return 2**8
    if dt.itemsize == 2:
        return 2**16
    
    signal_arr = np.asarray(signal).reshape(-1)
    if signal_arr.size:
        if np.issubdtype(signal_arr.dtype, np.floating):
            finite = signal_arr[np.isfinite(signal_arr)]
        else:
            finite = signal_arr
    else:
        finite = signal_arr

    if dt.itemsize == 4:
        # uint16 counters may be stored in int32 arrays.
        if finite.size:
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
            if min_val >= 0 and max_val <= 0xFFFF:
                return 2**16
        else:
            return 2**32
        return 2**32

    if dt.itemsize >= 8:
        if finite.size:
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
            if min_val >= 0:
                if max_val <= 0xFFFF:
                    return 2**16
                if max_val <= 0xFFFFFFFF:
                    return 2**32
        return 2**64

    return 2**64


def unwrap_signal(counter_array: np.ndarray, modulus: int) -> np.ndarray:
    """Unwrap a sequence into monotonic int64 values using a provided modulus."""
    counter = np.asarray(counter_array).reshape(-1).astype(np.int64)

    if counter.size <= 1:
        return counter.copy()
    if modulus <= 0:
        raise ValueError(f"Modulus must be > 0, got {modulus}.")

    modulus_i64 = np.int64(modulus)

    diffs = np.diff(counter)
    steps = diffs % modulus_i64
    
    unwrapped = np.empty_like(counter)
    unwrapped[0] = counter[0]
    unwrapped[1:] = counter[0] + np.cumsum(steps)
    
    return unwrapped


def calculate_packet_loss(
    counter_array: np.ndarray,
    dtype: np.dtype,
    signal_name: str,
) -> tuple[int, int, list[tuple[str, int, int]]]:
    """Return (lost, received, missing_entries) for a counter signal."""
    modulus = compute_modulus(np.asarray(counter_array).reshape(-1), dtype)
    counter = unwrap_signal(counter_array, modulus)

    missing_entries: list[tuple[str, int, int]] = []

    # The counter is defined to start at 0, so counter[0] itself tells us
    # how many packets were dropped before the first one we received.
    initial_loss = int(counter[0]) if counter.size > 0 else 0
    if initial_loss > 0:
        missing_entries.append((signal_name, 0, initial_loss))

    # Packets lost between consecutive received packets
    diffs = np.diff(counter)
    missing_counts = np.clip(diffs - 1, 0, None).astype(np.int64)
    lost = initial_loss + int(missing_counts.sum())

    missing_entries += [
        (signal_name, int(i + 1), int(missing_counts[i]))
        for i in np.nonzero(missing_counts)[0]
    ]

    return lost, int(counter.size), missing_entries


def print_results(
    counter_key: str,
    fs: float,
    dtype: np.dtype,
    lost: int,
    received: int,
    missing_entries: list[tuple[str, int, int]],
) -> None:
    total = received + lost
    loss_rate = (lost / total * 100.0) if total > 0 else 0.0

    print()
    print(f"[{counter_key}]")
    print(f"  dtype      : {np.dtype(dtype).name}")
    print(f"  fs         : {fs:.3f} Hz")
    print(f"  received   : {received}")
    print(f"  lost       : {lost}")
    print(f"  loss rate  : {loss_rate:.4f}%")
    print()
    print("  Missing indices:")

    if not missing_entries:
        print("    - none")
        return

    for sig_name, idx, count in missing_entries:
        if idx == 0:
            print(f"    - [{sig_name}] before index 0: {count} missing")
        else:
            time_s = idx / fs if fs > 0 else float("nan")
            print(f"    - [{sig_name}] index {idx} (t={time_s:.6f} s): {count} missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate packet loss in .bio files.")
    parser.add_argument("files", nargs="+", help="Paths to .bio files")
    args = parser.parse_args()

    for filepath in args.files:
        print(f"File: {filepath}")
        try:
            signals = read_bio_file(filepath)
        except Exception as exc:
            print(f"  Error reading file: {exc}")
            continue

        counter_keys = [k for k in signals if k.startswith("counter_")]
        if not counter_keys:
            print("  No counter signals found.")
            continue

        for key in counter_keys:
            sig = signals[key]
            dtype = sig["data"].dtype
            lost, received, missing = calculate_packet_loss(sig["data"], dtype, key)
            print_results(key, float(sig["fs"]), dtype, lost, received, missing)


if __name__ == "__main__":
    main()