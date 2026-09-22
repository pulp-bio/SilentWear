# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Compute the delay between the onsets of multiple periodic signals across files.

Pipeline:
  1. Read multiple .bio files and extract available known signals.
  2. Detect onsets for each signal independently.
  3. Pool all onsets onto a master timeline.
  4. Cluster the onsets by time to identify simultaneous physiological events.
  5. Compute intra-event delays dynamically based ONLY on available signals.
  6. Save a dynamically named CSV in a specific output directory.

Usage:
    python compute_peak_delay.py <file.bio> [<file.bio> ...] [--output-dir DIR]
"""

import csv
import sys
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.V_data_alignment.read_bio_file import read_bio_file
from utils.V_data_alignment.check_packet_loss import calculate_packet_loss
from utils.IV_plots.filter import apply_filters_nan


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------


DEFAULT_OUTPUT_DIR = Path("./delay_analysis_results")

SIGNAL_CONFIG = {
    "emg":     {"channel": 0, "threshold": 10000.0},
    "mic_emg": {"channel": 0, "threshold": 0.01},
    "eeg":     {"channel": 0, "threshold": 10000.0},
    "mic_eeg": {"channel": 0, "threshold": 0.01},
}

DELAY_PAIRS = [
    ("emg", "mic_emg"),
    ("eeg", "mic_eeg"),
    ("emg", "eeg"),
    ("mic_emg", "mic_eeg")
]

TRIM_SECONDS = 0.1           # [s]
PERIOD_S = 1.159             # [s]
MIN_DISTANCE_RATIO = 0.8    # 80 % of the period

MAX_DELAY_S = 0.200
CUTOFF_HZ = 20.0
HIGHPASS_FILTERS = [{"type": "highpass", "order": 4, "cutoff": CUTOFF_HZ}]

# Use half a period to group peaks of the same event
CLUSTER_MARGIN_S = PERIOD_S / 2.0  

# Exclude delays larger than this value (in ms) from Mean and Std Dev calculations
OUTLIER_THRESHOLD_MS = 100.0


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def extract_timestamp(filepath: str) -> str:
    """Extracts a timestamp like 2026-05-21_11-09-18 from the filename."""
    match = re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', Path(filepath).name)
    return match.group(0) if match else "unknown_time"


def summarize_packet_loss(signals: dict) -> dict[str, int]:
    """Return lost packet counts per known signal name using counter signals."""
    packet_loss: dict[str, int] = {}

    for counter_name in sorted(name for name in signals if name.startswith("counter_")):
        signal_name = counter_name.replace("counter_", "", 1)
        if signal_name not in SIGNAL_CONFIG:
            continue

        counter_signal = signals[counter_name]
        lost, _, _ = calculate_packet_loss(
            counter_signal["data"],
            counter_signal["data"].dtype,
            signal_name,
        )
        packet_loss[signal_name] = packet_loss.get(signal_name, 0) + lost

    return packet_loss


def detect_onsets(signal: np.ndarray, fs: float, thr_absolute: float) -> tuple[np.ndarray, np.ndarray]:
    """Detect onsets in a signal using high-pass filtering, rectification, and thresholding."""
    trim_samples = int(round(TRIM_SECONDS * fs))
    
    if len(signal) > trim_samples:
        clean_signal = signal[trim_samples:]
    else:
        clean_signal = signal
        trim_samples = 0

    finite_mask = np.isfinite(clean_signal)
    if not np.any(finite_mask):
        return np.array([], dtype=np.int64) + trim_samples, np.array([], dtype=np.float64)

    filtered = apply_filters_nan(clean_signal.reshape(-1, 1), fs, HIGHPASS_FILTERS).reshape(-1)
    rectified = np.abs(filtered)

    dist = int(round(PERIOD_S * MIN_DISTANCE_RATIO * fs))
    onsets = []
    last_onset = -dist  

    finite_edges = np.diff(np.concatenate(([False], np.isfinite(rectified), [False]))).nonzero()[0]
    for start, end in zip(finite_edges[::2], finite_edges[1::2]):
        segment = rectified[start:end]
        is_over_threshold = segment > thr_absolute
        crossings = np.where((~is_over_threshold[:-1]) & (is_over_threshold[1:]))[0] + start

        for crossing in crossings:
            if crossing - last_onset >= dist:
                onsets.append(crossing)
                last_onset = crossing   

    onsets_absolute = np.array(onsets) + trim_samples
    return onsets_absolute, rectified


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligned delay analysis from .bio files")
    parser.add_argument("files", nargs="+", help="One or more paths to .bio files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the delay CSV is written.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir

    global_raw_signals = {}
    packet_loss_by_signal: dict[str, int] = {}
    
    print(f"\n{'-'*80}\nReading and aggregating files\n{'-'*80}")
    for bio_file_str in args.files:
        bio_file = Path(bio_file_str)
        try:
            signals = read_bio_file(str(bio_file))
            print(f"File '{bio_file.name}' read. Signals found: {list(signals.keys())}")

            file_packet_loss = summarize_packet_loss(signals)
            for signal_name, lost_packets in file_packet_loss.items():
                packet_loss_by_signal[signal_name] = packet_loss_by_signal.get(signal_name, 0) + lost_packets
            
            for sig_name, sig_data in signals.items():
                if sig_name in SIGNAL_CONFIG:
                    global_raw_signals[sig_name] = sig_data
        except Exception as e:
            print(f"Error reading file {bio_file.name}: {e}")

    if not global_raw_signals:
        print("\nNo useful signals found in the provided files.")
        return

    print(f"\n{'-'*80}\nPacket Loss\n{'-'*80}")
    if packet_loss_by_signal:
        signals_with_loss = [sig for sig, lost in packet_loss_by_signal.items() if lost > 0]
        for sig_name in [s for s in ["emg", "mic_emg", "eeg", "mic_eeg"] if s in packet_loss_by_signal]:
            print(f"[{sig_name}] lost packets: {packet_loss_by_signal[sig_name]}")

        if signals_with_loss:
            print(f"Signals with packet loss: {', '.join(signals_with_loss)}")
        else:
            print("No packet loss detected in the available counter signals.")
    else:
        print("No counter signals found, so packet loss could not be computed.")

    # Determine which signals are actually present to build columns
    found_signals_set = set(global_raw_signals.keys())
    ordered_signals = [s for s in ["emg", "mic_emg", "eeg", "mic_eeg"] if s in found_signals_set]

    # Determine which delay pairs can be calculated based on present signals
    valid_pairs = [(a, b) for a, b in DELAY_PAIRS if a in found_signals_set and b in found_signals_set]

    # 1. Process all available signals
    processed_data = {}
    print(f"\n{'-'*80}\nProcessing Signals\n{'-'*80}")
    for sig_name in ordered_signals:
        sig_data = global_raw_signals[sig_name]
        conf = SIGNAL_CONFIG[sig_name]
        fs = float(sig_data["fs"])
        raw_data = sig_data["data"][:, conf["channel"]].astype(np.float64)
        
        onsets, rect = detect_onsets(raw_data, fs, conf["threshold"])
        processed_data[sig_name] = onsets / fs
        
        print(f"[{sig_name}]  fs={fs} Hz | Thr={conf['threshold']} | Onsets={len(onsets)}")

    # 2. Clustering
    all_onsets = []
    for sig_name, onset_times_s in processed_data.items():
        for onset_time in onset_times_s:
            all_onsets.append((onset_time, sig_name))
            
    all_onsets.sort(key=lambda x: x[0])
    
    clusters = []
    current_cluster = {}
    last_onset_time = None
    
    for onset_time, sig in all_onsets:
        # Checking if the time distance from the last onset exceeds the margin
        if last_onset_time is None or (onset_time - last_onset_time) > CLUSTER_MARGIN_S:
            if current_cluster:
                clusters.append(current_cluster)
            # Starting a new cluster with the current onset
            current_cluster = {sig: onset_time}
        else:
            # If the signal is not already in the current cluster, add it
            if sig not in current_cluster:
                current_cluster[sig] = onset_time
                
        last_onset_time = onset_time
        
    if current_cluster:
        clusters.append(current_cluster)

    # 3. Building the results table
    rows = []
    for idx, c in enumerate(clusters, 1):
        row_dict: dict[str, Any] = {"pair_id": idx}
        
        # Get times
        for sig in ordered_signals:
            row_dict[sig] = c.get(sig, np.nan)
            
        # Calculate delays
        for a, b in valid_pairs:
            onset_a = row_dict[a]
            onset_b = row_dict[b]
            row_dict[f"delay_{a}_{b}"] = (onset_b - onset_a) * 1000 if not np.isnan(onset_a) and not np.isnan(onset_b) else np.nan
            
        rows.append(row_dict)

    # Compute statistics (excluding outliers)
    delay_keys = [f"delay_{a}_{b}" for a, b in valid_pairs]
    
    if delay_keys and rows:
        delay_matrix = np.array([[r[k] for k in delay_keys] for r in rows], dtype=np.float64)
        
        # Identify outliers
        outlier_mask = np.abs(delay_matrix) > OUTLIER_THRESHOLD_MS
        outliers_count = np.sum(outlier_mask, axis=0)
        
        # Create a filtered matrix where absolute delays > OUTLIER_THRESHOLD_MS are set to NaN
        filtered_matrix = np.where(outlier_mask, np.nan, delay_matrix)

        finite_mask = np.isfinite(filtered_matrix)
        finite_counts = np.sum(finite_mask, axis=0)
        mean_delays = np.full(len(delay_keys), np.nan, dtype=np.float64)
        std_delays = np.full(len(delay_keys), np.nan, dtype=np.float64)

        valid_stats = finite_counts > 0
        if np.any(valid_stats):
            valid_matrix = filtered_matrix[:, valid_stats]
            mean_delays[valid_stats] = np.nanmean(valid_matrix, axis=0)
            std_delays[valid_stats] = np.nanstd(valid_matrix, axis=0)
    else:
        outliers_count = []
        mean_delays = []
        std_delays = []
        outlier_mask = np.zeros((len(rows), 0), dtype=bool)

    # 4. Print tables
    def fmt(val, template="{:.3f}"):
        return template.format(val) if not np.isnan(val) else "---"

    header_cols = ["pair_id"] + [f"{s}_s" for s in ordered_signals] + [f"d_{a}_{b}_ms" for a, b in valid_pairs]
    
    # Exact formatting setup
    id_fmt = "{:>16}"
    sig_fmt = " | {:>12}" * len(ordered_signals)
    delay_fmt = " | {:>26}" * len(valid_pairs)
    full_fmt = id_fmt + sig_fmt + delay_fmt
    
    # Compute separator length dynamically based on the format
    dummy_row = [""] * len(header_cols)
    sep_len = len(full_fmt.format(*dummy_row))
    
    # ALL ROWS
    print(f"\n[ALL EVENTS]")
    print(f"{'-' * sep_len}")
    print(full_fmt.format(*header_cols))
    print("-" * sep_len)
    
    for r in rows:
        row_vals = [r["pair_id"]]
        row_vals.extend([fmt(r[s]) for s in ordered_signals])
        row_vals.extend([fmt(r[k], "{:+.2f}") for k in delay_keys])
        print(full_fmt.format(*row_vals))

    # OUTLIERS
    print(f"\n[OUTLIER EVENTS (|delay| > {OUTLIER_THRESHOLD_MS:g} ms)]")
    print(f"{'-' * sep_len}")
    print(full_fmt.format(*header_cols))
    print("-" * sep_len)
    
    outliers_printed = False
    for i, r in enumerate(rows):
        # Print ONLY rows that have at least one outlier
        if np.any(outlier_mask[i]):
            outliers_printed = True
            row_vals = [r["pair_id"]]
            row_vals.extend([fmt(r[s]) for s in ordered_signals])
            row_vals.extend([fmt(r[k], "{:+.2f}") for k in delay_keys])
            print(full_fmt.format(*row_vals))

    if not outliers_printed:
        empty_msg = f"NO OUTLIERS DETECTED (> {OUTLIER_THRESHOLD_MS:g} ms)"
        print(f"{empty_msg:^{sep_len}}")

    # Statistics
    print("-" * sep_len)
    print()
    if delay_keys:
        stat_outliers = ["OUTLIERS"] + [""] * len(ordered_signals) + [f"{int(c)}" for c in outliers_count]
        stat_mean = [f"MEAN (<={OUTLIER_THRESHOLD_MS:g}ms)"] + [""] * len(ordered_signals) + [fmt(m, "{:+.2f}") for m in mean_delays]
        stat_std = [f"STD (<={OUTLIER_THRESHOLD_MS:g}ms)"] + [""] * len(ordered_signals) + [fmt(s, "{:.2f}") for s in std_delays]
        
        print(full_fmt.format(*stat_outliers))
        print(full_fmt.format(*stat_mean))
        print(full_fmt.format(*stat_std))

    # 5. CSV Naming and Saving
    output_dir.mkdir(parents=True, exist_ok=True)
    
    name_parts = []
    if "eeg" in ordered_signals or "mic_eeg" in ordered_signals: name_parts.append("eeg")
    if "emg" in ordered_signals or "mic_emg" in ordered_signals: name_parts.append("emg")
    if "mic_eeg" in ordered_signals or "mic_emg" in ordered_signals: name_parts.append("mic")
    
    prefix = "_".join(name_parts)
    timestamp = extract_timestamp(args.files[0]) # Get timestamp from the first file
    
    csv_filename = f"{prefix}_{timestamp}_delay.csv"
    output_csv = output_dir / csv_filename
    
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        w.writerow(["Analyzed files"])
        for file_path in args.files:
            w.writerow([Path(file_path).name])
        w.writerow([])
        
        csv_headers = ["pair_id"] + [f"{s}_s" for s in ordered_signals] + [f"delay_{a}_{b}_ms" for a, b in valid_pairs]
        w.writerow(csv_headers)
        
        for r in rows:
            csv_row = [r["pair_id"]]
            csv_row.extend([f"{r[s]:.6f}" if not np.isnan(r[s]) else "" for s in ordered_signals])
            csv_row.extend([f"{r[k]:.3f}" if not np.isnan(r[k]) else "" for k in delay_keys])
            w.writerow(csv_row)
            
        if delay_keys:
            w.writerow([])
            w.writerow([f"Statistics (excluding |delay| > {OUTLIER_THRESHOLD_MS:g} ms)"])
            
            outlier_row = ["Outliers Excluded (Count)"] + [""] * len(ordered_signals)
            outlier_row.extend([f"{int(c)}" for c in outliers_count])
            w.writerow(outlier_row)
            
            mean_row = [f"Mean (ms) (<={OUTLIER_THRESHOLD_MS:g}ms)"] + [""] * len(ordered_signals)
            mean_row.extend([f"{m:.3f}" if not np.isnan(m) else "" for m in mean_delays])
            w.writerow(mean_row)
            
            std_row = [f"Std (ms) (<={OUTLIER_THRESHOLD_MS:g}ms)"] + [""] * len(ordered_signals)
            std_row.extend([f"{s:.3f}" if not np.isnan(s) else "" for s in std_delays])
            w.writerow(std_row)

        w.writerow([])
        w.writerow(["Packet loss summary"])
        for sig_name in [s for s in ["emg", "mic_emg", "eeg", "mic_eeg"] if s in packet_loss_by_signal]:
            w.writerow([f"Lost packets ({sig_name})", packet_loss_by_signal[sig_name]])

    print(f"\n[SAVED]: {output_csv}")

if __name__ == "__main__":
    main()