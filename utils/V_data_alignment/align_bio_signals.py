# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Align and repair signals stored in a .bio file (intra-file alignment).

Usage:
-----
    python align_bio_signals.py <input_file.bio> <output_directory> [--debug]

The script reads signals from the specified input .bio file, aligns them based on their hardware timestamps,
repairs any missing packets by filling them with NaNs, and saves the aligned signals to a new .bio file in the
specified output directory.
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.V_data_alignment.read_bio_file import read_bio_file
from utils.V_data_alignment.write_bio_file import write_bio_file
from utils.V_data_alignment.check_packet_loss import compute_modulus, unwrap_signal

# ---------------------------------------------------------------------------
# Debugging utilities
# ---------------------------------------------------------------------------


def _us_to_ms(value_us: float) -> str:
    return f"{value_us / 1_000.0:.3f} ms"


def _print_timestamp_debug_info(signals: dict, title: str) -> None:
    """Print the first/last sample and sampling info of the EMG and mic_EMG hardware timestamps."""
    print(f"\n{title}")
    first_values = {}
    last_values = {}
    for ts_name in ("timestamp_emg", "timestamp_mic_emg"):
        data = np.asarray(signals[ts_name]["data"]).reshape(-1)
        first_value = float(data[0])
        last_value = float(data[-1])
        first_values[ts_name] = first_value
        last_values[ts_name] = last_value

        fs_ts = signals[ts_name]["fs"]
        step_ms = 1_000_000.0 / fs_ts / 1_000.0
        signal_name = ts_name.replace("timestamp_", "", 1)
        fs_signal = signals[signal_name]["fs"] if signal_name in signals else None
        samples_per_packet = (
            int(round(fs_signal / fs_ts)) if fs_signal is not None else None
        )

        print(
            f"  {ts_name}: fs={fs_ts:.1f} Hz, "
            f"nominal step={step_ms:.3f} ms, "
            f"samples/packet={samples_per_packet}"
        )
        print(
            f"    first value = {_us_to_ms(first_value)}, "
            f"last value = {_us_to_ms(last_value)}"
        )

    if "timestamp_emg" in first_values and "timestamp_mic_emg" in first_values:
        diff_first = first_values["timestamp_emg"] - first_values["timestamp_mic_emg"]
        diff_last  = last_values["timestamp_emg"]  - last_values["timestamp_mic_emg"]
        print(f"  difference first values (timestamp_emg - timestamp_mic_emg) = {_us_to_ms(diff_first)}")
        print(f"  difference last values  (timestamp_emg - timestamp_mic_emg) = {_us_to_ms(diff_last)}")


def _print_trigger_rising_edge_debug_info(
    signals: dict,
    title: str,
    trim_offsets: dict[str, int] | None = None,
) -> None:
    """Print timestamp values around the first and last rising edge of the trigger signal."""
    trigger = signals.get("trigger")
    if trigger is None:
        return

    trigger_data = np.asarray(trigger["data"]).reshape(-1)
    rising_edges = np.flatnonzero((trigger_data[:-1] <= 0) & (trigger_data[1:] > 0)) + 1

    print(f"\n{title}")

    if rising_edges.size == 0:
        print("  no rising edges found in trigger signal")
        return

    edge_positions = [("first", int(rising_edges[0])), ("last", int(rising_edges[-1]))]
    trigger_fs = signals["trigger"]["fs"]

    for edge_label, edge_idx in edge_positions:
        print(f"  trigger {edge_label} rising edge index = {edge_idx} (sample)")

        emg_value = None
        mic_value = None

        for ts_name in ("timestamp_emg", "timestamp_mic_emg"):
            ts_data = np.asarray(signals[ts_name]["data"]).reshape(-1)
            if ts_data.size == 0:
                print(f"  {ts_name}: empty")
                continue

            ts_idx_global = int(edge_idx * signals[ts_name]["fs"] / trigger_fs)
            offset = (trim_offsets or {}).get(ts_name, 0)
            ts_idx = ts_idx_global - offset

            value, clamped = _value_at_edge(ts_data, ts_idx)
            suffix = " (clamped to last sample)" if clamped else ""
            print(f"  {ts_name}: value at trigger edge = {_us_to_ms(value)}{suffix}")

            if ts_name == "timestamp_emg":
                emg_value = value
            elif ts_name == "timestamp_mic_emg":
                mic_value = value

        if emg_value is not None and mic_value is not None:
            diff = emg_value - mic_value
            print(
                f"  difference at {edge_label} trigger edge "
                f"(timestamp_emg - timestamp_mic_emg) = {_us_to_ms(diff)}"
            )


def _value_at_edge(ts_data: np.ndarray, ts_idx: int) -> tuple[float, bool]:
    """Return the timestamp at ts_idx, clamping to the last sample if out of bounds."""
    if ts_idx < ts_data.size:
        return float(ts_data[ts_idx]), False
    return float(ts_data[-1]), True


# ---------------------------------------------------------------------------
# Main alignment utilities
# ---------------------------------------------------------------------------


def _trim_signals(signals: dict) -> tuple[dict, dict[str, int]]:
    """Trim all signals to the common time window defined by hardware timestamps."""
    hw_timestamps_names = [
        name for name in signals
        if name.startswith("timestamp_") and signals[name]["data"].size > 0
    ]

    if not hw_timestamps_names:
        return signals, {}

    # Unwraps the hardware timestamps
    unwrapped_timestamps = {}
    for ts_name in hw_timestamps_names:
        ts_raw = signals[ts_name]["data"].reshape(-1)
        if ts_raw.size == 0:
            continue
        ts_modulus = compute_modulus(ts_raw, ts_raw.dtype)
        unwrapped_timestamps[ts_name] = unwrap_signal(ts_raw, ts_modulus).astype(np.float64)

    if not unwrapped_timestamps:
        return signals, {}

    # Computes the common time window across unwrapped hardware timestamps
    starts = [float(unwrapped_timestamps[t][0]) for t in unwrapped_timestamps]
    ends = [float(unwrapped_timestamps[t][-1]) for t in unwrapped_timestamps]
    common_start = max(starts)
    common_end = min(ends)

    if common_start >= common_end:
        raise ValueError("Signals have no overlapping time window.")

    # Computes the trimming indices for hardware timestamps and their associated signals
    packet_windows: dict[str, tuple[int, int]] = {}
    for ts_name in hw_timestamps_names:
        ts_data = unwrapped_timestamps.get(ts_name)
        if ts_data is None or ts_data.size == 0:
            continue
        start_packet = int(np.searchsorted(ts_data, common_start, side="left"))
        end_packet = int(np.searchsorted(ts_data, common_end, side="right"))
        packet_windows[ts_name] = (start_packet, end_packet)

    # Applies the trimming to hardware signals
    trimmed = {}
    for signal_name, signal in signals.items():
        # Ignoring the software signals
        if signal_name in ("timestamp", "trigger"):
            trimmed[signal_name] = {"fs": signal["fs"], "data": signal["data"].copy()}
            continue

        # Gets the name of the signal associated with the hw timestamp
        if signal_name.startswith("timestamp_"):
            ts_name = signal_name
        elif signal_name.startswith("counter_"):
            ts_name = f"timestamp_{signal_name.replace('counter_', '', 1)}"
        else:
            ts_name = f"timestamp_{signal_name}"

        # If the associated timestamp is not among the valid HW ones, it is copied without trimming
        if ts_name not in packet_windows:
            trimmed[signal_name] = {"fs": signal["fs"], "data": signal["data"].copy()}
            continue

        start_packet, end_packet = packet_windows[ts_name]
        samples_per_packet = int(round(signal["fs"] / signals[ts_name]["fs"]))

        sample_start = start_packet * samples_per_packet
        sample_end = end_packet * samples_per_packet

        trimmed[signal_name] = {"fs": signal["fs"], "data": signal["data"][sample_start:sample_end].copy()}

    trim_offsets = {ts_name: packet_windows[ts_name][0] for ts_name in packet_windows}
    return trimmed, trim_offsets


def _repair_signals(signals: dict) -> None:
    """Repair signals by filling in missing packets with NaNs, reconstructs the correct counter values and timestamps."""
    counter_names = [name for name in signals if name.startswith("counter_")]

    for counter_name in counter_names:
        # Derives the associated payload and timestamp signal names
        signal_name = counter_name.replace("counter_", "", 1)
        timestamp_name = f"timestamp_{signal_name}"

        if signal_name not in signals or timestamp_name not in signals:
            continue

        counter_raw = signals[counter_name]["data"].reshape(-1)
        if counter_raw.size == 0:
            continue

        counter_original_dtype = counter_raw.dtype

        modulus = compute_modulus(counter_raw, counter_original_dtype)
        unwrapped = unwrap_signal(counter_raw, modulus)

        relative_indices = (unwrapped - unwrapped[0]).astype(np.int64)
        total_expected_packets = int(relative_indices[-1]) + 1

        payload = signals[signal_name]["data"].astype(np.float64)
        payload_channels = payload.shape[1]
        samples_per_packet = int(round(signals[signal_name]["fs"] / signals[counter_name]["fs"]))
        payload_packets = payload.reshape(len(unwrapped), samples_per_packet, payload_channels)

        rebuilt_counter = np.arange(total_expected_packets) % modulus

        rebuilt_payload_packets = np.full((total_expected_packets, samples_per_packet, payload_channels), np.nan)
        rebuilt_payload_packets[relative_indices] = payload_packets

        # Reconstruct the hardware timestamps
        timestamp_raw = signals[timestamp_name]["data"].reshape(-1)
        timestamp_original_dtype = timestamp_raw.dtype
        timestamp_modulus = compute_modulus(timestamp_raw, timestamp_original_dtype)
        timestamp_unwrapped = unwrap_signal(timestamp_raw, timestamp_modulus).astype(np.float64)

        timestamp_step = 1_000_000.0 / float(signals[timestamp_name]["fs"])
        rebuilt_timestamps = timestamp_unwrapped[0] + (np.arange(total_expected_packets, dtype=np.float64) * timestamp_step)
        rebuilt_timestamps[relative_indices] = timestamp_unwrapped
        rebuilt_timestamps_wrapped = (rebuilt_timestamps % timestamp_modulus).astype(timestamp_original_dtype)

        signals[counter_name]["data"] = rebuilt_counter.astype(counter_original_dtype).reshape(-1, 1)
        signals[signal_name]["data"] = rebuilt_payload_packets.reshape(-1, payload_channels)
        signals[timestamp_name]["data"] = rebuilt_timestamps_wrapped.reshape(-1, 1)


def align_bio_signals(file_path: str, debug: bool = False) -> dict:
    """Align and repair signals stored in a .bio file."""
    signals = read_bio_file(file_path)

    if debug:
        _print_timestamp_debug_info(signals, "[BEFORE ALIGNMENT] timestamp")
        _print_trigger_rising_edge_debug_info(signals, "[BEFORE ALIGNMENT] trigger edge timestamp")

    aligned_signals, trim_offsets = _trim_signals(signals)
    _repair_signals(aligned_signals)

    if debug:
        _print_timestamp_debug_info(aligned_signals, "[AFTER ALIGNMENT] timestamp")
        _print_trigger_rising_edge_debug_info(
            aligned_signals,
            "[AFTER ALIGNMENT] trigger edge timestamp",
            trim_offsets=trim_offsets,
        )

    return aligned_signals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align and repair signals stored in a .bio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_bio", type=Path, help="Path to the input .bio file.")
    parser.add_argument("output_dir", type=Path, help="Directory where the aligned file will be saved.")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Print diagnostic information: first/last timestamp values, "
            "sampling frequencies, nominal packet step, samples-per-packet, "
            "and trigger edge mapping."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_bio = args.output_dir / f"{args.input_bio.stem}_aligned{args.input_bio.suffix}"

    aligned = align_bio_signals(str(args.input_bio), debug=args.debug)
    write_bio_file(str(output_bio), aligned)
    print(f"\n[SAVED]: {output_bio}")