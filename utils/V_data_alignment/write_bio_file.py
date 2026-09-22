# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Write utilities for .bio files."""

from __future__ import annotations

import struct
import numpy as np


DTYPE_TO_CHAR = {
    np.dtype("bool"): b"?",
    np.dtype("int8"): b"b",
    np.dtype("uint8"): b"B",
    np.dtype("int16"): b"h",
    np.dtype("uint16"): b"H",
    np.dtype("int32"): b"i",
    np.dtype("uint32"): b"I",
    np.dtype("int64"): b"q",
    np.dtype("uint64"): b"Q",
    np.dtype("float32"): b"f",
    np.dtype("float64"): b"d",
}


def write_bio_file(file_path: str, signals: dict) -> None:
    """Write signals, timestamps, and triggers to a .bio file."""
    signal_names = [name for name in signals if name not in ("timestamp", "trigger")]
    timestamp = np.asarray(signals["timestamp"]["data"], dtype=np.float64)
    fs_base = float(signals["timestamp"]["fs"])
    n_samp_base = int(timestamp.shape[0])

    with open(file_path, "wb") as f:
        f.write(struct.pack("<I", len(signal_names)))
        f.write(struct.pack("<fI", fs_base, n_samp_base))

        for signal_name in signal_names:
            signal = signals[signal_name]
            data = np.asarray(signal["data"])
            signal_name_bytes = signal_name.encode("utf-8")

            f.write(struct.pack("<I", len(signal_name_bytes)))
            f.write(struct.pack(f"<{len(signal_name_bytes)}s", signal_name_bytes))
            f.write(
                struct.pack(
                    "<f2Ic",
                    float(signal["fs"]),
                    int(data.shape[0]),
                    int(data.shape[1]),
                    DTYPE_TO_CHAR[np.dtype(data.dtype)],
                )
            )

        has_trigger = "trigger" in signals
        f.write(struct.pack("<?", has_trigger))

        f.write(timestamp.tobytes(order="C"))

        for signal_name in signal_names:
            data = np.asarray(signals[signal_name]["data"])
            f.write(data.tobytes(order="C"))

        if has_trigger:
            trigger = np.asarray(signals["trigger"]["data"], dtype=np.uint32)
            f.write(trigger.tobytes(order="C"))