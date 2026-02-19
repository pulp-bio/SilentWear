# Copyright 2026 Giusy Spacone
# Copyright 2026 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
This file contains functions to pre-process raw EMG recordings
"""

from scipy.signal import butter, filtfilt


def butter_highpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def notch_filter(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(
        order, [normal_cutoff - 0.01, normal_cutoff + 0.01], btype="bandstop", analog=False
    )
    return b, a


def apply_filters(signal_data, fs, highpass_cutoff=0.5, notch_cutoff=50.0):
    # High-pass filter
    b_hp, a_hp = butter_highpass(highpass_cutoff, fs)
    filtered_data = filtfilt(b_hp, a_hp, signal_data, axis=0)

    # Notch filter
    b_notch, a_notch = notch_filter(notch_cutoff, fs)
    filtered_data = filtfilt(b_notch, a_notch, filtered_data, axis=0)

    return filtered_data
