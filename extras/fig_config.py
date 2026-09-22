# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Configuration for figure generation."""

ADS1298_LSB_UV = 2 * 2.4 / (6 * 2 ** 24) * 1e6       # 0.0476837 uV per code
BIO_FILE_LSB = 4e6 / 2 ** 23                         # 0.476837, measured
RAW_TO_UV = ADS1298_LSB_UV / BIO_FILE_LSB            # = 0.1

channel_colors = [
    # Blue pair
    "#1f77b4",  # CH0
    "#6baed6",  # CH1
    # Orange pair
    "#ff7f0e",  # CH2
    "#ffbb78",  # CH3
    # Green pair
    "#2ca02c",  # CH4
    "#98df8a",  # CH5
    # Red pair
    "#d62728",  # CH6
    "#ff9896",  # CH7
    # Purple pair
    "#9467bd",  # CH8
    "#c5b0d5",  # CH9
    # Brown pair
    "#8c564b",  # CH10
    "#c49c94",  # CH11
    # Teal pair
    "#17becf",  # CH12
    "#9edae5",  # CH13
]

neckband_ch_order = [0, 1, 2, 5, 3, 4, 7, 6, 8, 15, 9, 14, 10, 13]