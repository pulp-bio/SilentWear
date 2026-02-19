# Copyright 2026 Giusy Spacone
# Copyright 2026 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Main Script to Generate EMG-windows from data and (optionally) extract features
=========================================================================

This module provides utilities for performing exploratory data analysis (EDA)
on extracted EMG features from a single recording.

It implements a pipeline that:

1. Loads a preprocessed EMG recording stored as HDF5.
2. Identifies contiguous word segments based on label transitions.
3. Extracts fixed-length windows from each word segment.
4. Optionally performs manual feature extraction using FeatureExtractor.
5. Returns a DataFrame containing:
    - Raw window data (filtered channels)
    - Extracted features (if enabled)
    - Metadata (label, session, batch, start/end indices)

Main Class
----------
Single_Recording_Windower_and_Feature_Extractor

This class operates on a single HDF5 recording and supports:

- Manual segmentation using index-based label transitions
- Pandas-based segmentation using group-by logic
- Multi-channel window extraction
- Sub-window feature extraction within each main window (see https://www.arxiv.org/pdf/2509.21964 for details)

Expected Input Format
---------------------
The HDF5 file must contain:
    - Filtered EMG channels (e.g., Ch_0_filt, Ch_1_filt, ...)
    - Label_int (integer labels)
    - Label_str (string labels)
    - batch_id
    - session_id
"""

import pandas as pd
from pathlib import Path
import sys

from typing import Dict, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.I_data_preparation.experimental_config import *
from utils.II_feature_extraction.FeatExtractorManager import FeatureExtractor
from utils.I_data_preparation.read_bio_file import print_label_statistics


class Single_Recording_Windower_and_Feature_Extractor:
    def __init__(
        self,
        data_directory: Path,
        h5_file_path: Path,
        window_size_s: float,
        manual_feature_extraction: bool,
        num_subwindows: Optional[int] = None,
    ) -> None:
        pass
        """
        Data Windower and Feature Extractor, operating on a single recording

        Aregs:
            data_directory: main data directory for the current subject
        """
        self.data_directory = data_directory
        self.h5_file = h5_file_path

        self.window_size_s = window_size_s
        self.manual_feature_extraction = manual_feature_extraction
        if self.manual_feature_extraction:
            self.num_subwin = num_subwindows
        else:
            self.num_subwin = None

        self.feature_extractor = FeatureExtractor(fs=FS)

    def find_word_segments_manual_index(
        self,
        df: pd.DataFrame,
        valid_vals: Optional[Set[int]] = None,
        label_col: str = "Label_int",
        label_to_word_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Manual run segmentation that returns start/end in df.index LABEL space,
        matching find_word_segments_df:

        - start_idx: first index label of the run (inclusive)
        - end_idx:   last index label of the run + 1 (exclusive, label space)
        - run_len:   number of rows in run
        """
        labels = df[label_col].to_numpy()
        idx = df.index.to_numpy()  # index labels (note: might not start from 0)

        n = len(labels)
        if n == 0:
            return pd.DataFrame(
                columns=["start_idx", "end_idx", "label_int", "label_str", "run_len"]
            )

        segments = []
        start_pos = 0
        curr = labels[0]

        for i in range(1, n):
            if labels[i] != curr:
                # run is [start_pos, i) in positional space
                if (valid_vals is None) or (curr in valid_vals):
                    start_label = int(idx[start_pos])
                    last_label = int(idx[i - 1])
                    segments.append(
                        {
                            "start_idx": start_label,
                            "end_idx": last_label + 1,  # exclusive in label space (matches pandas)
                            "label_int": int(curr),
                            "run_len": int(i - start_pos),
                        }
                    )
                start_pos = i
                curr = labels[i]

        # last run: [start_pos, n)
        if (valid_vals is None) or (curr in valid_vals):
            start_label = int(idx[start_pos])
            last_label = int(idx[n - 1])
            segments.append(
                {
                    "start_idx": start_label,
                    "end_idx": last_label + 1,  # exclusive in label space
                    "label_int": int(curr),
                    "run_len": int(n - start_pos),
                }
            )

        seg_df = pd.DataFrame(segments)

        # Add label_str consistent with your pandas function
        if "Label_str" in df.columns:
            # safest: take the first label_str of each run via mapping from df
            # but since we segmented on Label_int, mapping is simpler & consistent
            pass

        if label_to_word_map is not None and len(seg_df) > 0:
            seg_df["label_str"] = seg_df["label_int"].map(label_to_word_map)
        else:
            # fallback: if df has Label_str, use mapping from it (optional)
            seg_df["label_str"] = None

        return seg_df.reset_index(drop=True)

    def find_word_segments_df(self, df: pd.DataFrame, valid_vals: set[int], label_col="Label_str"):
        s = df[label_col]
        run_id = (s != s.shift(1)).cumsum()

        seg = (
            df.assign(_run=run_id)
            .groupby("_run", sort=False)
            .agg(
                start_idx=(label_col, lambda x: int(x.index[0])),
                end_idx=(label_col, lambda x: int(x.index[-1]) + 1),
                label_int=(label_col, "first"),
                label_str=(
                    ("Label_str", "first") if "Label_str" in df.columns else (label_col, "first")
                ),
                run_len=(label_col, "size"),
            )
            .reset_index(drop=True)
        )
        # self.plot_lables(s)
        # Print number of segments per label
        seg_valid = seg[seg["label_int"].isin(valid_vals)].reset_index(drop=True)
        # print("\nSegment count per label_int:")
        # print(seg_valid["label_int"].value_counts())

        # DEBUG
        # seg["span"] = seg["end_idx"] - seg["start_idx"]
        # print((seg["span"] - seg["run_len"]).value_counts().head(10))
        # print(seg.loc[(seg["span"] - seg["run_len"]) != 0].head())
        # print("df.index type:", type(df.index))
        # print("df.index example:", df.index[:10].to_list())

        return seg_valid

    def extract_channel_features(
        self,
        df_filtered: pd.DataFrame,
        start_idx: int,
        channel_tag: str,
        sample_per_big_window: int,
        sample_per_small_window: int,
    ) -> Dict[str, float]:
        """Extract features for a SINGLE channel across all small windows.

        Args:
            emg_filtered: Filtered EMG data (samples x channels).
            start_idx: Starting sample index for the big window.
            channel_ind: Channel index in the data array.
            channel_tag: Channel name tag (e.g., '01n').
            sample_per_big_window: Number of samples in big window.
            sample_per_small_window: Number of samples in small window.

        Returns:
            Dictionary with all features for this channel, keyed by feature name.
        """

        num_small_windows = sample_per_big_window // sample_per_small_window
        feature_row = {}

        for small_index in range(num_small_windows):
            small_start = start_idx + small_index * sample_per_small_window
            small_end = small_start + sample_per_small_window
            small_window_data = df_filtered.loc[small_start:small_end, channel_tag].values

            window_features = self.feature_extractor.extract_window_features(small_window_data)
            window_num = small_index + 1

            for feature_name, feature_value in window_features.items():
                feature_name = self.feature_extractor._build_feature_name(
                    feature_name, window_num, channel_tag
                )
                feature_row[feature_name] = feature_value

        return feature_row

    def extract_features_per_word(
        self,
        df_filtered: pd.DataFrame,
        df_channels: pd.Index,
        start_idx: int,
        sample_per_big_window: int,
        sample_per_small_window: int,
    ) -> dict:
        """Extract features for a single word across all channels.

        Args:
            emg_filtered: Filtered EMG data.
            start_idx: Start index for word.
            sample_per_big_window: Number of samples in big window.
            sample_per_small_window: Number of samples in small window.

        Returns:
            Dictionary of features for all channels.
        """
        feature_row = {}
        for channel in df_channels:

            channel_features = self.extract_channel_features(
                df_filtered,
                start_idx,
                channel,
                sample_per_big_window,
                sample_per_small_window,
            )
            feature_row.update(channel_features)

        return feature_row

    def extract_windows_and_features_from_df(self, df, seg_df):

        sample_per_big_window = int(self.window_size_s * FS)
        if self.num_subwin is not None:
            sample_per_small_window = sample_per_big_window // self.num_subwin

        mask_ch = df.columns.str.contains("^Ch_")
        ch_cols = df.columns[mask_ch]
        mask_filt = ch_cols.str.contains("_filt")
        filt_cols = ch_cols[mask_filt]

        feature_data = []
        for _, seg in seg_df.iterrows():

            start_idx = int(seg["start_idx"])
            end_seg = int(seg["end_idx"])  # end of the run (exclusive)

            end_idx = (
                start_idx + sample_per_big_window - 1
            )  # since we work with pandas, loc includes last
            if end_idx >= df.index[-1]:
                continue

            # ======= Extract Features Manually ==============
            feature_row = {}
            if self.manual_feature_extraction:
                feature_row = self.extract_features_per_word(
                    df,
                    filt_cols,
                    start_idx,
                    sample_per_big_window,
                    sample_per_small_window,
                )

            # ---- Add metadata ----
            feature_row["Label_int"] = seg["label_int"]
            feature_row["Label_str"] = seg["label_str"]

            feature_row["batch_id"] = df["batch_id"].unique()[0]
            feature_row["session_id"] = df["session_id"].unique()[0]

            # ---- Add start/stop indices for this big window ----
            feature_row["start_idx"] = start_idx
            feature_row["end_idx"] = end_idx

            # ========= Extract Entire Windows ====================

            for ch in filt_cols:
                feature_row[ch] = df.loc[start_idx:end_idx, ch].values

            feature_data.append(feature_row)
        return pd.DataFrame(feature_data)

    def process_single_recording(self, valid_labels=label_to_word_map.keys()):
        # Read current file
        df = pd.read_hdf(self.h5_file, key="emg")
        print_label_statistics(df)
        # Find segments corresponding to each Word (or rest)
        seg_df = self.find_word_segments_df(df, valid_vals=valid_labels, label_col="Label_int")
        df_wins_feats = self.extract_windows_and_features_from_df(df, seg_df)
        print(df_wins_feats)
        return df_wins_feats


if __name__ == "__main__":
    # adjust here with the path.
    # Convention strucutre: \data\raw\<subject_id>\<condition>
    main_data_dire = Path(r"\data\raw\<subject_id>\<condition>")

    # read all bio files
    all_bios_in_folder = main_data_dire.rglob("*.h5")

    for curr_h5 in all_bios_in_folder:
        print(curr_h5)
        # intialize a new class
        feat_extract = Single_Recording_Windower_and_Feature_Extractor(
            main_data_dire, curr_h5, window_size_s=1.4, manual_feature_extraction=False
        )
