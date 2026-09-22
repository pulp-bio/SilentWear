# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Main Script to Generate EMG-windows from data and (optionally) extract features
=========================================================================

This module provides utilities for performing exploratory data analysis (EDA)
on extracted EMG features from a single recording.

It implements a pipeline that:

1. Loads a preprocessed EMG recording stored as HDF5.
2. Identifies contiguous text segments based on label transitions.
3. Extracts fixed-length windows from each text segment.
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

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

from typing import Dict, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.I_data_preparation.experimental_config import FS, get_active_labels
from utils.II_feature_extraction.FeatExtractorManager import FeatureExtractor
from utils.I_data_preparation.read_bio_file import print_label_statistics
from utils.I_data_preparation.onset_detection import (
    OnsetConfig,
    detect_events_in_dataframe,
    label_boxes,
    match_events_to_boxes,
    rest_intervals_between_events,
)


class Single_Recording_Windower_and_Feature_Extractor:
    def __init__(
        self,
        data_directory: Path,
        h5_file_path: Path,
        window_size_s: float,
        manual_feature_extraction: bool,
        data_augmentation: Optional[dict] = None,
        num_subwindows: Optional[int] = None,
        alignment: str = "cue",
        onset_detection: Optional[dict] = None,
    ) -> None:
        """Data windower and feature extractor operating on a single recording.

        Args:
            data_directory: main data directory for the current subject.
            alignment: where each window starts. ``"cue"`` (default) anchors it at
                the trigger transition, reproducing the historical behaviour;
                ``"onset"`` anchors it at the speech onset found by
                ``onset_detection`` without reading the trigger, which is what an
                online system would do.
            onset_detection: overrides for ``OnsetConfig``, plus the two
                windowing-only keys ``unmatched`` ("drop" | "rest") and
                ``rest_windows`` (bool).
        """
        self.data_directory = data_directory
        self.h5_file = h5_file_path

        self.window_size_s = window_size_s
        self.manual_feature_extraction = manual_feature_extraction
        self.data_augmentation = data_augmentation
        if self.manual_feature_extraction:
            self.num_subwin = num_subwindows
        else:
            self.num_subwin = None

        self.alignment = str(alignment).strip().lower()
        if self.alignment not in ("cue", "onset"):
            raise ValueError(f"alignment must be 'cue' or 'onset', got {alignment!r}")

        onset_cfg = dict(onset_detection or {})
        self.onset_unmatched = str(onset_cfg.pop("unmatched", "drop")).strip().lower()
        if self.onset_unmatched not in ("drop", "rest"):
            raise ValueError(
                f"onset_detection.unmatched must be 'drop' or 'rest', got {self.onset_unmatched!r}"
            )
        self.onset_rest_windows = bool(onset_cfg.pop("rest_windows", True))
        self.onset_rest_per_gap = int(onset_cfg.pop("rest_windows_per_gap", 1))
        if self.onset_rest_per_gap < 1:
            raise ValueError("onset_detection.rest_windows_per_gap must be >= 1")
        self.onset_config = OnsetConfig(**onset_cfg)

        self.feature_extractor = FeatureExtractor(fs=FS)

    def find_text_segments_manual_index(
        self,
        df: pd.DataFrame,
        valid_vals: Optional[Set[int]] = None,
        label_col: str = "Label_int",
        label_to_text_map: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Manual run segmentation that returns start/end in df.index LABEL space,
        matching find_text_segments_df:

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

        if label_to_text_map is not None and len(seg_df) > 0:
            seg_df["label_str"] = seg_df["label_int"].map(label_to_text_map)
        else:
            seg_df["label_str"] = None

        return seg_df.reset_index(drop=True)

    def find_text_segments_df(
        self, df: pd.DataFrame, valid_vals: Set[int], label_col: str = "Label_str"
    ) -> pd.DataFrame:
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

    def _augmentation_margin_samples(self) -> int:
        """Largest shift the augmentation will apply to a window start, in samples."""
        aug = self.data_augmentation or {}
        if str(aug.get("mode", "disabled")).lower() != "sliding_window":
            return 0
        stride_samples = int((aug.get("stride_ms", 10) * FS) / 1000)
        return stride_samples * int(aug.get("num_strides", 10))

    def find_text_segments_from_onsets(
        self, df: pd.DataFrame, label_mode: str = "word"
    ) -> pd.DataFrame:
        """Segments defined by the onset detector instead of by the trigger."""
        label_map = get_active_labels(label_mode)
        events = detect_events_in_dataframe(df, self.onset_config)
        boxes = label_boxes(df)
        matches = match_events_to_boxes(
            events, boxes, tolerance_s=self.onset_config.match_tolerance_s, fs=FS
        )

        rows = []
        n_speech = 0
        n_dropped = 0
        n_fragments = 0
        n_unmatched_as_rest = 0
        claimed: set = set()
        for event, box_idx in zip(events, matches):
            if box_idx is None:
                if self.onset_unmatched == "drop":
                    n_dropped += 1
                    continue
                label_int = 0
                n_unmatched_as_rest += 1
            else:
                if box_idx in claimed:
                    n_fragments += 1
                    continue
                claimed.add(box_idx)
                label_int = boxes[box_idx][2]
                n_speech += 1
            rows.append(
                {
                    "start_idx": int(event.onset),
                    "end_idx": int(event.offset),
                    "label_int": int(label_int),
                    "label_str": label_map.get(int(label_int)),
                    "run_len": int(event.duration_samples),
                }
            )

        n_rest = 0
        if self.onset_rest_windows:
            window_samples = int(self.window_size_s * FS)
            margin = self._augmentation_margin_samples()
            min_rest_stride = int(0.2 * FS)
            for gap_start, gap_stop in rest_intervals_between_events(
                events, len(df), min_gap_samples=window_samples + 2 * margin
            ):
                first = gap_start + margin
                last = gap_stop - margin - window_samples
                if last < first:
                    continue
                n_here = max(1, min(self.onset_rest_per_gap, 1 + (last - first) // min_rest_stride))
                if n_here == 1:
                    starts = [(first + last) // 2]
                else:
                    starts = np.unique(np.linspace(first, last, n_here).astype(int))
                for start in starts:
                    rows.append(
                        {
                            "start_idx": int(start),
                            "end_idx": int(start + window_samples),
                            "label_int": 0,
                            "label_str": label_map.get(0),
                            "run_len": int(window_samples),
                        }
                    )
                    n_rest += 1

        print(
            f"[ONSET] {len(events)} events -> {n_speech}/{len(boxes)} utterances "
            f"({n_speech / max(1, len(boxes)):.0%} recovered), "
            f"{n_fragments} mid-utterance fragments merged away, "
            f"{n_dropped} unmatched dropped, {n_unmatched_as_rest} unmatched kept as rest, "
            f"{n_rest} rest windows from gaps"
        )
        if not rows:
            return pd.DataFrame(
                columns=["start_idx", "end_idx", "label_int", "label_str", "run_len"]
            )
        return pd.DataFrame(rows).sort_values("start_idx").reset_index(drop=True)

    def extract_channel_features(
        self,
        df_filtered: pd.DataFrame,
        start_idx: int,
        channel_tag: str,
        sample_per_big_window: int,
        sample_per_small_window: int | None,
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

        if sample_per_small_window is None:
            raise ValueError("sample_per_small_window must not be None when extracting features")

        num_small_windows = sample_per_big_window // sample_per_small_window
        feature_row = {}

        for small_index in range(num_small_windows):
            small_start = start_idx + small_index * sample_per_small_window
            small_end = small_start + sample_per_small_window
            small_window_data = df_filtered.loc[small_start:small_end, channel_tag].to_numpy(copy=False)

            window_features = self.feature_extractor.extract_window_features(small_window_data)
            window_num = small_index + 1

            for feature_name, feature_value in window_features.items():
                feature_name = self.feature_extractor._build_feature_name(
                    feature_name, window_num, channel_tag
                )
                feature_row[feature_name] = feature_value

        return feature_row

    def extract_features_per_text(
        self,
        df_filtered: pd.DataFrame,
        df_channels: pd.Index,
        start_idx: int,
        sample_per_big_window: int,
        sample_per_small_window: int | None,
    ) -> dict:
        """Extract features for a single text across all channels.

        Args:
            emg_filtered: Filtered EMG data.
            start_idx: Start index for text.
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

    def extract_windows_and_features_from_df(
        self, df: pd.DataFrame, seg_df: pd.DataFrame
    ) -> pd.DataFrame:

        sample_per_big_window = int(self.window_size_s * FS)
        sample_per_small_window = None
        if self.num_subwin is not None:
            sample_per_small_window = sample_per_big_window // self.num_subwin

        # Data Augmentation Configuration
        aug_config = self.data_augmentation or {}

        augmentation_mode = str(aug_config.get("mode", "disabled")).lower()
        stride_ms = aug_config.get("stride_ms", 10)
        num_strides = aug_config.get("num_strides", 10)

        stride_samples = int((stride_ms * FS) / 1000)
        
        if augmentation_mode == "sliding_window":
            augmentation_offsets = [(0, "base")]
            augmentation_offsets += [(-step * stride_samples, "backward") for step in range(1, num_strides + 1)]
            augmentation_offsets += [(step * stride_samples, "forward") for step in range(1, num_strides + 1)]
        elif augmentation_mode == "disabled":
            augmentation_offsets = [(0, "base")]
        else:
            raise NotImplementedError(f"Unsupported data_augmentation mode: {augmentation_mode}")

        mask_ch = df.columns.str.contains("^Ch_")
        ch_cols = df.columns[mask_ch]
        mask_filt = ch_cols.str.contains("_filt")
        filt_cols = ch_cols[mask_filt]

        # Converting to NumPy array once and slicing it.
        ch_indices = [df.columns.get_loc(c) for c in filt_cols]
        df_numpy = df.values

        feature_data = []
        total_segments = len(seg_df)
        print(f"\n[DEBUG] Starting extraction of {total_segments} segments. (Augmentation mode: {augmentation_mode})")

        for index, seg in tqdm(seg_df.iterrows(), total=total_segments, desc="Analyzed segments"):
            start_idx = int(seg["start_idx"])

            for shift_samples, shift_direction in augmentation_offsets:
                augmented_start_idx = start_idx + shift_samples
                end_idx = augmented_start_idx + sample_per_big_window - 1             
                if augmented_start_idx < 0 or end_idx >= len(df):
                    continue

                # ======= Extract Features Manually ==============
                feature_row = {}
                if self.manual_feature_extraction:
                    feature_row = self.extract_features_per_text(
                        df,
                        filt_cols,
                        augmented_start_idx,
                        sample_per_big_window,
                        sample_per_small_window,
                    )

                # ---- Add metadata ----
                subject_id = self.h5_file.parents[1].name
                condition = self.h5_file.parent.name
                feature_row["Label_int"] = seg["label_int"]
                feature_row["Label_str"] = seg["label_str"]
                feature_row["subject_id"] = subject_id
                feature_row["condition"] = condition

                feature_row["batch_id"] = df["batch_id"].unique()[0]
                feature_row["session_id"] = df["session_id"].unique()[0]
                feature_row["augmentation_source_id"] = (f"{subject_id}_{condition}_{feature_row['session_id']}_{feature_row['batch_id']}_{start_idx}")
                feature_row["augmentation_direction"] = shift_direction
                feature_row["augmentation_shift_ms"] = int((shift_samples * 1000) / FS)

                # ---- Add start/stop indices for this big window ----
                feature_row["start_idx"] = augmented_start_idx
                feature_row["end_idx"] = end_idx

                # ========= Extract Entire Windows ====================
                
                for ch, ch_idx in zip(filt_cols, ch_indices):
                    feature_row[ch] = df_numpy[augmented_start_idx : end_idx + 1, ch_idx]

                feature_data.append(feature_row)
                
        return pd.DataFrame(feature_data)

    def process_single_recording(
        self, valid_labels=None, label_mode: str = "word"
    ) -> pd.DataFrame:
        if valid_labels is None:
            valid_labels = get_active_labels(label_mode).keys()
        # Read current file
        df = pd.read_hdf(self.h5_file, key="emg")
        df = pd.DataFrame(df)
        df = df.reset_index(drop=True)
        print_label_statistics(df)
        # Find segments corresponding to each Text (or rest)
        if self.alignment == "onset":
            seg_df = self.find_text_segments_from_onsets(df, label_mode=label_mode)
            seg_df = seg_df[seg_df["label_int"].isin(set(valid_labels))].reset_index(drop=True)
        else:
            seg_df = self.find_text_segments_df(
                df, valid_vals=set(valid_labels), label_col="Label_int"
            )
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
