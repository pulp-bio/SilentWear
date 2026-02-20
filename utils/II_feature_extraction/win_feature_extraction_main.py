# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Global Windowing and Feature Extraction Pipeline
================================================

This module generates fixed-length EMG windows (and optionally extracted
features) for all processed recordings of a given subject.

It operates on preprocessed HDF5 recordings stored (typically) under:

    <data_directory>/data_raw_and_filt/<subject_id>/(silent|vocalized)/*.h5

For each recording, the pipeline:

1. Loads the processed EMG DataFrame.
2. Identifies valid labeled segments (words / commands).
3. Extracts fixed-length windows starting at each segment.
4. Optionally performs manual feature extraction per window.
5. Saves the resulting windows and features to:

    <data_directory>/wins_and_features/<subject_id>/<condition>/WIN_<window_ms>/
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.II_feature_extraction.SingleRecordingExtractor import (
    Single_Recording_Windower_and_Feature_Extractor,
)
from utils.general_utils import *


class Global_Windower_and_Feature_Extractor:

    def __init__(
        self,
        subject_config: SubjectConfig,
    ) -> None:

        pass

        """Initialize Windower_and_Feature_Extractor

        Args:
            data_directory: main data directory for the current subject

            manual_feature_extraction: if True, return also manual features. Else, return only windows
        """

        self.data_directory = subject_config.data_directory
        self.sub_id = subject_config.subject_id
        self.manual_feature_extraction = subject_config.manual_feature_extraction
        self.save_data = subject_config.save_wins_and_feats
        self.dire_wins_feats_silent = None
        self.dire_wins_feats_vocalized = None
        self.win_size_sec = subject_config.window_size_s
        self.num_subwins = subject_config.num_subwindows

    def find_all_processed_h5(self):
        "Function to return all h5 files contained in the processed directory"
        # Find all .bio files under the subject raw folder
        h5_files = sorted((self.data_directory / "data_raw_and_filt" / self.sub_id).rglob("*.h5"))
        # print("Found files for current user:")
        # print(h5_files)
        return h5_files

    def create_saving_directory(self):
        "Function to create saving directory for the extracted data"

        # Extract the parent directory

        self.data_dire_wins_and_feats = self.data_directory / "wins_and_features"
        # if self.data_dire_wins_and_feats.exists() == False:
        #     self.data_dire_wins_and_feats.mkdir()
        #     print("[INFO] Created directory:", self.data_dire_wins_and_feats)
        self.dire_wins_feats_silent = Path(
            self.data_dire_wins_and_feats
            / self.sub_id
            / "silent"
            / f"WIN_{int(self.win_size_sec*1000)}"
        )
        self.dire_wins_feats_vocalized = Path(
            self.data_dire_wins_and_feats
            / self.sub_id
            / "vocalized"
            / f"WIN_{int(self.win_size_sec*1000)}"
        )

        if (self.dire_wins_feats_silent).exists() == False:
            self.dire_wins_feats_silent.mkdir(parents=True)
            print("[INFO] Created directory:", self.dire_wins_feats_silent)
        if (self.dire_wins_feats_vocalized).exists() == False:
            self.dire_wins_feats_vocalized.mkdir(parents=True)
            print("[INFO] Created directory:", self.dire_wins_feats_vocalized)

    def main(self):

        if self.save_data:
            self.create_saving_directory()

        h5_files = self.find_all_processed_h5()
        print("hf files found:")
        print(h5_files)
        for curr_h5_file in h5_files:
            print("Processing file:", curr_h5_file)

            if self.save_data:
                if curr_h5_file.parent.name == "silent":
                    print("Silent recording")
                    df_save_path = self.dire_wins_feats_silent / curr_h5_file.name
                elif curr_h5_file.parent.name == "vocalized":
                    df_save_path = self.dire_wins_feats_vocalized / curr_h5_file.name
                print("Dataset will be saved at:", df_save_path)

            # check if the file already exists
            if df_save_path.exists() == False:

                single_rec_win_feat = Single_Recording_Windower_and_Feature_Extractor(
                    data_directory=self.data_directory,
                    h5_file_path=curr_h5_file,
                    window_size_s=self.win_size_sec,
                    manual_feature_extraction=self.manual_feature_extraction,
                    num_subwindows=self.num_subwins,
                )

                df_wins_feats = single_rec_win_feat.process_single_recording()
                if self.save_data:
                    df_wins_feats.to_hdf(df_save_path, key="wins_feats", mode="w")

                print(
                    f"Done with: SESSION:{df_wins_feats['session_id'].unique()} - BATCH: {df_wins_feats['batch_id'].unique()} - TYPE: {curr_h5_file.parent.name }"
                )

            else:
                print(f"File already exists at {df_save_path}, skipping")


if __name__ == "__main__":
    print("Extracting features....")

    project_root = Path(__file__).resolve().parent.parent.parent / "config"

    # Create Subject Config
    subject_config = SubjectConfig(project_root / "create_windows.yaml")

    win_and_feat_extractor = Global_Windower_and_Feature_Extractor(subject_config)
    win_and_feat_extractor.main()
