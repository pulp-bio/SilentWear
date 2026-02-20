# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
This file contains General Classes and Functions shared across the project
"""

import yaml
from pathlib import Path
import pandas as pd
import json
from typing import Any, Iterable

######################################### SUBJECT CONFIGURATION CLASS #################################################


class SubjectConfig:
    def __init__(self, yaml_file=Path("config.yaml")):
        with open(yaml_file, "r") as f:
            cfg = yaml.safe_load(f)

        self.data_directory = Path(cfg["data"]["data_directory"])
        self.subject_id = cfg["data"]["subject_id"]

        self.raw_dir = self.data_directory / "raw" / self.subject_id
        self.processed_dir = self.data_directory / "processed" / self.subject_id
        self.wins_and_feats_dir = self.data_directory / "win_and_feats" / self.subject_id

        # self.silent_dir = self.processed_dir / "silent"
        # self.vocalized_dir = self.processed_dir / "vocalized"

        self.window_size_s = cfg["window"]["window_size_s"]

        self.manual_feature_extraction = cfg["feature_extraction"]["manual_feature_extraction"]
        self.num_subwindows = cfg["feature_extraction"]["num_subwindows"]

        self.save_wins_and_feats = cfg["save_wins_and_feats"]


######################################### LOADING UTILS #################################################


### TO-DO: remove these functions, fix scripts using them using open_file (added later)
def load_yaml_config(config_path=Path("config.yaml")):
    with open((config_path), "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_yaml(path: Path) -> dict:
    print("Loading file from", path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def open_file(file_path: Path) -> Any:
    """
    Open a file and return its content based on file extension.
    Supported formats: .json, .csv
    """

    if not file_path.is_file():
        print(f"File {file_path} note found!")
        return None

    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    elif suffix == ".csv":
        return pd.read_csv(file_path)

    elif suffix == ".yaml":
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_all_h5files_from_folder(
    data_directory: Path, key: str = None, print_statistics: bool = False
) -> pd.DataFrame:
    """
    Load and concatenate all `.h5` files found recursively inside a folder.

    Parameters
    ----------
    data_directory : Path
        Root directory where `.h5` files will be searched recursively.

    key : str, optional
        Dataset key inside the HDF5 files (required if files contain multiple datasets).
        If None, files will be skipped.

    print_statistics: bool, optional
        True if we want to print statistics (number of sessions, batches, label distributions from loaded data)

    Returns
    -------
    pd.DataFrame
        A single DataFrame containing all concatenated data from the loaded files.

    Notes
    -----
    This function also prints basic statistics about loaded sessions, batches, and labels.
    """
    # 1. Find all HDF5 files
    h5_files = list(data_directory.rglob("*.h5"))

    if len(h5_files) == 0:
        print(f"No .h5 files found in: {data_directory}")
        return pd.DataFrame()

    print(f"Found {len(h5_files)} .h5 files in: {data_directory}")

    # 2. Load and concatenate all DataFrames
    df_list = []

    for file_path in h5_files:
        if key is None:
            print(f"Skipping {file_path.name} (no key provided)")
            continue

        try:
            # Load file
            df = pd.read_hdf(file_path, key=key)
            # Extract subject_id and condition from path. Example: .../S01/silent/WIN_1400/file.h5
            # ----------------------------------------------------
            subject_id = file_path.parts[-4]  # S01
            condition = file_path.parts[-3]  # silent or vocalized
            df["subject_id"] = subject_id
            df["condition"] = condition
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")

    if len(df_list) == 0:
        print("No data loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    df_all = pd.concat(df_list, ignore_index=True)
    if print_statistics:
        print_dataset_summary_statistics(df_all)

    return df_all


def load_subjects_data(
    data_directories: Iterable[Path], print_statistics: bool = False
) -> pd.DataFrame:
    """
    Load and concatenate data from a list of data directories.

    :param data_directories: Iterable of paths containing subject data files
    :return: Concatenated DataFrame with all loaded data
    """
    df = pd.DataFrame()
    for curr_data_dire in data_directories:
        df_curr = load_all_h5files_from_folder(
            curr_data_dire, key="wins_feats", print_statistics=print_statistics
        )
        if df_curr is not None and not df_curr.empty:
            df = pd.concat((df, df_curr))
    return df


######################################### Datasets UTILS #################################################


def print_dataset_summary_statistics(df):
    print("\n Loaded DataFrame Summary")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    unique_subjects = df["subject_id"].unique()
    print(f"Subjects in Dataset: {unique_subjects}")
    unique_conditions = df["condition"].unique()
    print(f"Conditions in dataset:{unique_conditions}")

    # Sessions overview
    unique_sessions = df["session_id"].unique()
    print(f"\nContains data from {len(unique_sessions)} sessions:")

    for subject_id in unique_subjects:
        print("\n----------------------------------------")

        df_subj = df[df["subject_id"] == subject_id]
        print("SUBJECT:", df_subj["subject_id"].unique())
        for session_id in unique_sessions:
            df_sess = df_subj[df_subj["session_id"] == session_id]
            for condition in unique_conditions:
                df_condition = df_sess[df_sess["condition"] == condition]
                batches = df_condition["batch_id"].unique()
                labels = df_condition["Label_str"].value_counts()
                print(f"Session: {session_id} - condition: {condition}")
                print(f"  Unique batches: {len(batches)}")
                print(f"  Labels distribution:\n{labels}")
