# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
This file is used to convert data from a .bio format into h5.
It is also used to visualize data after the acquisiton
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from I_data_preparation.read_bio_file import (
    process_all_recordings_for_subject,
    read_single_recording,
    find_bio_file,
)
from I_data_preparation.experimental_config import (
    RAW_DIRNAME,
    RAW_AND_FILTERED_DIRNAME,
    SILENT_DIRNAME,
    VOCALIZED_DIRNAME,
)


# ---------------------------------------------------------------------------
# User-editable settings
# ---------------------------------------------------------------------------


DATA_DIRECTORY = Path(r"path/to/your/data")
sub_ids = ["S01", "S02", "S03", "S04"]
# sub_ids = ["S01"]
LABEL_MODE = "word"  # "word" or "sentence"
process_all = True
# If Process_all = False -> select what you want to process
session_id = 1
batch_id = 5
condition = "vocalized"  # silent or vocalized

HP_CUTOFF = 20  # Frequency for HP Filter
PLI_CUTOFF = 50  # Frequency for PLI Filter

plots = False  # Set to True if you want to display plots


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    for sub_id in sub_ids:
        data_dire_raw = (
            DATA_DIRECTORY / RAW_DIRNAME / sub_id
        )  # directory where you placed your .bio recordings
        data_dire_processed = DATA_DIRECTORY / RAW_AND_FILTERED_DIRNAME / sub_id

        if data_dire_processed.exists() == False:
            data_dire_processed.mkdir(parents=True)
            print("Created directory: ", data_dire_processed)
        if Path(data_dire_processed / SILENT_DIRNAME).exists() == False:
            Path(data_dire_processed / SILENT_DIRNAME).mkdir()
        if Path(data_dire_processed / VOCALIZED_DIRNAME).exists() == False:
            Path(data_dire_processed / VOCALIZED_DIRNAME).mkdir()

        if process_all:
            # If process_all=True -> process everything found for that subject
            process_all_recordings_for_subject(
                data_dir_raw=data_dire_raw,
                data_dir_processed=data_dire_processed,
                subject=sub_id,
                hp_cutoff=HP_CUTOFF,
                notch_cutoff=PLI_CUTOFF,
                plot=plots,
                label_mode=LABEL_MODE,
            )

        else:
            ## =============== PROCESS A SINGLE RECORDING ==============================
            print("Processing single recording from:", data_dire_raw)
            bio_file_path = find_bio_file(
                data_dir_raw=data_dire_raw.parent,
                subject=sub_id,
                condition=condition,
                session_id=session_id,
                batch_id=batch_id,
            )

            print("Found file:", bio_file_path)

            save_fig_path = None  # change with a Path to save the figure
            emg_df = read_single_recording(
                bio_file_path,
                session_id,
                batch_id,
                HP_CUTOFF,
                PLI_CUTOFF,
                plot=plots,
                save_path=save_fig_path,
                label_mode=LABEL_MODE,
            )

            # Save it as an HDF5 file under the condition subfolder.
            file_path_processed = (data_dire_processed / condition / f"sess_{session_id}_batch_{batch_id}.h5")
            if file_path_processed.exists():
                print(f"File: {file_path_processed} already exists! Not saving...")
            else:
                emg_df.to_hdf(file_path_processed, key="emg", mode="w")
                print("Saved:", file_path_processed)
