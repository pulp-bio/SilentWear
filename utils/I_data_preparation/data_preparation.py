# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
This file is used to convert data from a .bio format into h5.
It is also used to visualize data after the acquisiton
"""

from pathlib import Path
from read_bio_file import *
from visualizations import *

# ================ USER_EDITABLE =========================

DATA_DIRECTORY = Path(r"... path_to_your_data")
sub_ids = ["S01", "S02", "S03", "S04"]
# sub_ids = ["S01"]
process_all = True
# If Process_all = False -> select what you want to process
session_id = 1
batch_id = 5
condition = "vocalized"  # silent or vocalized

HP_CUTOFF = 20  # Frequency for HP Filter
PLI_CUTOFF = 50  # Frequency for PLI Filter

plots = False  # Set to True if you want to display plots
# ============================================================


if __name__ == "__main__":

    for sub_id in sub_ids:
        data_dire_raw = (
            DATA_DIRECTORY / "raw" / sub_id
        )  # directory where you placed your .bio recordings
        data_dire_processed = DATA_DIRECTORY / "raw_and_processed" / sub_id

        if data_dire_processed.exists() == False:
            data_dire_processed.mkdir(parents=True)
            print("Created directory: ", data_dire_processed)
        if Path(data_dire_processed / "silent").exists() == False:
            Path(data_dire_processed / "silent").mkdir()
        print("Created directory: ", data_dire_processed)
        if Path(data_dire_processed / "vocalized").exists() == False:
            Path(data_dire_processed / "vocalized").mkdir()

        if process_all:
            # If process_all=True -> process everything found for that subject
            process_all_recordings_for_subject(
                data_dir_raw=data_dire_raw,
                data_dir_processed=data_dire_processed,
                subject=sub_id,
                hp_cutoff=HP_CUTOFF,
                notch_cutoff=PLI_CUTOFF,
                plot=plots,
            )

        else:
            ## =============== PROCESS A SINGLE RECORDING ==============================
            print("===")
            print(data_dire_raw)
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
                plot=True,
                save_path=None,
            )
            # save it as hdf file

            file_path_processed = data_dire_processed / f"sess_{session_id}_batch_{batch_id}.hf"
            if file_path_processed.exists():
                print(f"File:{file_path_processed} already exist! Not saving...")
            else:
                print("Saved:", file_path_processed)
