"""
BIOGUI Recording Loader and Preprocessing Utilities
===================================================

This module provides utilities to:
1) Read `.bio` recordings produced by BIOGUI / BioGAP.
2) Parse session/batch metadata from filenames.
3) Convert raw recordings into a labeled EMG pandas DataFrame.
4) Apply preprocessing (HP + notch filtering) and save processed outputs.

Expected directory convention (raw):
    <data_dir_raw>/<subject>/<condition>/*.bio
where condition is typically: {silent, vocalized}

Processed outputs:
    - Per-recording HDF5 files (key="emg")
    - A per-subject CSV index (summary_file.csv)

Notes
-----
- The `.bio` format is parsed according to the file header and per-signal metadata.
- Trigger labels are mapped using `ORIGINAL_LABELS` from `experimental_config`.
"""

import struct
import numpy as np
from pathlib import Path
import sys
import pandas as pd
import re 

PROJECT_ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(PROJECT_ROOT))
from I_data_preparation.emg_processing import *
from I_data_preparation.experimental_config import * 
from I_data_preparation.visualizations import * 

BIO_RE = re.compile(r"sess_(?P<session>\d+)_batch_(?P<batch>\d+)(?:_[^_]*)?_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.bio$")

def read_bio_file(file_path: str) -> dict:
    """
    Parameters
    ----------
    file_path : str
        Path to the .bio file.

    Returns
    -------
    dict
        Dictionary containing timestamp, signals and trigger.
    """
    dtypeMap = {
        "?": np.dtype("bool"),
        "b": np.dtype("int8"),
        "B": np.dtype("uint8"),
        "h": np.dtype("int16"),
        "H": np.dtype("uint16"),
        "i": np.dtype("int32"),
        "I": np.dtype("uint32"),
        "q": np.dtype("int64"),
        "Q": np.dtype("uint64"),
        "f": np.dtype("float32"),
        "d": np.dtype("float64"),
    }

    # Read data
    with open(file_path, "rb") as f:
        # Read number of signals
        n_signals = struct.unpack("<I", f.read(4))[0]
        
        #print("N signals set to, ", n_signals)

        # Read other metadata
        fs_base, n_samp_base = struct.unpack("<fI", f.read(8))
        signals = {}
        for _ in range(n_signals):
            sig_name_len = struct.unpack("<I", f.read(4))[0]
            sig_name = struct.unpack(f"<{sig_name_len}s", f.read(sig_name_len))[
                0
            ].decode()
            fs, n_samp, n_ch, dtype = struct.unpack("<f2Ic", f.read(13))
            dtype = dtypeMap[dtype.decode("ascii")]

            # Initialize signal array
            signals[sig_name] = {
                "fs": fs,
                "n_samp": n_samp,
                "n_ch": n_ch,
                "dtype": dtype,
            }

        # Read whether the trigger is available
        is_trigger = struct.unpack("<?", f.read(1))[0]

        # Read actual signals:
        # 1. Timestamp
        ts = np.frombuffer(f.read(8 * n_samp_base), dtype=np.float64).reshape(
            n_samp_base, 1
        )
        signals["timestamp"] = {"data": ts, "fs": fs_base}

        # 2. Signals data
        for sig_name, sig_data in signals.items():
            if sig_name == "timestamp":
                continue

            n_samp = sig_data.pop("n_samp")
            n_ch = sig_data.pop("n_ch")
            dtype = sig_data.pop("dtype")
            data = np.frombuffer(
                f.read(dtype.itemsize * n_samp * n_ch), dtype=dtype
            ).reshape(n_samp, n_ch)
            sig_data["data"] = data

        # 3. Trigger (optional)
        if is_trigger:
            trigger = np.frombuffer(f.read(), dtype=np.int32).reshape(-1, 1)
            for sig_name, sig_data in signals.items():
                if sig_name == "timestamp":
                    # Align the timestamp length with the trigger length
                    sig_data["data"] = sig_data["data"][: len(trigger), :]
                else:
                    samples_per_packet = sig_data["data"].shape[0] // n_samp_base
                    # Align the signal length with the trigger length
                    sig_data["data"] = sig_data["data"][
                        : len(trigger) * samples_per_packet, :
                    ]
            signals["trigger"] = {"data": trigger, "fs": fs_base}

    return signals


def read_single_recording(bio_file_path, session_id, batch_id, hp_cutoff, notch_cutoff, plot=True, save_path=None):
    
    
    if bio_file_path.exists() == False:
        print(f"File: {bio_file_path} does not exist, provide a valid file")
        sys.exit()
    
    # check extension
    if bio_file_path.suffix == '.bio':
        signals = read_bio_file(str(bio_file_path))
        emg_df = prepare_dataset(signals, hp_cutoff=hp_cutoff, notch_cutoff=notch_cutoff)

    emg_df['session_id'] = session_id
    emg_df['batch_id'] = batch_id
    print(emg_df.columns)

    
    # Drop channels 12 and 13 (both raw + filtered if present)
    drop_idxs = [11, 12]                # FIXED!! was
    drop_cols = []
    for i in drop_idxs:
        drop_cols += [f"Ch_{i}", f"Ch_{i}_filt"]

    plot_emg_color_by_label(emg_df, fs=FS, use_filtered=True, save_path=save_path)
    emg_df = emg_df.drop(columns=[c for c in drop_cols if c in emg_df.columns])

    # if plot:
    #     plot_emg_color_by_label(emg_df, fs=FS, use_filtered=True, save_path=save_path)

    # drop channel 12 and 13
    print(emg_df['session_id'])
    return emg_df

def find_bio_file(data_dir_raw, subject, condition, session_id, batch_id):
    """
    Search for the .bio file matching sess_<id>_batch_<id> even if extra text exists.
    """

    base_dir = Path(data_dir_raw) / subject / condition

    # Pattern matches anything after batch_id
    pattern = f"sess_{session_id}_batch_{batch_id}*.bio"

    matches = list(base_dir.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No .bio file found for session={session_id}, batch={batch_id} in {base_dir}"
        )

    if len(matches) > 1:
        print("Warning: Multiple matches found, using first one:")
        for m in matches:
            print("  ", m)

    return matches[0]



def parse_bio_filename(path: Path):
    """
    Returns (session_id:int, batch_id:int, timestamp:str) or None if no match.
    """
    m = BIO_RE.search(path.name)
    if not m:
        return None
    return int(m.group("session")), int(m.group("batch")), m.group("ts")

def processed_path_for(raw_path: Path, data_dir_processed: Path, condition:str, session_id: int, batch_id: int):
    """
    Choose a consistent processed filename.
    """
    return data_dir_processed / condition / f"sess_{session_id}_batch_{batch_id}.h5"

def update_index_csv(index_csv_path: Path, rows: list[dict]):
    """
    Append new rows to index CSV, de-duplicate by raw_path (or processed_path).
    """
    new_df = pd.DataFrame(rows)
    if index_csv_path.exists():
        old_df = pd.read_csv(index_csv_path)
        df = pd.concat([old_df, new_df], ignore_index=True)
        # de-dup: keep last entry if reprocessed
        df = df.drop_duplicates(subset=["raw_path"], keep="last")
    else:
        df = new_df

    df = df.sort_values(by=["subject", "condition", "session_id", "batch_id"]).reset_index(drop=True)
    df.to_csv(index_csv_path, index=False)

def process_all_recordings_for_subject(data_dir_raw: Path, 
                                       data_dir_processed: Path, 
                                       subject: str, 
                                       hp_cutoff: int, 
                                       notch_cutoff: int, 
                                       plot : bool):
    """
    Scans DATA/raw/<SUB_ID>/(silent|vocalized)/*.bio, processes each, saves to HDF5,
    and updates an index CSV.
    """
    data_dir_processed.mkdir(parents=True, exist_ok=True)

    # Put the index CSV in the subject processed folder
    index_csv_path = data_dir_processed / "summary_file.csv"

    # Find all .bio files under the subject raw folder
    bio_files = sorted(data_dir_raw.rglob("*.bio"))
    
    #sys.exit()
    if not list(bio_files):
        print(f"No .bio files found under: {data_dir_raw}")
        sys.exit()
    index_rows = []
    n_done, n_skip, n_fail = 0, 0, 0

    for bio_path in bio_files:
        print("Processing file:", bio_path)
        # condition inferred from folder name if available
        # expected: .../S01/silent/... or .../S01/vocalized/...
        parts_lower = [p.lower() for p in bio_path.parts]
        condition = "unknown"
        if "silent" in parts_lower:
            condition = "silent"
        elif "vocalized" in parts_lower:
            condition = "vocalized"
        

        if bio_path.suffix == '.bio':
            parsed = parse_bio_filename(bio_path)
        if parsed is None:
            print(f"[SKIP] Could not parse session/batch/timestamp from filename: {bio_path.name}")
            n_fail += 1
            continue

        session_id, batch_id, ts = parsed
        out_path = processed_path_for(bio_path, data_dir_processed, condition, session_id, batch_id)

        if out_path.exists():
            print(f"[SKIP] Already processed: {out_path.name}")
            n_skip += 1
            # still ensure it’s indexed (optional): comment out if you only want new entries
            index_rows.append({
                "subject": subject,
                "condition": condition,
                "session_id": session_id,
                "batch_id": batch_id,
                "timestamp": ts,
                "raw_path": str(bio_path.name),
                "processed_path": str(out_path.name),
                "status": "already_exists",
            })
            continue

        try:
            print(f"[PROC] {bio_path.name}")
            emg_df = read_single_recording(bio_path, session_id, batch_id, hp_cutoff, notch_cutoff, plot=plot)
            
            emg_df.to_hdf(out_path, key="emg", mode="w")
            
            # check if the file has been saved
            if out_path.exists():
                print(f"[OK] Saved: {out_path}")
                print_label_statistics(emg_df)
            else:
                print(f"[FAIL]: file could not be saved under path:", out_path)
                print(f"Code will stop executing, check Path structure!")
            n_done += 1

            index_rows.append({
                "subject": subject,
                "condition": condition,
                "session_id": session_id,
                "batch_id": batch_id,
                "timestamp": ts,
                "raw_path": str(bio_path.name),
                "processed_path": str(out_path.name),
                "status": "processed",
            })

        except Exception as e:
            print(f"[FAIL] {bio_path.name} -> {e}")
            n_fail += 1
            index_rows.append({
                "subject": subject,
                "condition": condition,
                "session_id": session_id,
                "batch_id": batch_id,
                "timestamp": ts,
                "raw_path": str(bio_path.name),
                "processed_path": str(out_path.name),
                "status": f"failed: {type(e).__name__}",
            })

    # Write/update the subject index
    if index_rows:
        update_index_csv(index_csv_path, index_rows)

    print("\n=== Summary ===")
    print(f"Processed: {n_done}")
    print(f"Skipped (exists): {n_skip}")
    print(f"Failed/Unparsed: {n_fail}")
    print(f"Index CSV: {index_csv_path}")

def data_losses_check(counter):
    """
    Docstring for data_losses_check
    
    :param counter: counter retured by the GUI
    This function check if data were lost during the data collection

    """

    counter_reconstructed = np.zeros(len(counter), dtype=np.int32)
    counter_reconstructed[0] = counter[0]
    prev_counter = counter[0]
    losses_cnts = 0

    for i, curr_counter in enumerate(counter[1:]):
        
        # Handle counter reset
        if prev_counter == 255:
            # current counter value should be 0. If not, data were lost
            losses = curr_counter
            losses_cnts+=losses
        else:
            losses = curr_counter - (prev_counter+1)
            losses_cnts+=losses

        counter_reconstructed[i+1] = counter_reconstructed[i] + losses + 1
        prev_counter = curr_counter
    # Sometimes counter does not start from 0, reset
    counter = counter - counter[0]
    print(f"Lost:{losses} samples")
    # Note: first value of the counter might not be 0. This is how the FW is designed
    if losses_cnts!=0:
        print(f"[COUNTER_CHECK], Lost: {losses_cnts} samples")
        count_diffs = np.diff(counter_reconstructed)
        
        if np.sum(count_diffs!=1)!=losses:
            print(f'[COUNTER CHECK], dimension mismatch!')
            sys.exit()


def print_label_statistics(emg_df):

    print("\n================ Label Statistics ================")

    
    # ---- Unique labels ----
    print("\nUnique Label_int values:")
    print(emg_df["Label_int"].unique())

    print("\nUnique Label_str values:")
    print(emg_df["Label_str"].unique())

    # ---- Total sample count per label (non-contiguous) ----
    print("\nTotal sample counts per label:")
    print(emg_df["Label_int"].value_counts())

    # ---- Total sample count per label (string form) ----
    print("\nTotal sample counts per label (Label_str):")
    print(emg_df["Label_str"].value_counts())

    # ---- Count how many contiguous segments per label ----
    run_id = (emg_df["Label_int"] != emg_df["Label_int"].shift()).cumsum()

    segment_counts = (
        emg_df.groupby(run_id)["Label_int"]
        .first()
        .value_counts()
    )

    print("\nNumber of contiguous segments (runs) per label:")
    print(segment_counts)

    print("\n=================================================\n")
def prepare_dataset(signals, hp_cutoff, notch_cutoff):
    
    # ------------------------------------------------------------
    # 1. Extract signals
    emg_data = signals["biogap"]["data"]
    counter = np.hstack(signals["counter"]["data"])
    # check for data losses 
    # ========== TO-DO: implement extra adjstement in case of data lossess (not needed for recorded data) ============
    data_losses_check(counter)
    trigger = np.hstack(signals["trigger"]["data"])
    # 2. Expand trigger (1 packet contains 4 EMG samples)
    trigger = np.repeat(trigger, 4)
    # 3. Convert integer labels into string labels
    labels = np.vectorize(ORIGINAL_LABELS.get)(trigger)
    # 4. Build EMG DataFrame
    emg_df = pd.DataFrame(
        emg_data,
        columns=[f"Ch_{i}" for i in range(emg_data.shape[1])]
    )

    # Add labels
    emg_df["Label_int"] = trigger
    emg_df["Label_str"] = labels
    
    # print("\nLabel Summary")
    # print("Unique labels found:", emg_df["Label_str"].unique())

    # print("\nLabel counts:")
    # print(emg_df["Label_str"].value_counts())

    # print("\nNumber of unique labels:", emg_df["Label_str"].nunique())

    # Filter data
    for i in range(emg_data.shape[1]):
        emg_df[f'Ch_{i}_filt'] = apply_filters(emg_data[:,i], FS, highpass_cutoff=hp_cutoff, notch_cutoff=notch_cutoff)

    # Trim recording considering only first and last label
    first_label_loc = emg_df['Label_int'][emg_df['Label_int'] != 0].index[0]
    # give some margin (1 sec before)
    first_label_loc = first_label_loc - FS

    last_label_loc= emg_df['Label_int'][emg_df['Label_int'] != 0].index[-1]
    # give some margin (1 sec after)
    last_label_loc = last_label_loc + FS

    emg_df = emg_df.iloc[first_label_loc:last_label_loc]


    return emg_df



if __name__=='__main__':
    # Modify here with the path with your raw data (.bio file)
    # Convention strucutre: \data\raw\<subject_id>\<condition>
    sub_raw_data_folder = Path(r"\DATA\raw\<subject_id>\<condition>")
    # read all bio files
    all_bios_in_folder = sub_raw_data_folder.rglob("*.bio")
    for curr_bio in all_bios_in_folder:
        #print("curr bio is")
        if curr_bio.suffix == '.bio':
            parsed = parse_bio_filename(curr_bio)
        if parsed is None:
            print(f"[SKIP] Could not parse session/batch/timestamp from filename: {curr_bio.name}")
            sys.exit()

        session_id, batch_id, ts = parsed
        emg_df=read_single_recording(curr_bio, session_id, batch_id, 20, 50, plot=False)

        print(print_label_statistics(emg_df))
    
        
        
