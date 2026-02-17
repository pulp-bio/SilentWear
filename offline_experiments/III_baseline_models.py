"""
Script to run Fine Tuning
(SUBJECT-SPECIFIC SWEEP VERSION)

Key changes (as requested):
- NO inter-session experiments at all (no Inter_Session_Model_Trainer, no baseline training).
- NEVER start from a pre-trained model:
  * first batch starts from RANDOM INIT (model_to_ft_path=None)
  * subsequent batches load previous FT checkpoint
- Directory root: instead of using base_template.parents[3] directly, insert "baseline_models".

Everything else (data loading, batch scheme, train/val split, FT summary writing) stays the same.
"""

import sys
from pathlib import Path
import json
from typing import Optional, Dict, Any, List
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.seeds import *  # noqa: F401,F403
from offline_experiments.Model_Fine_Tuner import Model_Fine_Tuner
from offline_experiments.general_utils import check_data_directories
from utils.general_utils import load_subjects_data, open_file


def build_ft_directory(model_base_folder: Path) -> Path:
    """
    Create a new subfolder ft_config_<N> under model_base_folder, choosing
    the first N that doesn't exist. Returns the created path.
    """
    model_base_folder = Path(model_base_folder)
    model_base_folder.mkdir(parents=True, exist_ok=True)

    for model_ft_id in range(10_000_000):  # practically infinite
        model_ft_folder = model_base_folder / f"bs_config_{model_ft_id}"
        try:
            model_ft_folder.mkdir(parents=True, exist_ok=False)  # atomic
            return model_ft_folder
        except FileExistsError:
            continue

    raise RuntimeError("Could not find a free ft_config_<N> directory name.")


def build_ft_config_dict(model_config: dict, ft_cfg: dict, save_path: Path = None) -> dict:
    """
    Build a fine-tuning model config starting from the base model_cfg.
    """
    model_config = copy.deepcopy(model_config)
    train_cfg = model_config["model"]["kwargs"]["train_cfg"]
    print("Train cfg:")
    print(train_cfg)

    optimizer_old = train_cfg["optimizer_cfg"]
    lr_new = ft_cfg["ft_lr"]
    optimizer_cfg_new = {"lr": lr_new, "name": optimizer_old["name"]}

    scheduler = train_cfg.get("scheduler", None)
    weight_decay = train_cfg.get("weight_decay", 0)
    es_patience = train_cfg.get("early_stop_patience", 10)

    model_config["model"]["kwargs"]["train_cfg"] = {
        "lr": lr_new,
        "num_epochs": ft_cfg["num_ft_epochs"],
        "optimizer_cfg": optimizer_cfg_new,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "early_stop_patience" : es_patience
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(model_config, f, indent=4)

    return model_config

def ft_config_dict_first_run(cfg: dict, lr_new: float = 1e-3, num_epochs_new: int = 50) -> dict:
    """
    Adjust FT config for the first run.
    Common behavior:
      - do NOT increase LR beyond base LR (use min)
      - do NOT increase epochs beyond base epochs (use min)  # i.e., "decrease/cap"
    """
    cfg = copy.deepcopy(cfg)

    train_cfg = cfg["model"]["kwargs"]["train_cfg"]

    optimizer_old = train_cfg["optimizer_cfg"]
    lr_base = float(train_cfg["lr"])  # fail loudly if missing
    epochs_base = int(train_cfg.get("num_epochs", 20))

    lr_final = max(lr_base, lr_new)
    epochs_final = max(epochs_base, num_epochs_new)
    es_patience = train_cfg.get("early_stop_patience", 10)

    print(f"Base lr: {lr_base} -> requested: {lr_new} -> using: {lr_final}")
    print(f"Base epochs: {epochs_base} -> requested: {num_epochs_new} -> using: {epochs_final}")

    optimizer_cfg_new = {"lr": lr_final, "name": optimizer_old["name"]}
    scheduler = train_cfg.get("scheduler", None)
    weight_decay = train_cfg.get("weight_decay", 0)

    cfg["model"]["kwargs"]["train_cfg"] = {
        "lr": lr_final,
        "num_epochs": epochs_final,
        "optimizer_cfg": optimizer_cfg_new,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "early_stop_patience" : es_patience, 
    }

    print("Building first run cfg")
    print("Early stop patience")

    return cfg

def return_batches_for_ft(ft_cfg: dict, df_for_ft: pd.DataFrame):
    """
    Return list of batch IDs to process for FT, plus last_batch (only meaningful in 'base' scheme).
    """
    if ft_cfg["batch_ft_scheme"] == "single_batch" and ft_cfg["single_batch_id"] is None:
        raise ValueError("single_batch_id must be set when batch_ft_scheme='single_batch'")

    if ft_cfg["batch_ft_scheme"] == "base":
        unique_batches = np.sort(df_for_ft["batch_id"].unique())
        last_batch = np.max(unique_batches)
        return unique_batches, last_batch

    if ft_cfg["batch_ft_scheme"] == "single_batch":
        return [ft_cfg["single_batch_id"]], None

    raise ValueError(f"Unknown batch_ft_scheme: {ft_cfg['batch_ft_scheme']}")


def run_bs_for(sub: str, cond: str):
    config_root = Path(__file__).resolve().parent.parent / "config"
    ft_config_path = config_root / "bs_configs.yaml"
    ft_cfg = open_file(ft_config_path)
    print(ft_cfg)

    # --- load base cfg + override subject/condition for THIS run ---
    base_cfg = open_file(config_root / "base_models_config.yaml")    
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["data"]["subject_id"] = sub
    base_cfg["condition"] = cond

    # --- model cfg (same for all runs) ---
    model_config_path = config_root / "models_configs" / "speechnet_base_with_padding.yaml"
    model_cfg = open_file(model_config_path)

    base_template = Path(ft_cfg["model_parent_directory"])
    model_name = model_cfg["model"]["name"]
    MODEL_ID = ft_cfg.get("model_name_id")

    # IMPORTANT: insert "baseline_models" as root component
    model_base_folder = (
        base_template.parents[3]
        / "baseline_models"
        / sub
        / cond
        / model_name
        / MODEL_ID
    )

    model_base_folder.mkdir(parents=True, exist_ok=True)
    print(f"[{sub} | {cond}] Using FT base folder: {model_base_folder}")

    # --- FT output dir ---
    model_ft_base_folder = build_ft_directory(model_base_folder)

    # Save a minimal run_cfg-like record for reproducibility (since we no longer have run_cfg.json)
    run_cfg_min = {
        "base_cfg": base_cfg,
        "model_cfg": model_cfg,
        "note": "FT-from-scratch: no inter-session training; first batch random init.",
    }
    with open(model_ft_base_folder / "run_cfg_min.json", "w") as f:
        json.dump(run_cfg_min, f, indent=4)

    # --- build ft cfg for trainer ---
    new_ft_model_cfg = build_ft_config_dict(
        model_cfg,
        ft_cfg,
        save_path=model_ft_base_folder / "bs_cfg.json",
    )
    

    

    # --- load data ---
    subjects = base_cfg["data"]["subject_id"]
    condition = base_cfg["condition"]

    win_size_ms = int(base_cfg["window"]["window_size_s"] * 1000)
    main_data_dire = base_cfg["data"]["data_directory"]

    data_directories = check_data_directories(
        main_data_directory=main_data_dire,
        all_subjects_models=False,
        sub_id=subjects,
        condition=condition,
        window_size_ms=win_size_ms,
        base_config=base_cfg,
    )

    df = load_subjects_data(data_directories=data_directories, print_statistics=False)
    print(f"[{sub} | {cond}] Dataset subjects={df['subject_id'].unique()} sessions={np.sort(df['session_id'].unique())}")

    # --- session loop (replaces inter-session folds/cv_summary) ---
    experiment_summary = []
    sessions = np.sort(df["session_id"].unique())

    for test_session in sessions:
        df_test_session = df[df["session_id"] == test_session]
        if len(df_test_session) == 0:
            continue

        unique_batches, last_batch = return_batches_for_ft(ft_cfg, df_test_session)

        # Keep a similar naming scheme, but fold_id is now the session id
        fold_id = int(test_session)

        print(f"\n[{sub} | {cond}] ===== Train on session={test_session} batches={unique_batches.tolist()} =====")

        prev_round_of_ft = 0
        for batch_id_for_ft in unique_batches:
            print("-----------------------------")

            # NEVER load pretrained weights:
            # - First batch => RANDOM INIT (model_to_ft_path=None)
            # - Next batches => continue from previous FT checkpoint
            if batch_id_for_ft == unique_batches[0]:
                model_to_ft = None
                model_to_fine_tune_name = "RANDOM_INIT"

                print("First batch, check lr/epochs")

                lr_first = ft_cfg.get("lr_first_train", new_ft_model_cfg["model"]["kwargs"]["train_cfg"]["lr"])
                num_epochs_first = ft_cfg.get("num_epochs_first_train", new_ft_model_cfg["model"]["kwargs"]["train_cfg"]["num_epochs"])

                print("Lr for first run:", lr_first)
                print("Num epochs first:", num_epochs_first)

                # IMPORTANT: use a *separate* cfg for this batch
                batch_model_cfg = ft_config_dict_first_run(
                    new_ft_model_cfg,
                    lr_new=float(lr_first),
                    num_epochs_new=int(num_epochs_first)
                )
            else:
                model_to_ft = model_ft_base_folder / f"session_{fold_id}_bs_{batch_id_for_ft-1}.pt"
                model_to_fine_tune_name = model_to_ft.name
                batch_model_cfg = new_ft_model_cfg

            new_model_save_path = model_ft_base_folder / f"session_{fold_id}_bs_{batch_id_for_ft}.pt"

            df_batch = df_test_session[df_test_session["batch_id"] == batch_id_for_ft]
            # Balance if we have also the rest class
            if base_cfg["experiment"]["include_rest"]:
                min_samples = df_batch['Label_int'].value_counts().min()
                # downsample majority class
                idx_rest = df_batch[df_batch['Label_str']=='rest'].index.values
                index_rest_ds = df_batch[df_batch['Label_str']=='rest'].sample(n=min_samples, random_state=base_cfg["experiment"]["seed"]).index.values
                idx_to_drop = np.setdiff1d(idx_rest, index_rest_ds)
                df_batch = df_batch.drop(index=idx_to_drop)
    
            df_for_ft_train, df_for_ft_val = train_test_split(
                df_batch,
                test_size=0.3,
                shuffle=True,
                random_state=42,
                stratify=df_batch["Label_int"],
            )
            # df_for_ft_train, df_for_ft_val = train_test_split(
            #     df_batch,
            #     test_size=0.3,
            #     shuffle=True,
            #     random_state=42,
            #     stratify=df_batch["Label_int"],
            # )

            model_fine_tuner = Model_Fine_Tuner(
                base_cfg=base_cfg,
                model_cfg=batch_model_cfg,
                model_to_ft_path=model_to_ft,  # None => random init
                new_model_save_path=new_model_save_path,
                ft_cfg_settings=ft_cfg,
                df_for_ft_train=df_for_ft_train,
                df_for_ft_val=df_for_ft_val,
            )

            metrics_before = model_fine_tuner.test_zero_shot_acc()

            row = {
                "subject": sub,
                "condition": cond,
                "test_session": int(test_session),
                "model_to_fine_tune_name": model_to_fine_tune_name,
                "zero_shot_test_batch": int(batch_id_for_ft),
                "num_prev_ft_rounds": int(prev_round_of_ft),
                "zero_shot_balanced_acc": float(metrics_before["balanced_accuracy"]),
                "zero_shot_cm": json.dumps(np.asarray(metrics_before["confusion_matrix"]).tolist()),
                "new_model_name": new_model_save_path.name,
            }

            # keep your behavior: FT on all but last batch (base scheme) OR always in single-batch mode
            if batch_id_for_ft < np.max(unique_batches) or last_batch is None:
                model_fine_tuner.main_ft()

            prev_round_of_ft += 1
            experiment_summary.append(row)
            pd.DataFrame(experiment_summary).to_csv(model_ft_base_folder / "baseline_summary.csv", index=False)

            print("-----------------------------")


def main():
    subjects = ["S01", "S02", "S03", "S04"]
    conditions = ["silent", "vocalized"]

    for sub in subjects:
        for cond in conditions:
            print("\n" + "=" * 80)
            print(f"Running Fine-Tuning FROM SCRATCH | subject={sub} | condition={cond}")
            print("=" * 80)
            run_bs_for(sub, cond)


if __name__ == "__main__":
    main()
