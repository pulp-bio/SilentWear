"""
Script to run Fine Tuning
(REVERTED BACK to the subject-specific sweep version)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.seeds import *  
from models.TorchTrainer import evaluate_model
from models.utils import load_pretrained_model
from offline_experiments.Model_Fine_Tuner import Model_Fine_Tuner
from offline_experiments.Model_Master import Model_Master
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import check_data_directories
from utils.general_utils import load_subjects_data, open_file
from sklearn.model_selection import train_test_split

import json
from typing import Optional, Dict, Any, List
import copy
import pandas as pd
import numpy as np


def build_ft_directory(model_base_folder: Path) -> Path:
    """
    Create a new subfolder ft_config_<N> under model_base_folder, choosing
    the first N that doesn't exist. Returns the created path.
    """
    model_base_folder = Path(model_base_folder)
    model_base_folder.mkdir(parents=True, exist_ok=True)

    for model_ft_id in range(10_000_000):  # practically infinite
        model_ft_folder = model_base_folder / f"ft_config_{model_ft_id}"
        try:
            model_ft_folder.mkdir(parents=True, exist_ok=False)  # atomic
            return model_ft_folder
        except FileExistsError:
            continue

    raise RuntimeError("Could not find a free ft_config_<N> directory name.")


def check_base_models_exist(
    model_base_folder: Path,
    model_to_ft_name: Optional[str] = None,
    expected_in_dir: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """
    Check that base models exist in model_base_folder.

    If model_to_ft_name is provided, returns [that_model_path] after validating it exists.
    Otherwise, returns a list of expected model paths based on expected_in_dir.

    expected_in_dir format:
      {
        "base_models_prefix": "leave_one_session_out_fold",
        "expected_num": 3
      }
    """
    if expected_in_dir is None:
        expected_in_dir = {
            "base_models_prefix": "leave_one_session_out_fold",
            "expected_num": 3,
        }

    if model_base_folder is None:
        raise ValueError("model_base_folder not provided")

    model_base_folder = Path(model_base_folder)

    if not model_base_folder.exists():
        raise FileNotFoundError(f"Model base folder does not exist: {model_base_folder}")
    if not model_base_folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {model_base_folder}")

    # User requested a specific model
    if model_to_ft_name:
        model_path = model_base_folder / model_to_ft_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find model file: {model_path}")
        return [model_path]

    # Otherwise, check expected set
    prefix = expected_in_dir.get("base_models_prefix", "leave_one_session_out_fold")
    expected_num = int(expected_in_dir.get("expected_num", 0))
    if expected_num <= 0:
        raise ValueError(f"expected_num must be > 0 (got {expected_num})")

    ft_models_in_folder: List[Path] = []
    for model_id in range(1, expected_num + 1):
        model_path = model_base_folder / f"{prefix}_{model_id}.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find expected model: {model_path}")
        ft_models_in_folder.append(model_path)

    return ft_models_in_folder


def build_ft_config_dict(model_config: dict, ft_cfg: dict, save_path: Path = None) -> dict:
    """
    Build a fine-tuning model config starting from the base run_cfg model_cfg.
    """
    model_config = copy.deepcopy(model_config)
    train_cfg = model_config["model"]["kwargs"]["train_cfg"]

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
        "early_stop_patience" : es_patience, 
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(model_config, f, indent=4)

    return model_config


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


def run_ft_for(sub: str, cond: str):
    config_root = Path(__file__).resolve().parent.parent / "config"
    ft_config_path = config_root / "ft_configs.yaml"
    ft_cfg = open_file(ft_config_path)

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

    #model_parent_directory: "/scratch2/gspacone/sensors_2026_speech_models/models/inter_session_ft/S01/vocalized/speechnet_base/model_1"

    #sys.exit()
    model_base_folder = (
        base_template.parents[3]                # go up to inter_session_ft -> we will create another folder inside
        / sub
        / cond
        / model_name
        / MODEL_ID
    )

    print(f"[{sub} | {cond}] Using base model folder:  {model_base_folder}")
    
    retrain_from_scratch = ft_cfg.get("retrain_intersessions", False)

    if retrain_from_scratch or not model_base_folder.exists():
        # We enter here if we are running this experiment for the first time or if we explicitly want to re-train
        # Results should be in line with what achieved with the "inter_session_models.py"
        print(f"[{sub} | {cond}] Need to (re)train inter-session models from scratch!")
        trainer = Inter_Session_Model_Trainer(
            base_config=base_cfg,
            model_config=model_cfg,
            experiment_subdir="inter_session_ft"
        )
        trainer.main()
        model_base_folder = Path(trainer.model_dire)


    # --- load run cfg + csv summary from that folder ---
    run_cfg = open_file(model_base_folder / "run_cfg.json")
    csv_summary = open_file(model_base_folder / "cv_summary.csv")

    model_to_ft_name = ft_cfg.get("model_ft_name", None)
    ft_models_in_folder = check_base_models_exist(
        model_base_folder=model_base_folder,
        model_to_ft_name=model_to_ft_name,
        expected_in_dir={"base_models_prefix": "leave_one_session_out_fold", "expected_num": 3},
    )

    print(f"[{sub} | {cond}] Fine-tuning will be done on models:\n{ft_models_in_folder}")

    # --- FT output dir ---
    model_ft_base_folder = build_ft_directory(model_base_folder)

    # --- build ft cfg for trainer ---
    base_cfg = run_cfg["base_cfg"]
    model_config = run_cfg["model_cfg"]
    new_ft_model_cfg = build_ft_config_dict(model_config, ft_cfg, save_path=model_ft_base_folder / "ft_cfg.json")

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
    print(f"[{sub} | {cond}] Dataset subjects={df['subject_id'].unique()} batches={df['batch_id'].unique()}")

    # --- FT loop ---
    experiment_summary = []
    for base_model_to_ft in ft_models_in_folder:
        model_ft_name = base_model_to_ft.name
        print(f"\n[{sub} | {cond}] ===== Starting to fine tune model: {model_ft_name} =====")

        # Initialize also the Base model and load its weights. 
        # We will always test the performance of this model on every new batch for comparison
        model_intersess = load_pretrained_model(base_cfg=base_cfg, model_cfg=model_config, pretrained_model_path=base_model_to_ft)

        fold_id = base_model_to_ft.name.split("fold_")[-1].split(".pt")[0]

        csv_summary_row_interest = csv_summary[csv_summary["fold_num"] == int(fold_id)]
        test_session = csv_summary_row_interest["test_session"].values[0]
        # Here, it should always be fold1 - testsession1; fold2- tessession2; fold3 - testsession3
        print(f"[{sub} | {cond}] Model was tested on session: {test_session}")

        df_test_session = df[df["session_id"] == test_session]
        unique_batches, last_batch = return_batches_for_ft(ft_cfg, df_test_session)

        prev_round_of_ft = 0
        for batch_id_for_ft in unique_batches:
            print("-----------------------------")

            model_to_ft = (base_model_to_ft if batch_id_for_ft == unique_batches[0]
                else (model_ft_base_folder / f"fold_{fold_id}_ft_{batch_id_for_ft-1}.pt")
            )

            new_model_save_path = model_ft_base_folder / f"fold_{fold_id}_ft_{batch_id_for_ft}.pt"

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

            model_fine_tuner = Model_Fine_Tuner(
                base_cfg=base_cfg,
                model_cfg=new_ft_model_cfg,
                model_to_ft_path=model_to_ft,
                new_model_save_path=new_model_save_path,
                ft_cfg_settings=ft_cfg,
                df_for_ft_train=df_for_ft_train,
                df_for_ft_val=df_for_ft_val,
            )

            # These are the metrics BEFORE, on the fine-tuned model
            metrics_before = model_fine_tuner.test_zero_shot_acc()
            batch_loader = model_fine_tuner.model_master.trainer_manager.test_loader 
            metrics_without_ft, _, _ = evaluate_model(model_intersess, batch_loader)

            # IMPORTANT. On batch_id=1 (first batch), metrics_without ft and betrics before must match

            row = {
                "subject": sub,
                "condition": cond,
                "base_model": base_model_to_ft.name,
                "test_session": int(test_session),
                "model_to_fine_tune_name": model_to_ft.name,
                "zero_shot_test_batch": int(batch_id_for_ft),
                "num_prev_ft_rounds": int(prev_round_of_ft),
                "zero_shot_balanced_acc": float(metrics_before["balanced_accuracy"]),
                "zero_shot_cm": json.dumps(np.asarray(metrics_before["confusion_matrix"]).tolist()),
                "balanced_acc_no_ft": float(metrics_without_ft["balanced_accuracy"]),
                "cm_no_ft": json.dumps(np.asarray(metrics_without_ft["confusion_matrix"]).tolist()),
                "new_model_name": new_model_save_path.name,
            }
            

            # keep your behavior: FT on all but last batch (base scheme) OR always in single-batch mode
            if batch_id_for_ft < np.max(unique_batches) or last_batch is None:
                model_fine_tuner.main_ft()

            prev_round_of_ft += 1
            experiment_summary.append(row)
            pd.DataFrame(experiment_summary).to_csv(model_ft_base_folder / "ft_summary.csv", index=False)

            print("-----------------------------")


def main():
    subjects = ["S01", "S02", "S03", "S04"]
    conditions = ["silent", "vocalized"]

    for sub in subjects:
        for cond in conditions:
            print("\n" + "=" * 80)
            print(f"Running Fine-Tuning | subject={sub} | condition={cond}")
            print("=" * 80)
            run_ft_for(sub, cond)


if __name__ == "__main__":
    main()
