# Copyright 2026 Giusy Spacone
# Copyright 2026 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Inter-session + Fine-Tuning Evaluation Setting.

This experiment fine-tunes inter-session base models progressively over batches of the
held-out test session (as defined by the inter-session CV fold):

- For each inter-session base model (fold), identify its test_session from cv_summary.csv
- For each batch in that test_session:
    * evaluate zero-shot on the batch (before fine-tuning)
    * fine-tune on the batch (except last batch in progressive scheme)
    * save checkpoint fold_<k>_ft_<batch>.pt

Folders:
  Base inter-session models (created by II_inter_session_models):
    <ARTIFACTS_DIR>/models/inter_session/<subject>/<condition>/<model_name>/<MODEL_NAME_ID>/

  Fine-tuning outputs (this script):
    <ARTIFACTS_DIR>/models/inter_session_ft/<subject>/<condition>/<model_name>/<MODEL_NAME_ID>/ft_config_<N>/

Compatibility goals:
1) Importable by `scripts/30_run_experiments.py` (runs ONE subject/condition per call).
2) Runnable as a standalone script (loops subjects/conditions by default).
"""

import sys
from pathlib import Path
import json
import copy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure repo root on path (repo_root/offline_experiments/this_file.py -> repo_root)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.seeds import *  # noqa: F401,F403
from models.TorchTrainer import evaluate_model
from models.utils import load_pretrained_model
from offline_experiments.Model_Fine_Tuner import Model_Fine_Tuner
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import check_data_directories
from utils.general_utils import load_subjects_data, open_file


def build_ft_directory(model_base_folder: Path) -> Path:
    """Create a new subfolder ft_config_<N> under model_base_folder."""
    model_base_folder = Path(model_base_folder)
    model_base_folder.mkdir(parents=True, exist_ok=True)

    for model_ft_id in range(10_000_000):
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
    """Validate existence of base inter-session models."""
    if expected_in_dir is None:
        expected_in_dir = {"base_models_prefix": "leave_one_session_out_fold", "expected_num": 3}

    model_base_folder = Path(model_base_folder)
    if not model_base_folder.exists():
        raise FileNotFoundError(f"Model base folder does not exist: {model_base_folder}")
    if not model_base_folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {model_base_folder}")

    if model_to_ft_name:
        model_path = model_base_folder / model_to_ft_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find model file: {model_path}")
        return [model_path]

    prefix = expected_in_dir.get("base_models_prefix", "leave_one_session_out_fold")
    expected_num = int(expected_in_dir.get("expected_num", 0))
    if expected_num <= 0:
        raise ValueError(f"expected_num must be > 0 (got {expected_num})")

    paths: List[Path] = []
    for model_id in range(1, expected_num + 1):
        model_path = model_base_folder / f"{prefix}_{model_id}.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Could not find expected model: {model_path}")
        paths.append(model_path)

    return paths


def build_ft_model_cfg(model_config: dict, ft_cfg: dict, save_path: Optional[Path] = None) -> dict:
    """Build a fine-tuning model config starting from base model_cfg."""
    model_config = copy.deepcopy(model_config)
    train_cfg = model_config["model"]["kwargs"]["train_cfg"]

    optimizer_old = train_cfg["optimizer_cfg"]
    lr_new = float(ft_cfg["ft_lr"])
    optimizer_cfg_new = {"lr": lr_new, "name": optimizer_old["name"]}

    scheduler = train_cfg.get("scheduler", None)
    weight_decay = train_cfg.get("weight_decay", 0)
    es_patience = train_cfg.get("early_stop_patience", 10)

    model_config["model"]["kwargs"]["train_cfg"] = {
        "lr": lr_new,
        "num_epochs": int(ft_cfg["num_ft_epochs"]),
        "optimizer_cfg": optimizer_cfg_new,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "early_stop_patience": es_patience,
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(model_config, f, indent=4)

    return model_config


def return_batches_for_ft(ft_cfg: dict, df_for_ft: pd.DataFrame) -> Tuple[List[int], Optional[int]]:
    """Return list of batch IDs to process for FT, plus last_batch."""
    scheme = ft_cfg.get("batch_ft_scheme", "base")

    if scheme == "single_batch" and ft_cfg.get("single_batch_id", None) is None:
        raise ValueError("single_batch_id must be set when batch_ft_scheme='single_batch'")

    if scheme == "base":
        unique_batches = np.sort(df_for_ft["batch_id"].unique()).tolist()
        last_batch = int(np.max(unique_batches))
        return unique_batches, last_batch

    if scheme == "single_batch":
        return [int(ft_cfg["single_batch_id"])], None

    raise ValueError(f"Unknown batch_ft_scheme: {scheme}")


def _base_intersession_folder(
    base_cfg: dict, model_cfg: dict, ft_cfg: dict, sub: str, cond: str
) -> Path:
    model_name = model_cfg["model"]["name"]
    model_id = ft_cfg.get("model_name_id", base_cfg.get("model_name_id", "w1400ms"))
    artifacts_root = Path(base_cfg["data"]["models_main_directory"])

    return artifacts_root / "models" / "inter_session_ft" / sub / cond / model_name / str(model_id)


def _ft_output_root(
    base_cfg: dict, model_cfg: dict, ft_cfg: dict, sub: str, cond: str, model_id: str
) -> Path:
    model_name = model_cfg["model"]["name"]
    model_id = ft_cfg.get("model_name_id", base_cfg.get("model_name_id", "w1400ms"))
    artifacts_root = Path(base_cfg["data"]["models_main_directory"])

    return artifacts_root / "models" / "inter_session_ft" / sub / cond / model_name / model_id


def run_ft_for(
    sub: str,
    cond: str,
    base_cfg: dict,
    model_cfg: dict,
    ft_cfg: dict,
) -> Path:
    """Run FT for one subject/condition. Returns created ft_config_<N> folder."""
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["data"]["subject_id"] = sub
    base_cfg["condition"] = cond

    # Base models are expected in inter_session_ft/
    model_base_folder = _base_intersession_folder(base_cfg, model_cfg, ft_cfg, sub, cond)
    model_base_folder.mkdir(parents=True, exist_ok=True)

    retrain_from_scratch = bool(ft_cfg.get("retrain_intersessions", False))
    if retrain_from_scratch or not (model_base_folder / "cv_summary.csv").exists():
        print(f"[{sub} | {cond}] (Re)training inter-session base models...")
        trainer = Inter_Session_Model_Trainer(
            base_config=base_cfg,
            model_config=model_cfg,
            experiment_subdir="inter_session_ft",
        )
        trainer.main()
        model_base_folder = Path(trainer.model_dire)

    # Load run cfg + summary from base folder
    run_cfg = open_file(model_base_folder / "run_cfg.json")
    csv_summary = open_file(model_base_folder / "cv_summary.csv")

    model_to_ft_name = ft_cfg.get("model_ft_name", None)
    ft_models_in_folder = check_base_models_exist(
        model_base_folder=model_base_folder,
        model_to_ft_name=model_to_ft_name,
        expected_in_dir={"base_models_prefix": "leave_one_session_out_fold", "expected_num": 3},
    )

    # FT output root (separate from base inter-session folder)
    ft_root = _ft_output_root(
        base_cfg, model_cfg, ft_cfg, sub, cond, model_id=str(model_base_folder.name)
    )
    ft_root.mkdir(parents=True, exist_ok=True)
    model_ft_base_folder = build_ft_directory(ft_root)

    # Build fine-tune model config based on the base run config to match paper
    base_cfg_used = run_cfg["base_cfg"]
    model_cfg_used = run_cfg["model_cfg"]
    new_ft_model_cfg = build_ft_model_cfg(
        model_cfg_used, ft_cfg, save_path=model_ft_base_folder / "ft_cfg.json"
    )

    # Load data
    win_size_ms = int(base_cfg_used["window"]["window_size_s"] * 1000)
    data_directories = check_data_directories(
        main_data_directory=base_cfg_used["data"]["data_directory"],
        all_subjects_models=False,
        sub_id=base_cfg_used["data"]["subject_id"],
        condition=base_cfg_used["condition"],
        window_size_ms=win_size_ms,
        base_config=base_cfg_used,
    )

    df = load_subjects_data(data_directories=data_directories, print_statistics=False)

    experiment_summary: List[Dict[str, Any]] = []

    for base_model_to_ft in ft_models_in_folder:
        model_intersess = load_pretrained_model(
            base_cfg=base_cfg_used,
            model_cfg=model_cfg_used,
            pretrained_model_path=base_model_to_ft,
        )

        fold_id = base_model_to_ft.name.split("fold_")[-1].split(".pt")[0]
        row_interest = csv_summary[csv_summary["fold_num"] == int(fold_id)]
        test_session = int(row_interest["test_session"].values[0])

        df_test_session = df[df["session_id"] == test_session]
        unique_batches, last_batch = return_batches_for_ft(ft_cfg, df_test_session)

        prev_round = 0
        for batch_id in unique_batches:
            model_to_ft = (
                base_model_to_ft
                if batch_id == unique_batches[0]
                else (model_ft_base_folder / f"fold_{fold_id}_ft_{batch_id-1}.pt")
            )
            new_model_save_path = model_ft_base_folder / f"fold_{fold_id}_ft_{batch_id}.pt"

            df_batch = df_test_session[df_test_session["batch_id"] == batch_id]

            if base_cfg_used["experiment"].get("include_rest", False):
                min_samples = df_batch["Label_int"].value_counts().min()
                idx_rest = df_batch[df_batch["Label_str"] == "rest"].index.values
                index_rest_ds = (
                    df_batch[df_batch["Label_str"] == "rest"]
                    .sample(n=min_samples, random_state=base_cfg_used["experiment"]["seed"])
                    .index.values
                )
                idx_to_drop = np.setdiff1d(idx_rest, index_rest_ds)
                df_batch = df_batch.drop(index=idx_to_drop)

            df_train, df_val = train_test_split(
                df_batch,
                test_size=0.3,
                shuffle=True,
                random_state=42,
                stratify=df_batch["Label_int"],
            )

            model_fine_tuner = Model_Fine_Tuner(
                base_cfg=base_cfg_used,
                model_cfg=new_ft_model_cfg,
                model_to_ft_path=model_to_ft,
                new_model_save_path=new_model_save_path,
                ft_cfg_settings=ft_cfg,
                df_for_ft_train=df_train,
                df_for_ft_val=df_val,
            )

            metrics_before = model_fine_tuner.test_zero_shot_acc()
            batch_loader = model_fine_tuner.model_master.trainer_manager.test_loader
            metrics_without_ft, _, _ = evaluate_model(model_intersess, batch_loader)

            row = {
                "subject": sub,
                "condition": cond,
                "base_model": base_model_to_ft.name,
                "test_session": test_session,
                "model_to_fine_tune_name": (
                    Path(model_to_ft).name if model_to_ft is not None else "NONE"
                ),
                "zero_shot_test_batch": int(batch_id),
                "num_prev_ft_rounds": int(prev_round),
                "zero_shot_balanced_acc": float(metrics_before["balanced_accuracy"]),
                "zero_shot_cm": json.dumps(np.asarray(metrics_before["confusion_matrix"]).tolist()),
                "balanced_acc_no_ft": float(metrics_without_ft["balanced_accuracy"]),
                "cm_no_ft": json.dumps(np.asarray(metrics_without_ft["confusion_matrix"]).tolist()),
                "new_model_name": new_model_save_path.name,
            }

            if batch_id < max(unique_batches) or last_batch is None:
                model_fine_tuner.main_ft()

            prev_round += 1
            experiment_summary.append(row)
            pd.DataFrame(experiment_summary).to_csv(
                model_ft_base_folder / "ft_summary.csv", index=False
            )

    return model_ft_base_folder


class FineTuning_Model_Trainer:
    """Importable trainer compatible with scripts/30_run_experiments.py."""

    def __init__(self, base_config: dict, model_config: dict, ft_cfg: dict):
        self.base_cfg = copy.deepcopy(base_config)
        self.model_cfg = copy.deepcopy(model_config)
        self.ft_cfg = copy.deepcopy(ft_cfg)

        if "subject_id" not in self.base_cfg.get("data", {}):
            raise ValueError("base_config['data']['subject_id'] is required")
        if "condition" not in self.base_cfg:
            raise ValueError("base_config['condition'] is required")

    def main(self) -> Path:
        sub = self.base_cfg["data"]["subject_id"]
        cond = self.base_cfg["condition"]
        return run_ft_for(sub, cond, self.base_cfg, self.model_cfg, self.ft_cfg)


def main():
    """Standalone entrypoint."""
    config_root = REPO_ROOT / "config"

    ft_cfg = open_file(config_root / "paper_ft_config.yaml")
    base_cfg = open_file(config_root / "paper_models_config.yaml")
    model_cfg = open_file(config_root / "models_configs" / "speechnet.yaml")

    subjects = ["S01", "S02", "S03", "S04"]
    conditions = ["silent", "vocalized"]

    for sub in subjects:
        for cond in conditions:
            cfg_run = copy.deepcopy(base_cfg)
            cfg_run["data"]["subject_id"] = sub
            cfg_run["condition"] = cond

            print("\n" + "=" * 80)
            print(f"Running INTER-SESSION+FT | subject={sub} | condition={cond}")
            print("=" * 80)

            trainer = FineTuning_Model_Trainer(cfg_run, model_cfg, ft_cfg)
            out_dir = trainer.main()
            print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
