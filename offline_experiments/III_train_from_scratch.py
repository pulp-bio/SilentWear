# # SPDX-FileCopyrightText: 2026 ETH Zurich
# # SPDX-License-Identifier: Apache-2.0

"""Train-from-scratch Evaluation Setting

This experiment performs a progressive fine-tuning (FT) procedure **without any pre-training**:
- The first batch starts from **random initialization** (model_to_ft_path=None).
- Subsequent batches continue from the previous checkpoint.
- Results are saved under:
    <ARTIFACTS_DIR>/models/train_from_scratch/<subject>/<condition>/<model_name>/<MODEL_NAME_ID>/bs_config_<N>/

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
from offline_experiments.Model_Fine_Tuner import Model_Fine_Tuner
from offline_experiments.general_utils import check_data_directories
from utils.general_utils import load_subjects_data, open_file


def build_bs_directory(model_base_folder: Path) -> Path:
    """Create a new subfolder bs_config_<N> under model_base_folder."""
    model_base_folder = Path(model_base_folder)
    model_base_folder.mkdir(parents=True, exist_ok=True)

    for bs_id in range(10_000_000):
        bs_folder = model_base_folder / f"bs_config_{bs_id}"
        try:
            bs_folder.mkdir(parents=True, exist_ok=False)  # atomic
            return bs_folder
        except FileExistsError:
            continue

    raise RuntimeError("Could not find a free bs_config_<N> directory name.")


def build_tfs_model_cfg(
    model_config: dict, tfs_cfg: dict, save_path: Optional[Path] = None
) -> dict:
    """Build the incremental training config starting from the base model_cfg."""
    model_config = copy.deepcopy(model_config)
    train_cfg = model_config["model"]["kwargs"]["train_cfg"]

    optimizer_old = train_cfg["optimizer_cfg"]
    lr_new = float(tfs_cfg["fs_lr"])
    optimizer_cfg_new = {"lr": lr_new, "name": optimizer_old["name"]}

    scheduler = train_cfg.get("scheduler", None)
    weight_decay = train_cfg.get("weight_decay", 0)
    es_patience = train_cfg.get("early_stop_patience", 10)

    model_config["model"]["kwargs"]["train_cfg"] = {
        "lr": lr_new,
        "num_epochs": int(tfs_cfg["num_fs_epochs"]),
        "optimizer_cfg": optimizer_cfg_new,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "early_stop_patience": es_patience,
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(model_config, f, indent=4)

    return model_config


def tfs_cfg_first_run(cfg: dict, lr_new: float = 1e-3, num_epochs_new: int = 50) -> dict:
    """Adjust config for the first batch training run."""
    cfg = copy.deepcopy(cfg)
    train_cfg = cfg["model"]["kwargs"]["train_cfg"]

    optimizer_old = train_cfg["optimizer_cfg"]
    lr_base = float(train_cfg["lr"])
    epochs_base = int(train_cfg.get("num_epochs_first_train", 100))

    lr_final = max(lr_base, float(lr_new))
    epochs_final = max(epochs_base, int(num_epochs_new))
    es_patience = train_cfg.get("early_stop_patience", 10)

    optimizer_cfg_new = {"lr": lr_final, "name": optimizer_old["name"]}
    scheduler = train_cfg.get("scheduler", None)
    weight_decay = train_cfg.get("weight_decay", 0)

    cfg["model"]["kwargs"]["train_cfg"] = {
        "lr": lr_final,
        "num_epochs": epochs_final,
        "optimizer_cfg": optimizer_cfg_new,
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "early_stop_patience": es_patience,
    }
    return cfg


def return_batches_for_training(tfs_cfg: dict, df: pd.DataFrame) -> Tuple[List[int], Optional[int]]:
    """Return list of batch IDs to process, plus last_batch (only meaningful in 'base' scheme)."""
    scheme = tfs_cfg.get("batch_ft_scheme", "base")

    if scheme == "single_batch" and tfs_cfg.get("single_batch_id", None) is None:
        raise ValueError("single_batch_id must be set when batch_ft_scheme='single_batch'")

    if scheme == "base":
        unique_batches = np.sort(df["batch_id"].unique()).tolist()
        last_batch = int(np.max(unique_batches))
        return unique_batches, last_batch

    if scheme == "single_batch":
        return [int(tfs_cfg["single_batch_id"])], None

    raise ValueError(f"Unknown batch_ft_scheme: {scheme}")


def run_train_from_scratch_for(
    sub: str,
    cond: str,
    base_cfg: dict,
    model_cfg: dict,
    tfs_cfg: dict,
) -> Path:
    """Run the train-from-scratch sweep for a single subject and condition."""
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["data"]["subject_id"] = sub
    base_cfg["condition"] = cond

    model_name = model_cfg["model"]["name"]
    model_id = tfs_cfg.get("model_name_id", "w1400ms")

    artifacts_root = Path(base_cfg["data"]["models_main_directory"])
    model_base_folder = (
        artifacts_root / "models" / "train_from_scratch" / sub / cond / model_name / str(model_id)
    )
    model_base_folder.mkdir(parents=True, exist_ok=True)

    model_bs_folder = build_bs_directory(model_base_folder)

    run_cfg_min = {
        "base_cfg": base_cfg,
        "model_cfg": model_cfg,
        "note": "Train-from-scratch: first batch random init; subsequent batches continue from previous checkpoint.",
    }
    with open(model_bs_folder / "run_cfg_min.json", "w") as f:
        json.dump(run_cfg_min, f, indent=4)

    tfs_model_cfg = build_tfs_model_cfg(
        model_cfg, tfs_cfg, save_path=model_bs_folder / "tfs_cfg.json"
    )

    win_size_ms = int(base_cfg["window"]["window_size_s"] * 1000)
    data_directories = check_data_directories(
        main_data_directory=base_cfg["data"]["data_directory"],
        all_subjects_models=False,
        sub_id=sub,
        condition=cond,
        window_size_ms=win_size_ms,
        base_config=base_cfg,
    )

    df = load_subjects_data(data_directories=data_directories, print_statistics=False)

    experiment_summary: List[Dict[str, Any]] = []
    sessions = np.sort(df["session_id"].unique())

    for test_session in sessions:
        df_test_session = df[df["session_id"] == test_session]
        if len(df_test_session) == 0:
            continue

        unique_batches, last_batch = return_batches_for_training(tfs_cfg, df_test_session)
        fold_id = int(test_session)

        prev_round = 0
        for batch_id in unique_batches:
            if batch_id == unique_batches[0]:
                model_to_ft = None
                model_to_name = "RANDOM_INIT"

                lr_first = tfs_cfg.get(
                    "lr_first_train", tfs_model_cfg["model"]["kwargs"]["train_cfg"]["lr"]
                )
                epochs_first = tfs_cfg.get(
                    "num_epochs_first_train",
                    tfs_model_cfg["model"]["kwargs"]["train_cfg"]["num_epochs"],
                )
                batch_model_cfg = tfs_cfg_first_run(
                    tfs_model_cfg, lr_new=float(lr_first), num_epochs_new=int(epochs_first)
                )
            else:
                model_to_ft = model_bs_folder / f"session_{fold_id}_bs_{batch_id-1}.pt"
                model_to_name = model_to_ft.name
                batch_model_cfg = tfs_model_cfg

            new_model_save_path = model_bs_folder / f"session_{fold_id}_bs_{batch_id}.pt"
            df_batch = df_test_session[df_test_session["batch_id"] == batch_id]

            if base_cfg["experiment"].get("include_rest", False):
                min_samples = df_batch["Label_int"].value_counts().min()
                idx_rest = df_batch[df_batch["Label_str"] == "rest"].index.values
                index_rest_ds = (
                    df_batch[df_batch["Label_str"] == "rest"]
                    .sample(n=min_samples, random_state=base_cfg["experiment"]["seed"])
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
                base_cfg=base_cfg,
                model_cfg=batch_model_cfg,
                model_to_ft_path=model_to_ft,
                new_model_save_path=new_model_save_path,
                ft_cfg_settings=tfs_cfg,
                df_for_ft_train=df_train,
                df_for_ft_val=df_val,
            )

            metrics_before = model_fine_tuner.test_zero_shot_acc()

            row = {
                "subject": sub,
                "condition": cond,
                "test_session": int(test_session),
                "model_to_fine_tune_name": model_to_name,
                "zero_shot_test_batch": int(batch_id),
                "num_prev_ft_rounds": int(prev_round),
                "zero_shot_balanced_acc": float(metrics_before["balanced_accuracy"]),
                "zero_shot_cm": json.dumps(np.asarray(metrics_before["confusion_matrix"]).tolist()),
                "new_model_name": new_model_save_path.name,
            }

            if batch_id < max(unique_batches) or last_batch is None:
                model_fine_tuner.main_ft()

            prev_round += 1
            experiment_summary.append(row)
            pd.DataFrame(experiment_summary).to_csv(
                model_bs_folder / "train_from_scratch_summary.csv", index=False
            )

    return model_bs_folder


class TrainFromScratch_Model_Trainer:
    """Importable trainer compatible with scripts/30_run_experiments.py."""

    def __init__(self, base_config: dict, model_config: dict, tfs_cfg: dict):
        self.base_cfg = copy.deepcopy(base_config)
        self.model_cfg = copy.deepcopy(model_config)
        self.tfs_cfg = copy.deepcopy(tfs_cfg)

        if "subject_id" not in self.base_cfg.get("data", {}):
            raise ValueError("base_config['data']['subject_id'] is required")
        if "condition" not in self.base_cfg:
            raise ValueError("base_config['condition'] is required")

    def main(self) -> Path:
        sub = self.base_cfg["data"]["subject_id"]
        cond = self.base_cfg["condition"]
        return run_train_from_scratch_for(
            sub=sub,
            cond=cond,
            base_cfg=self.base_cfg,
            model_cfg=self.model_cfg,
            tfs_cfg=self.tfs_cfg,
        )


def main():
    """Standalone entrypoint."""
    config_root = REPO_ROOT / "config"

    tfs_cfg = open_file(config_root / "paper_train_from_scratch_config.yaml")
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
            print(f"Running TRAIN-FROM-SCRATCH | subject={sub} | condition={cond}")
            print("=" * 80)

            trainer = TrainFromScratch_Model_Trainer(cfg_run, model_cfg, tfs_cfg)
            out_dir = trainer.main()
            print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
