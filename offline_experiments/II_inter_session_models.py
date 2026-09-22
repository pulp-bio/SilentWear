# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Inter-session models (subject-specific or pooled).

Behavior:
- Load all windows/features for the given subject(s) + condition (and window size).
- Run LOSO CV across acquisition sessions:
    * train on all sessions except one
    * validate via random stratified split from train sessions
    * test on the held-out session
- Save outputs under:
    <ARTIFACTS_DIR>/models/<experiment_subdir>/<subject>/<condition>/<model_name>/<MODEL_NAME_ID>/model_<k>/
  or when pooled:
    <ARTIFACTS_DIR>/models/<experiment_subdir>/all_subjects/<condition>/<model_name>/<MODEL_NAME_ID>/model_<k>/

Compatibility goals:
1) Importable by `scripts/30_run_experiments.py`.
2) Runnable as a standalone script with CLI arguments and `--pool_subjects` support.
"""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure repo root on path (repo_root/offline_experiments/this_file.py -> repo_root)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.Model_Master import Model_Master
from models.seeds import RGN_SEED, TORCH_MANUAL_SEED, RANDOM_SEED
from utils.general_utils import load_all_h5files_from_folder, print_dataset_summary_statistics
from offline_experiments.general_utils import (
    base_window_rows,
    training_rows_with_augmentation,
    reset_all_seeds,
    check_data_directories,
    apply_datasets_normalization,
)
from offline_experiments.explainability import maybe_run_explainability


class Inter_Session_Model_Trainer:
    def __init__(
        self, base_config: dict, model_config: dict, experiment_subdir: str = "inter_session"
    ) -> None:
        self.base_config = deepcopy(base_config)
        self.model_config = deepcopy(model_config)
        self.model_master: Optional[Model_Master] = None
        self.experiment_subdir = str(experiment_subdir)

        # Subject(s)
        self.sub_id = self.base_config["data"]["subject_id"]
        if isinstance(self.sub_id, str):
            self.all_subjects_models = False
        elif isinstance(self.sub_id, list):
            self.all_subjects_models = True
        else:
            raise ValueError("base_config['data']['subject_id'] must be a string or list")

        self.condition = self.base_config[
            "condition"
        ]  # silent | vocalized | voc_and_silent (legacy)
        self.main_dire = Path(self.base_config["data"]["data_directory"])
        self.main_model_dire = Path(self.base_config["data"]["models_main_directory"])
        self.model_name = self.model_config["model"]["name"]
        self.window_size_ms = int(float(self.base_config["window"]["window_size_s"]) * 1000)
        self.include_rest = bool(self.base_config["experiment"]["include_rest"])
        self.data_normalization = bool(
            self.base_config["experiment"].get("data_normalization", False)
        )

        # Stable key for analysis grouping (e.g., w1400ms)
        self.model_name_id = self.base_config.get("model_name_id", f"w{self.window_size_ms}ms")

        self.model_dire = self._create_saving_directory()

        # Data directories
        self.data_dire_proc: List[Path] = []
        self._check_data_directory()

        self.df = pd.DataFrame()
        self.cv_summaries: List[Dict[str, Any]] = []

    def _create_saving_directory(self) -> Path:
        # models/<experiment_subdir>/<SUB_ID>/<condition>/<model_name>/<MODEL_NAME_ID>/model_<k>/
        if not self.all_subjects_models:
            model_parent_dire = (
                self.main_model_dire
                / "models"
                / self.experiment_subdir
                / str(self.sub_id)
                / str(self.condition)
                / str(self.model_name)
                / str(self.model_name_id)
            )
        else:
            model_parent_dire = (
                self.main_model_dire
                / "models"
                / self.experiment_subdir
                / "all_subjects"
                / str(self.condition)
                / str(self.model_name)
                / str(self.model_name_id)
            )

        model_parent_dire.mkdir(parents=True, exist_ok=True)

        # Optional run_tag for controlled paths (useful for sweeps)
        run_tag = self.base_config.get("experiment", {}).get("run_tag", None)
        if run_tag:
            model_dire = model_parent_dire / str(run_tag)
            model_dire.mkdir(parents=True, exist_ok=True)
            print("Models will be saved under (run_tag):", model_dire)
            return model_dire

        model_id_base = 1
        while True:
            model_dire = model_parent_dire / f"model_{model_id_base}"
            if model_dire.exists():
                model_id_base += 1
                continue
            model_dire.mkdir()
            print("Models will be saved under:", model_dire)
            return model_dire

    def _check_data_directory(self) -> None:
        self.data_dire_proc = check_data_directories(
            main_data_directory=self.main_dire,
            all_subjects_models=self.all_subjects_models,
            sub_id=self.sub_id,
            condition=self.condition,
            window_size_ms=self.window_size_ms,
            base_config=self.base_config,
        )

    def _save_run_cfg(self) -> None:
        train_cfg = self.model_config.get("model", {}).get("kwargs", {}).get("train_cfg", {})
        loss_name = str(train_cfg.get("loss_name", "unknown_loss"))
        loss_cfg = train_cfg.get("loss", None)
        label_mode = self.base_config.get("experiment", {}).get("label_mode", "word")

        run_cfg_dict = {
            "condition": self.condition,
            "experiment_type": self.experiment_subdir,
            "experimental_settings": {
                "window_size_ms": self.window_size_ms,
                "include_rest": self.include_rest,
                "label_mode": label_mode,
                "data_normalization": self.data_normalization,
                "cv_type": self.base_config.get("cv", {}),
                "loss_name": loss_name,
                "loss_cfg": loss_cfg,
            },
            "model_cfg": self.model_config,
            "base_cfg": self.base_config,
            "seeds": {
                "torch_manual_seed": TORCH_MANUAL_SEED,
                "random_seed": RANDOM_SEED,
                "rgn_seed": RGN_SEED,
            },
        }

        with open(self.model_dire / "run_cfg.json", "w") as f:
            json.dump(run_cfg_dict, f, indent=4, sort_keys=True)

    def run_inter_session_cv(self, val_size: float = 0.3, seed: int = 0) -> List[Dict[str, Any]]:
        self.cv_summaries = []
        df_base = base_window_rows(self.df)
        sessions = np.sort(df_base["session_id"].unique())

        for fold_id, test_session_id in enumerate(sessions):
            print(
                f"\n\n=== LOSO FOLD {fold_id+1}/{len(sessions)} | test_session={test_session_id} ==="
            )

            train_val_base = df_base[df_base["session_id"] != test_session_id]
            test_data = df_base[df_base["session_id"] == test_session_id]

            if self.include_rest:
                min_samples = train_val_base["Label_int"].value_counts().min()
                idx_rest = train_val_base[train_val_base["Label_str"] == "rest"].index.values
                index_rest_ds = (
                    train_val_base[train_val_base["Label_str"] == "rest"]
                    .sample(n=min_samples, random_state=seed)
                    .index.values
                )
                idx_to_drop = np.setdiff1d(idx_rest, index_rest_ds)
                train_val_base = train_val_base.drop(index=idx_to_drop)

            train_base, val_data = train_test_split(
                train_val_base,
                test_size=val_size,
                shuffle=True,
                random_state=seed,
                stratify=train_val_base["Label_int"],
            )

            train_data = training_rows_with_augmentation(
                self.df,
                train_base,
                mode=self.base_config.get("experiment", {}).get("augmentation_train_mode", "augmented_size"),
                seed=int(self.base_config.get("experiment", {}).get("seed", 0)),
            )

            row_summary = self._run_one_fold(
                fold_id=fold_id,
                train_df=train_data,
                val_df=val_data,
                test_df=test_data,
                mode="leave_one_session_out",
                test_session_id=int(test_session_id),
            )
            self.cv_summaries.append(row_summary)

        return self.cv_summaries

    def _run_one_fold(
        self,
        fold_id: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mode: str,
        test_session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        
        reset_all_seeds()

        self.model_master = Model_Master(self.base_config, self.model_config)
        self.model_master.df_train = train_df
        self.model_master.df_val = val_df
        self.model_master.df_test = test_df

        save_model_path = self.model_dire / f"{mode}_fold_{fold_id+1}"

        apply_datasets_normalization(self.model_master, self.base_config, save_model_path)

        # Feature scaling for ML models
        if getattr(self.model_master, "kind", None) == "ml":
            if self.model_config.get("model", {}).get("features", {}).get("scale_feats", False):
                print("Scaling feats")
                feat_cols = self.model_master.extract_dataset_train_columns()
                scaler = StandardScaler()

                train_df, val_df, test_df = (
                    self.model_master.df_train,
                    self.model_master.df_val,
                    self.model_master.df_test,
                )
                train_df.loc[:, feat_cols] = scaler.fit_transform(train_df[feat_cols])
                val_df.loc[:, feat_cols] = scaler.transform(val_df[feat_cols])
                test_df.loc[:, feat_cols] = scaler.transform(test_df[feat_cols])

        self.model_master.generate_training_labels()
        self.model_master.remap_all_datasets()
        self.model_master.register_model()

        model, metrics, y_true, y_pred = self.model_master.train_model(
            test=True, save_model_path=save_model_path
        )

        maybe_run_explainability(
            self.model_master,
            self.base_config,
            df_trainval=pd.concat([self.model_master.df_train, self.model_master.df_val]),
            df_test=self.model_master.df_test,
            out_dir=self.model_dire / "explainability" / f"fold_{fold_id+1}",
        )

        row_summary: Dict[str, Any] = {
            "cv_mode": mode,
            "fold_num": int(fold_id + 1),
            "test_session": int(test_session_id) if test_session_id is not None else None,
        }

        if metrics is not None:
            for k, v in metrics.items():
                if isinstance(v, (np.ndarray, list, tuple)):
                    row_summary[k] = json.dumps(np.asarray(v).tolist())
                elif isinstance(v, (np.floating,)):
                    row_summary[k] = float(v)
                else:
                    row_summary[k] = v

        print(f'{self.model_master.df_test["Label_str"].value_counts()} | {self.model_master.df_test.shape[0]} test samples')
        row_summary["train_idx"] = self.model_master.df_train.index.tolist()
        row_summary["val_idx"] = self.model_master.df_val.index.tolist()
        row_summary["test_idx"] = self.model_master.df_test.index.tolist()
        row_summary["y_true"] = None if y_true is None else np.asarray(y_true).tolist()
        row_summary["y_pred"] = None if y_pred is None else np.asarray(y_pred).tolist()

        return row_summary

    def main(self) -> Path:
        # Load data for the current subject/condition
        df = pd.DataFrame()
        for curr_data_dire in self.data_dire_proc:
            df_curr = load_all_h5files_from_folder(
                curr_data_dire, key="wins_feats", print_statistics=False
            )
            df = pd.concat((df, df_curr), ignore_index=True)

        self.df = df.reset_index(drop=True)

        if self.base_config.get("experiment", {}).get("print_dataset_stats", True):
            print_dataset_summary_statistics(self.df)

        self._save_run_cfg()

        cv_cfg = self.base_config.get("cv", {})
        val_size = float(cv_cfg.get("val_size", 0.3))
        seed = int(self.base_config.get("experiment", {}).get("seed", 0))

        self.run_inter_session_cv(val_size=val_size, seed=seed)

        pd.DataFrame(self.cv_summaries).to_csv(self.model_dire / "cv_summary.csv", index=False)

        return self.model_dire


def run_all_subjects(
    base_config: dict,
    model_config: dict,
    subjects: List[str],
    conditions: List[str],
    experiment_subdir: str = "inter_session",
) -> None:
    """
    Run inter-session evaluation pooling all specified subjects together into a single dataset.
    Passing a list of subjects automatically sets all_subjects_models = True in the trainer.
    """
    for cond in conditions:
        print("\n" + "=" * 80)
        print(f"Running Pooled Inter-Session Model (all_subjects) | subjects={subjects} | condition={cond}")
        print("=" * 80)

        cfg_run = deepcopy(base_config)
        cfg_run["data"]["subject_id"] = subjects  # Passing a list activates pooled training
        cfg_run["condition"] = cond

        trainer = Inter_Session_Model_Trainer(
            base_config=cfg_run, model_config=model_config, experiment_subdir=experiment_subdir
        )
        out_dir = trainer.main()
        print(f"[DONE] outputs in: {out_dir}")


def main():
    """Standalone entrypoint."""
    parser = argparse.ArgumentParser(description="Run Inter-Session Model Trainer standalone.")
    config_root = REPO_ROOT / "config"
    parser.add_argument("--base_config", type=Path, default=config_root / "paper_models_config.yaml")
    parser.add_argument(
        "--model_config", type=Path, default=config_root / "models_configs" / "random_forest_config.yaml"
    )
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    parser.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    parser.add_argument(
        "--pool_subjects",
        action="store_true",
        help="Pool all specified subjects together into a single dataset.",
    )
    args = parser.parse_args()

    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())

    if args.pool_subjects:
        run_all_subjects(base_cfg, model_cfg, args.subjects, args.conditions, experiment_subdir="inter_session")
    else:
        for sub in args.subjects:
            for cond in args.conditions:
                cfg_run = deepcopy(base_cfg)
                cfg_run["data"]["subject_id"] = sub
                cfg_run["condition"] = cond
                print("\n" + "=" * 80)
                print(f"Running Inter-Session Model | subject={sub} | condition={cond}")
                print("=" * 80)

                trainer = Inter_Session_Model_Trainer(
                    base_config=cfg_run, model_config=model_cfg, experiment_subdir="inter_session"
                )
                out_dir = trainer.main()
                print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
