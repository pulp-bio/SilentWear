"""
Global Experiment Evaluation Setting

Behavior:
- Load all windows/features for the given subject + condition.
- Run CV as configured in base_config['cv'] (by defualt, leave_one_batch_out).
- Save outputs under:
    <ARTIFACTS_DIR>/models/global/<subject>/<condition>/<model_name>/<MODEL_NAME_ID>/model_<k>/

Compatibility goals:
1) Importable by `scripts/30_run_experiments.py` (runs ONE subject/condition per call).
2) Runnable as a standalone script (loops subjects/conditions by default).

"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure repo root on path (repo_root/offline_experiments/this_file.py -> repo_root)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.Model_Master import Model_Master
from models.seeds import TORCH_MANUAL_SEED, RANDOM_SEED, RGN_SEED
from utils.general_utils import load_all_h5files_from_folder, print_dataset_summary_statistics
from offline_experiments.general_utils import *  


class Global_Model_Trainer:
    def __init__(self, base_config: dict, model_config: dict) -> None:
        self.base_config = deepcopy(base_config)
        self.model_config = deepcopy(model_config)
        self.model_master: Optional[Model_Master] = None

        self.sub_id = self.base_config["data"]["subject_id"]
        if isinstance(self.sub_id, str):
            self.all_subjects_models = False
        elif isinstance(self.sub_id, list):
            self.all_subjects_models = True
        else:
            raise ValueError("base_config['data']['subject_id'] must be a string or list")

        self.condition = self.base_config["condition"]
        self.main_dire = Path(self.base_config["data"]["data_directory"])
        self.main_model_dire = Path(self.base_config["data"]["models_main_directory"])
        self.model_name = self.model_config["model"]["name"]
        self.window_size_ms = int(float(self.base_config["window"]["window_size_s"]) * 1000)
        self.include_rest = bool(self.base_config["experiment"]["include_rest"])

        # model_name_id is the stable grouping key used by analysis (e.g., w1400ms).
        self.model_name_id = self.base_config.get("model_name_id", f"w{self.window_size_ms}ms")

        self.model_dire = self._create_saving_directory()
        self.data_dire_proc: List[Path] = []
        self._check_data_directory()

        self.df = pd.DataFrame()
        self.cv_summaries: List[Dict[str, Any]] = []

    def _create_saving_directory(self) -> Path:
        # models/global/<SUB_ID>/<condition>/<model_name>/<MODEL_NAME_ID>/model_<k>/
        if not self.all_subjects_models:
            model_parent_dire = (
                self.main_model_dire
                / "models"
                / "global"
                / str(self.sub_id)
                / str(self.condition)
                / str(self.model_name)
                / str(self.model_name_id)
            )
        else:
            model_parent_dire = (
                self.main_model_dire
                / "models"
                / "global"
                / "all_subjects"
                / str(self.condition)
                / str(self.model_name)
                / str(self.model_name_id)
            )

        model_parent_dire.mkdir(parents=True, exist_ok=True)

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
        # wins_and_features/<sub>/<condition>/WIN_<ms>
        win_feats_root = self.base_config["paths"]["win_and_feats"]

        if not self.all_subjects_models:
            if self.condition != "voc_and_silent":
                self.data_dire_proc.append(
                    self.main_dire / win_feats_root / str(self.sub_id) / str(self.condition) / f"WIN_{self.window_size_ms}"
                )
            else:
                self.data_dire_proc.append(
                    self.main_dire / win_feats_root / str(self.sub_id) / "silent" / f"WIN_{self.window_size_ms}"
                )
                self.data_dire_proc.append(
                    self.main_dire / win_feats_root / str(self.sub_id) / "vocalized" / f"WIN_{self.window_size_ms}"
                )
        else:
            for curr_sub_id in self.sub_id:
                if self.condition != "voc_and_silent":
                    self.data_dire_proc.append(
                        self.main_dire / win_feats_root / str(curr_sub_id) / str(self.condition) / f"WIN_{self.window_size_ms}"
                    )
                else:
                    self.data_dire_proc.append(
                        self.main_dire / win_feats_root / str(curr_sub_id) / "silent" / f"WIN_{self.window_size_ms}"
                    )
                    self.data_dire_proc.append(
                        self.main_dire / win_feats_root / str(curr_sub_id) / "vocalized" / f"WIN_{self.window_size_ms}"
                    )

        for d in self.data_dire_proc:
            if not d.exists():
                raise FileNotFoundError(
                    f"Windows/features directory does not exist: {d}. " 
                    f"Did you run scripts/20_make_windows_and_features.py for window={self.window_size_ms}ms?"
                )

    def _save_run_cfg(self) -> None:
        run_cfg_dict = {
            "condition": self.condition,
            "experiment_type": "global",
            "experimental_settings": {
                "window_size_ms": self.window_size_ms,
                "include_rest": self.include_rest,
                "cv_type": self.base_config.get("cv", {}),
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

    def run_cv(self) -> List[Dict[str, Any]]:
        cv_cfg = self.base_config.get("cv", {})
        mode = cv_cfg.get("mode")
        val_size = float(cv_cfg.get("val_size", 0.3))
        seed = int(self.base_config["experiment"]["seed"])
        print("========================================\n\n")
        print("SEED set to", seed)
        df = self.df
        if not self.include_rest:
            df = df[df["Label_str"] != "rest"].copy()

        print_dataset_summary_statistics(df)

        if mode == "leave_one_batch_out":
            return self._cv_leave_one_batch_out(df, val_size=val_size, seed=seed)

        raise ValueError(f"Unknown cv.mode='{mode}' (expected 'leave_one_batch_out')")

    def _cv_leave_one_batch_out(self, df: pd.DataFrame, val_size: float = 0.3, seed: int = 0) -> List[Dict[str, Any]]:
        print("lobo")
        self.cv_summaries = []
        batches = df["batch_id"].unique()

        for fold_id, test_batch_id in enumerate(batches):
            print(f"\n\n=== LOBO FOLD {fold_id+1}/{len(batches)} | test_batch={test_batch_id} ===")

            train_val_data = df[df["batch_id"] != test_batch_id]
            test_data = df[df["batch_id"] == test_batch_id]

            if self.include_rest:
                min_samples = train_val_data["Label_int"].value_counts().min()
                idx_rest = train_val_data[train_val_data["Label_str"] == "rest"].index.values
                index_rest_ds = train_val_data[train_val_data["Label_str"] == "rest"].sample(n=min_samples, random_state=seed).index.values
                idx_to_drop = np.setdiff1d(idx_rest, index_rest_ds)
                train_val_data = train_val_data.drop(index=idx_to_drop)

            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_size,
                shuffle=True,
                random_state=seed,
                stratify=train_val_data["Label_int"],
            )

            row_summary = self._run_one_fold(
                fold_id=fold_id,
                train_df=train_data,
                val_df=val_data,
                test_df=test_data,
                mode="leave_one_batch_out",
                test_batch_id=int(test_batch_id),
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
        test_batch_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.model_master = Model_Master(self.base_config, self.model_config)
        self.model_master.df_train = train_df
        self.model_master.df_val = val_df
        self.model_master.df_test = test_df

        if getattr(self.model_master, "kind", None) == "ml":
            if self.model_config.get("model", {}).get("features", {}).get("scale_feats", False):
                feat_cols = self.model_master.extract_dataset_train_columns()
                scaler = StandardScaler()

                train_df.loc[:, feat_cols] = scaler.fit_transform(train_df[feat_cols])
                val_df.loc[:, feat_cols] = scaler.transform(val_df[feat_cols])
                test_df.loc[:, feat_cols] = scaler.transform(test_df[feat_cols])

        self.model_master.generate_training_labels()
        self.model_master.remap_all_datasets()
        self.model_master.register_model()

        save_model_path = self.model_dire / f"{mode}_fold_{fold_id+1}"
        model, metrics, y_true, y_pred = self.model_master.train_model(test=True, save_model_path=save_model_path)

        row_summary: Dict[str, Any] = {
            "cv_mode": mode,
            "fold_num": int(fold_id + 1),
            "test_batch": int(test_batch_id) if test_batch_id is not None else None,
        }

        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                row_summary[k] = json.dumps(np.asarray(v).tolist())
            elif isinstance(v, (np.floating,)):
                row_summary[k] = float(v)
            else:
                row_summary[k] = v

        row_summary["train_idx"] = self.model_master.df_train.index.tolist()
        row_summary["val_idx"] = self.model_master.df_val.index.tolist()
        row_summary["test_idx"] = self.model_master.df_test.index.tolist()
        row_summary["y_true"] = y_true.tolist()
        row_summary["y_pred"] = y_pred.tolist()

        return row_summary

    def main(self) -> Path:
        # Load data for the current subject/condition
        df = pd.DataFrame()
        for curr_data_dire in self.data_dire_proc:
            df_curr = load_all_h5files_from_folder(curr_data_dire, key="wins_feats", print_statistics=False)
            df = pd.concat((df, df_curr), ignore_index=True)

        self.df = df.reset_index(drop=True)

        self._save_run_cfg()
        self.run_cv()

        summary_df = pd.DataFrame(self.cv_summaries)
        summary_df.to_csv(self.model_dire / "cv_summary.csv", index=False)

        return self.model_dire


def main():
    """Standalone entrypoint."""
    config_root = REPO_ROOT / "config"
    base_config_path = config_root / "paper_models_config.yaml"
    model_config_path = config_root / "models_configs" / "random_forest_config.yaml"                # or speechnet_config.yaml

    base_cfg = yaml.safe_load(base_config_path.read_text())
    model_cfg = yaml.safe_load(model_config_path.read_text())

    subjects = ["S01", "S02", "S03", "S04"]
    conditions = ["silent", "vocalized"]

    for sub in subjects:
        for cond in conditions:
            print("\n" + "=" * 80)
            print(f"Running Global Model | subject={sub} | condition={cond}")
            print("=" * 80)

            cfg_run = deepcopy(base_cfg)
            cfg_run["data"]["subject_id"] = sub
            cfg_run["condition"] = cond

            trainer = Global_Model_Trainer(base_config=cfg_run, model_config=model_cfg)
            out_dir = trainer.main()
            print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
