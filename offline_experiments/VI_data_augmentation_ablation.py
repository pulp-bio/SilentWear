#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Data-augmentation ablation for offline experiments.

This script sweeps combinations of `data_augmentation` parameters (stride_ms, num_strides),
along with the number of sessions, regenerates windows/features for each combination, 
and runs either `global` or `inter_session`.
Saves results to cv_summary.csv for later plotting.
"""

from __future__ import annotations

import argparse
import tempfile
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Optional, Any, Tuple, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.I_global_models import Global_Model_Trainer
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import reset_all_seeds
from utils.general_utils import SubjectConfig, load_all_h5files_from_folder
from utils.II_feature_extraction.win_feature_extraction_main import Global_Windower_and_Feature_Extractor
from offline_experiments.general_utils import discover_sessions
from utils.I_data_preparation.experimental_config import (
    RAW_DIRNAME,
    RAW_AND_FILTERED_DIRNAME,
    WINS_AND_FEATURES_DIRNAME,
)


class Data_Augmentation_Ablation_Trainer:
    def __init__(self, base_config: dict, model_config: dict) -> None:
        self.base_cfg = deepcopy(base_config)
        self.model_cfg = deepcopy(model_config)
        
        self.window_cfg_template = self.base_cfg.get("experiment", {}).get("window_config_template", {})
        
        self.sub_id = self.base_cfg["data"]["subject_id"]
        self.condition = self.base_cfg["condition"]
        
        self.data_dir = Path(self.base_cfg["data"]["data_directory"]).resolve()
        self.main_model_dire = Path(self.base_cfg["data"]["models_main_directory"]).resolve()
        
        ablation_cfg = self.base_cfg.setdefault("experiment", {}).get("augmentation_ablation", {})
        self.n_sessions = ablation_cfg.get("num_sessions", 1)
        self.selected_sessions = ablation_cfg.get("session_ids", [])
        self.current_exp = ablation_cfg.get("experiment_type", "global")
        self.data_augmentation = ablation_cfg.get("data_augmentation", {"mode": "disabled"})
        self.run_label = ablation_cfg.get("run_label", "baseline")
        
        self.ablation_folder_name = ablation_cfg.get("ablation_folder_name", "data_augmentation_ablation")
        self.train_mode = self.base_cfg["experiment"].get("augmentation_train_mode", "augmented_size")
        self.window_size_s = float(self.base_cfg.get("window", {}).get("window_size_s", 1.4))

        self.working_data_root = self.main_model_dire / ".ablation_working_data"

    @staticmethod
    def _normalize_window_size_s(window_value: float) -> float:
        window_value = float(window_value)
        return window_value / 1000.0 if window_value > 10 else window_value

    def _ensure_combo_data_root(self, combo_root: Path) -> Path:
        """Ensure the ablation workspace exposes the same folders the extractor expects."""
        combo_root.mkdir(parents=True, exist_ok=True)

        # (folder_name, source, required)
        shared_dirs = [
            (RAW_AND_FILTERED_DIRNAME, self.data_dir / RAW_AND_FILTERED_DIRNAME, True),
            (RAW_DIRNAME, self.data_dir / RAW_DIRNAME, False),
        ]

        for folder_name, source, required in shared_dirs:
            if not source.exists():
                if required:
                    raise FileNotFoundError(f"Missing shared source folder: {source}")
                continue  # optional source (raw .bio), skip if absent
            target = combo_root / folder_name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source.resolve(), target_is_directory=True)
        return combo_root

    def _run_windowing(self, combo_data_root: Path) -> bool:
        normalized_window_s = self._normalize_window_size_s(self.window_size_s)

        cfg = deepcopy(self.base_cfg)
        template_cfg = deepcopy(self.base_cfg.get("experiment", {}).get("window_config_template", {}))

        if isinstance(template_cfg, dict):
            cfg.setdefault("paths", {}).update(template_cfg.get("paths", {}))
            cfg.setdefault("feature_extraction", {}).update(template_cfg.get("feature_extraction", {}))
            cfg.setdefault("data_augmentation", {}).update(template_cfg.get("data_augmentation", {}))
            cfg.setdefault("save_wins_and_feats", template_cfg.get("save_wins_and_feats", True))

        cfg.setdefault("data", {})
        cfg.setdefault("window", {})
        cfg.setdefault("feature_extraction", {"manual_feature_extraction": False, "num_subwindows": 7})
        cfg.setdefault("save_wins_and_feats", True)
        
        cfg["data"]["data_directory"] = str(combo_data_root.resolve())
        cfg["data"]["subject_id"] = str(self.sub_id)
        cfg["condition"] = str(self.condition)
        cfg["window"]["window_size_s"] = normalized_window_s
        cfg["data_augmentation"] = deepcopy(self.data_augmentation)
        
        cfg.setdefault("paths", {})
        cfg["paths"]["processed"] = RAW_AND_FILTERED_DIRNAME
        cfg["paths"]["win_and_feats"] = WINS_AND_FEATURES_DIRNAME

        with tempfile.TemporaryDirectory() as td:
            tmp_cfg = Path(td) / "create_windows_tmp.yaml"
            tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
            subject_cfg = SubjectConfig(tmp_cfg)
            extractor = Global_Windower_and_Feature_Extractor(subject_cfg)
            try:
                extractor.main()
                return True
            except KeyError as e:
                if "session_id" in str(e) or "batch_id" in str(e):
                    print("\n[WARNING] Extractor returned 0 windows. Skipping extraction.")
                    return False
                raise

    def main(self) -> Optional[Path]:
        if self.current_exp == "inter_session" and len(self.selected_sessions) < 2:
            print(f"[SKIP][inter_session] needs >=2 sessions. Got {len(self.selected_sessions)}.")
            return None

        reset_all_seeds()
        
        combo_data_root = self._ensure_combo_data_root(self.working_data_root / self.run_label)
        
        # 1. Performing windowing once per combination to generate the data for all subsequent training runs of that combo
        extraction_success = self._run_windowing(combo_data_root)
        if not extraction_success:
            print(f"[SKIP] No windows generated for {self.run_label}.")
            return None

        # 2. Build the artifacts directory path based on the ablation type and run label
        artifacts_n = self.main_model_dire / self.ablation_folder_name / self.run_label / f"{self.n_sessions}_sess"
        artifacts_n.mkdir(parents=True, exist_ok=True)
        
        cfg_run = deepcopy(self.base_cfg)
        cfg_run["data"]["data_directory"] = str(combo_data_root)
        cfg_run["data"]["models_main_directory"] = str(artifacts_n)
        cfg_run["data_augmentation"] = deepcopy(self.data_augmentation)
        
        # 3. Delegate to the right trainer for the training and evaluation
        try:
            if self.current_exp == "global":
                trainer = Global_Model_Trainer(base_config=cfg_run, model_config=self.model_cfg)
            elif self.current_exp == "inter_session":
                trainer = Inter_Session_Model_Trainer(base_config=cfg_run, model_config=self.model_cfg, experiment_subdir="inter_session")
            else:
                raise ValueError(f"Unknown experiment_type: {self.current_exp}")

            # Extract the already-generated windows for this combo and filter by the selected sessions
            df_augmented_size = load_all_h5files_from_folder(trainer.data_dire_proc[0], key="wins_feats", print_statistics=False)
            if df_augmented_size.empty: 
                raise ValueError("The loaded DataFrame is completely empty.")

            trainer.df = df_augmented_size[df_augmented_size["session_id"].isin(self.selected_sessions)].copy().reset_index(drop=True)
            if trainer.df.empty:
                raise ValueError("No rows left after filtering sessions.")
                
            trainer._save_run_cfg()
            trainer: Any = trainer
            
            if self.current_exp == "global":
                trainer.run_cv()
            elif self.current_exp == "inter_session":
                if len(np.sort(trainer.df["session_id"].unique())) < 2:
                    raise ValueError("inter_session requires at least 2 sessions after filtering.")
                val_size = float(cfg_run.get("cv", {}).get("val_size", 0.3))
                seed = int(cfg_run.get("experiment", {}).get("seed", 0))
                trainer.run_inter_session_cv(val_size=val_size, seed=seed)

            pd.DataFrame(trainer.cv_summaries).to_csv(trainer.model_dire / "cv_summary.csv", index=False)
            return trainer.model_dire

        except ValueError as e:
            print(f"[SKIP] Un-trainable for {self.run_label} ({self.n_sessions} sessions) - {self.current_exp}: {e}")
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=Path, required=True)
    ap.add_argument("--model_config", type=Path, required=True)
    ap.add_argument("--window_config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, required=True)
    ap.add_argument("--experiment", nargs="+", choices=["global", "inter_session"], default=["global", "inter_session"])
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    ap.add_argument("--aug_windows_s", nargs="*", type=float, default=[1.4])
    ap.add_argument("--stride_ms", nargs="*", type=int, default=[10])
    ap.add_argument("--num_strides", nargs="*", type=int, default=[2, 5, 10])
    ap.add_argument("--min_sessions", type=int, default=1)
    ap.add_argument("--train_mode", choices=["augmented_size", "original_size"], default=None)
    ap.add_argument("--skip_baseline", action="store_true")
    args = ap.parse_args()

    varies_stride = len(args.stride_ms) > 1
    varies_num = len(args.num_strides) > 1

    if varies_stride and not varies_num:
        ablation_folder_name = "data_augmentation_ablation_stride_dim"
    elif varies_num and not varies_stride:
        ablation_folder_name = "data_augmentation_ablation_num_strides"
    else:
        ablation_folder_name = "data_augmentation_ablation"

    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())
    window_cfg = yaml.safe_load(args.window_config.read_text())

    base_cfg.setdefault("data", {})
    base_cfg["data"]["data_directory"] = str(args.data_dir)
    base_cfg["data"]["models_main_directory"] = str(args.artifacts_dir)
    
    cli_train_mode = args.train_mode
    base_train_mode = base_cfg.get("experiment", {}).get("augmentation_train_mode")
    yaml_train_mode = window_cfg.get("data_augmentation", {}).get("train_mode")
    train_mode = str(cli_train_mode or base_train_mode or yaml_train_mode or "augmented_size").strip().lower()
    
    base_cfg.setdefault("experiment", {})["augmentation_train_mode"] = train_mode
    base_cfg["experiment"]["window_config_template"] = window_cfg

    augmentation_combos: Sequence[Tuple[Optional[int], Optional[int]]] = list(product(args.stride_ms, args.num_strides))
    if not args.skip_baseline:
        augmentation_combos = [(None, None)] + augmentation_combos

    print(f"\n[ABLATION] Data Augmentation Ablation (Folder: {ablation_folder_name})")

    # 1. Loop on augmentation combinations (stride_ms, num_strides)
    for stride_ms, num_strides in augmentation_combos:
        if stride_ms is None or num_strides is None:
            data_augmentation = {"mode": "disabled"}
            run_label = "baseline"
        else:
            data_augmentation = {
                "mode": "sliding_window", "stride_ms": int(stride_ms), "num_strides": int(num_strides)
            }
            run_label = f"stride{stride_ms}_n{num_strides}"

        print(f"\n{'='*90}")
        print(f"[STARTING VARIANT] -> {run_label.upper()}")
        print(f"{'='*90}")

        # Build a temporary trainer to access _ensure_combo_data_root and _run_windowing.
        cfg_windowing_base = deepcopy(base_cfg)
        cfg_windowing_base["data"]["subject_id"] = args.subjects[0]
        cfg_windowing_base["condition"] = args.conditions[0]
        cfg_windowing_base.setdefault("experiment", {})
        cfg_windowing_base["experiment"]["augmentation_ablation"] = {
            "num_sessions": 1,
            "session_ids": [],
            "experiment_type": "global",
            "data_augmentation": data_augmentation,
            "run_label": run_label,
            "ablation_folder_name": ablation_folder_name,
        }

        for window_s in args.aug_windows_s:
            cfg_windowing_base.setdefault("window", {})["window_size_s"] = float(window_s)

            for subject in args.subjects:
                for condition in args.conditions:
                    sessions = discover_sessions(args.data_dir, subject, condition)
                    if not sessions:
                        continue

                    cfg_w = deepcopy(cfg_windowing_base)
                    cfg_w["data"]["subject_id"] = subject
                    cfg_w["condition"] = condition

                    windowing_trainer = Data_Augmentation_Ablation_Trainer(
                        base_config=cfg_w, model_config=model_cfg
                    )
                    combo_data_root = windowing_trainer._ensure_combo_data_root(
                        windowing_trainer.working_data_root / run_label
                    )
                    print(f"\n[WINDOWING] {run_label} | {subject} | {condition} | w={window_s}s")
                    extraction_success = windowing_trainer._run_windowing(combo_data_root)
                    if not extraction_success:
                        print(f"[SKIP] No windows generated for {run_label} | {subject} | {condition}.")
                        continue

                    # 2. Loop on number of sessions (1 to all)
                    for n_sessions in range(args.min_sessions, len(sessions) + 1):
                        selected_sessions = sessions[:n_sessions]

                        # 3. Loop on experiment type
                        for current_exp in args.experiment:
                            if current_exp == "inter_session" and n_sessions < 2:
                                continue

                            cfg_run = deepcopy(base_cfg)
                            cfg_run["data"]["subject_id"] = subject
                            cfg_run["condition"] = condition
                            cfg_run.setdefault("window", {})["window_size_s"] = float(window_s)
                            cfg_run.setdefault("experiment", {})
                            cfg_run["experiment"]["augmentation_ablation"] = {
                                "num_sessions": n_sessions,
                                "session_ids": selected_sessions,
                                "experiment_type": current_exp,
                                "data_augmentation": data_augmentation,
                                "run_label": run_label,
                                "ablation_folder_name": ablation_folder_name,  # Pass the folder name dynamically
                            }

                            print(f"-> {current_exp.upper()} | {run_label} | {n_sessions} Sess | {subject} | {condition} | w={window_s}s")
                            trainer = Data_Augmentation_Ablation_Trainer(
                                base_config=cfg_run, model_config=model_cfg
                            )

                            # Skip windowing
                            combo_data_root_run = trainer.working_data_root / run_label
                            
                            # Creating artifacts directory for this run
                            artifacts_n = trainer.main_model_dire / ablation_folder_name / run_label / f"{n_sessions}_sess"
                            artifacts_n.mkdir(parents=True, exist_ok=True)

                            cfg_train = deepcopy(cfg_run)
                            cfg_train["data"]["data_directory"] = str(combo_data_root_run)
                            cfg_train["data"]["models_main_directory"] = str(artifacts_n)
                            cfg_train["data_augmentation"] = deepcopy(data_augmentation)

                            try:
                                if current_exp == "global":
                                    inner = Global_Model_Trainer(base_config=cfg_train, model_config=model_cfg)
                                else:
                                    inner = Inter_Session_Model_Trainer(
                                        base_config=cfg_train, model_config=model_cfg,
                                        experiment_subdir="inter_session"
                                    )

                                df_full = load_all_h5files_from_folder(
                                    inner.data_dire_proc[0], key="wins_feats", print_statistics=False
                                )
                                if df_full.empty:
                                    raise ValueError("The loaded DataFrame is completely empty.")

                                inner.df = df_full[df_full["session_id"].isin(selected_sessions)].copy().reset_index(drop=True)
                                if inner.df.empty:
                                    raise ValueError("No rows left after filtering sessions.")
                                if current_exp == "inter_session" and len(inner.df["session_id"].unique()) < 2:
                                    raise ValueError("inter_session requires at least 2 sessions after filtering.")

                                inner._save_run_cfg()
                                inner: Any = inner
                                if current_exp == "global":
                                    inner.run_cv()
                                else:
                                    val_size = float(cfg_train.get("cv", {}).get("val_size", 0.3))
                                    seed = int(cfg_train.get("experiment", {}).get("seed", 0))
                                    inner.run_inter_session_cv(val_size=val_size, seed=seed)

                                pd.DataFrame(inner.cv_summaries).to_csv(
                                    inner.model_dire / "cv_summary.csv", index=False
                                )
                                print(f"[DONE] Saved to: {inner.model_dire}")

                            except ValueError as e:
                                print(f"[SKIP] Un-trainable for {run_label} ({n_sessions} sessions) - {current_exp}: {e}")

if __name__ == "__main__":
    main()