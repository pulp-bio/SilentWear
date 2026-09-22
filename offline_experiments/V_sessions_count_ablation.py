#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Session-count ablation for offline experiments.

This script runs `global` or `inter_session` experiments by progressively increasing the
number of included sessions from 1 to N available.
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.I_global_models import Global_Model_Trainer
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.general_utils import reset_all_seeds
from utils.general_utils import load_all_h5files_from_folder
from offline_experiments.general_utils import discover_sessions

DEFAULT_WINDOWS_S = [1.4]
DEFAULT_SUBJECTS = ["S01", "S02", "S03", "S04"]
DEFAULT_EXPERIMENTS = ["global", "inter_session"]
DEFAULT_CONDITIONS = ["silent", "vocalized"]


class Session_Count_Ablation_Trainer:
    def __init__(self, base_config: dict, model_config: dict) -> None:
        self.base_config = deepcopy(base_config)
        self.model_config = deepcopy(model_config)

        self.sub_id = self.base_config["data"]["subject_id"]
        self.condition = self.base_config["condition"]
        self.main_model_dire = Path(self.base_config["data"]["models_main_directory"])
        
        ablation_cfg = self.base_config.setdefault("experiment", {}).get("sessions_ablation", {})
        self.n_sessions = ablation_cfg.get("num_sessions", 1)
        self.selected_sessions = ablation_cfg.get("session_ids", [])
        self.current_exp = ablation_cfg.get("experiment_type", "global")

    def _load_windows_df(self, data_dirs: List[Path]) -> pd.DataFrame:
        """Load and concatenate windows DataFrames from multiple directories."""
        frames = [
            load_all_h5files_from_folder(d, key="wins_feats", print_statistics=False)
            for d in data_dirs
        ]
        return pd.concat(frames, ignore_index=True).reset_index(drop=True)

    def main(self) -> Optional[Path]:
        if self.current_exp == "inter_session" and len(self.selected_sessions) < 2:
            print(f"[SKIP][inter_session] needs >=2 sessions. Got {len(self.selected_sessions)}.")
            return None

        cfg_run = deepcopy(self.base_config)
        artifacts_n = self.main_model_dire / "session_count_ablation" / f"{self.n_sessions}_sess"
        cfg_run["data"]["models_main_directory"] = str(artifacts_n)

        if self.current_exp == "global":
            trainer = Global_Model_Trainer(base_config=cfg_run, model_config=self.model_config)
        elif self.current_exp == "inter_session":
            trainer = Inter_Session_Model_Trainer(
                base_config=cfg_run, model_config=self.model_config, experiment_subdir="inter_session"
            )
        else:
            raise ValueError(f"Unknown experiment_type: {self.current_exp}")

        # Filter the unified DataFrame for the requested sessions
        df_augmented_size = self._load_windows_df(trainer.data_dire_proc)
        trainer.df = df_augmented_size[df_augmented_size["session_id"].isin(self.selected_sessions)].copy().reset_index(drop=True)

        reset_all_seeds()

        if self.current_exp == "global":
            if trainer.df.empty:
                print(f"[SKIP][global] No rows left for sessions={self.selected_sessions}")
                return None
            trainer._save_run_cfg()
            trainer.run_cv()
            pd.DataFrame(trainer.cv_summaries).to_csv(trainer.model_dire / "cv_summary.csv", index=False)
            return trainer.model_dire

        elif self.current_exp == "inter_session":
            if len(np.sort(trainer.df["session_id"].unique())) < 2:
                print(f"[SKIP][inter_session] not enough unique sessions left after filtering.")
                return None
            trainer._save_run_cfg()
            val_size = float(cfg_run.get("cv", {}).get("val_size", 0.3))
            seed = int(cfg_run.get("experiment", {}).get("seed", 0))
            trainer: Any = trainer
            trainer.run_inter_session_cv(val_size=val_size, seed=seed)
            pd.DataFrame(trainer.cv_summaries).to_csv(trainer.model_dire / "cv_summary.csv", index=False)
            return trainer.model_dire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=Path, required=True)
    ap.add_argument("--model_config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, required=True)
    ap.add_argument("--experiment", nargs="+", choices=DEFAULT_EXPERIMENTS, default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--min_sessions", type=int, default=1)
    ap.add_argument("--windows_s", nargs="*", type=float, default=DEFAULT_WINDOWS_S)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())
    base_cfg.setdefault("data", {})
    base_cfg["data"]["data_directory"] = str(args.data_dir)
    base_cfg["data"]["models_main_directory"] = str(args.artifacts_dir)

    max_sessions_global = max(
        len(discover_sessions(args.data_dir, subj, cond))
        for subj in args.subjects
        for cond in args.conditions
    )

    if max_sessions_global == 0:
        print("[ERROR] No sessions found for any subject/condition.")
        return

    print(f"\n[ABLATION] Running session-count ablation from {args.min_sessions} to {max_sessions_global} sessions.")

    for n_sessions in range(args.min_sessions, max_sessions_global + 1):
        print(f"\n{'='*90}")
        print(f"[STARTING PHASE] -> {n_sessions} SESSION(S)")
        print(f"{'='*90}")

        for current_exp in args.experiment:
            
            if current_exp == "inter_session" and n_sessions < 2:
                continue
                
            for window_s in args.windows_s:
                for subject in args.subjects:
                    for condition in args.conditions:
                        sessions = discover_sessions(args.data_dir, subject, condition)
                        if n_sessions > len(sessions):
                            continue
                        
                        selected_sessions = sessions[:n_sessions]
                        
                        cfg_run = deepcopy(base_cfg)
                        cfg_run["data"]["subject_id"] = subject
                        cfg_run["condition"] = condition
                        cfg_run.setdefault("window", {})["window_size_s"] = float(window_s)
                        
                        cfg_run.setdefault("experiment", {})
                        cfg_run["experiment"]["sessions_ablation"] = {
                            "num_sessions": n_sessions,
                            "session_ids": selected_sessions,
                            "experiment_type": current_exp,
                        }

                        print(f"-> {current_exp.upper()} | {subject} | {condition} | w={window_s}s")
                        trainer = Session_Count_Ablation_Trainer(base_config=cfg_run, model_config=model_cfg)
                        out_dir = trainer.main()
                        if out_dir:
                            print(f"[DONE] Saved to: {out_dir}")


if __name__ == "__main__":
    main()