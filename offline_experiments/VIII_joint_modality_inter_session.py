#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Joint-modality training for the inter-session setting.

Question
    Does training one model on silent AND vocalized data help, compared with the
    mode-specific models of II_inter_session_models.py?

Procedure (per subject, leave-one-session-out as in Setting II)
    - Data: all windows of both conditions, tagged with a "condition" column.
    - Fold j: train/val on sessions != j of both conditions; test on session j of both
      conditions (session j is recorded in the same device placement for both modes,
      so it is unseen in both). Silent and vocalized test windows are scored separately
      (whole session, rest class not downsampled).
    - Rest downsampling: separately per condition, as in Setting II.
    - Train/val split: val_size from the config, stratified by label and condition.
    - Model, training configuration and seeds: unchanged (SpeechNet, base config).
    The joint model sees twice the training data of a mode-specific model.

Outputs:
    <artifacts_dir>/models/inter_session_joint/<subject>/voc_and_silent/<model_name>/w<ms>ms/model_<k>/
        leave_one_session_out_fold_<k>.pt, cv_summary.csv (both test conditions), run_cfg.json
    <artifacts_dir>/models/inter_session_joint/<subject>/<test condition>/<model_name>/w<ms>ms/model_<k>/
        cv_summary.csv, run_cfg.json   (same layout as the inter-session runs, so that
        utils/III_results_analysis/I_global_intersession_analysis.py applies unchanged)

Usage:
    python offline_experiments/VIII_joint_modality_inter_session.py \
        --data_dir /path/to/data --artifacts_dir artifacts_rebuttal/seed_42 --seed 42
    python utils/III_results_analysis/I_global_intersession_analysis.py \
        --artifacts_dir artifacts_rebuttal/seed_42 --experiment inter_session_joint \
        --model_name speechnet --model_name_id w1400ms
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from models.seeds import configure_seed
from models.TorchTrainer import evaluate_model
from utils.general_utils import load_all_h5files_from_folder

CONDITIONS = ["silent", "vocalized"]
JOINT_CONDITION = "voc_and_silent"


def to_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for k, v in metrics.items():
        if isinstance(v, (np.ndarray, list, tuple)):
            row[k] = json.dumps(np.asarray(v).tolist())
        elif isinstance(v, np.floating):
            row[k] = float(v)
        else:
            row[k] = v
    return row


class JointModality_Inter_Session_Trainer(Inter_Session_Model_Trainer):
    """Inter-session trainer on both conditions, scored separately per test condition."""

    def __init__(self, base_config: dict, model_config: dict) -> None:
        base_config = deepcopy(base_config)
        base_config["condition"] = JOINT_CONDITION
        super().__init__(
            base_config=base_config,
            model_config=model_config,
            experiment_subdir="inter_session_joint",
        )

    def _load_condition(self, cond: str) -> pd.DataFrame:
        d = (
            self.main_dire
            / self.base_config["paths"]["win_and_feats"]
            / str(self.sub_id)
            / cond
            / f"WIN_{self.window_size_ms}"
        )
        df = load_all_h5files_from_folder(d, key="wins_feats", print_statistics=False)
        df["condition"] = cond
        return df

    def _downsample_rest(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Balance rest to the smallest word class, separately per condition (as Setting II)."""
        parts = []
        for _, d in df.groupby("condition", sort=True):
            min_samples = d["Label_int"].value_counts().min()
            rest = d[d["Label_str"] == "rest"]
            keep = rest.sample(n=min_samples, random_state=seed).index
            parts.append(d.drop(index=rest.index.difference(keep)))
        return pd.concat(parts)

    def _evaluate(self, test_df: pd.DataFrame):
        mm = self.model_master
        d = mm.apply_label_mapping(test_df, orig_to_train=mm.orig_to_train)
        loader = mm.trainer_manager.create_dataloader_from_df(
            d[mm.data_col_to_consider + ["Label_train"]], shuffle=False
        )
        metrics, y_true, y_pred = evaluate_model(mm.model, loader)
        return metrics, y_true, y_pred, d.index

    def run_inter_session_cv(self, val_size: float = 0.3, seed: int = 0) -> List[Dict[str, Any]]:
        self.cv_summaries = []
        sessions = np.sort(self.df["session_id"].unique())

        for fold_id, j in enumerate(sessions):
            print(f"\n\n=== JOINT LOSO FOLD {fold_id+1}/{len(sessions)} | test_session={j} ===")
            train_val = self.df[self.df["session_id"] != j]
            if self.include_rest:
                train_val = self._downsample_rest(train_val, seed)
            strata = train_val["Label_int"].astype(str) + "_" + train_val["condition"]
            train_df, val_df = train_test_split(
                train_val, test_size=val_size, shuffle=True, random_state=seed, stratify=strata
            )
            test = {c: self.df[(self.df["session_id"] == j) & (self.df["condition"] == c)] for c in CONDITIONS}

            # trains the model (and evaluates it on the first test condition)
            first = CONDITIONS[0]
            row = self._run_one_fold(
                fold_id=fold_id,
                train_df=train_df,
                val_df=val_df,
                test_df=test[first],
                mode="leave_one_session_out",
                test_session_id=int(j),
            )
            row["condition"] = first
            self.cv_summaries.append(row)

            for c in CONDITIONS[1:]:
                metrics, y_true, y_pred, idx = self._evaluate(test[c])
                row_c = {
                    "cv_mode": row["cv_mode"],
                    "fold_num": row["fold_num"],
                    "test_session": row["test_session"],
                    **to_row(metrics),
                    "train_idx": row["train_idx"],
                    "val_idx": row["val_idx"],
                    "test_idx": idx.tolist(),
                    "y_true": np.asarray(y_true).tolist(),
                    "y_pred": np.asarray(y_pred).tolist(),
                    "condition": c,
                }
                self.cv_summaries.append(row_c)

            for r in self.cv_summaries[-len(CONDITIONS):]:
                print(f"    test {r['condition']:9s}: balanced accuracy {r['balanced_accuracy']:.4f}")

        return self.cv_summaries

    def main(self) -> Path:
        self.df = pd.concat([self._load_condition(c) for c in CONDITIONS], ignore_index=True)
        self._save_run_cfg()

        cv_cfg = self.base_config.get("cv", {})
        val_size = float(cv_cfg.get("val_size", 0.3))
        seed = int(self.base_config.get("experiment", {}).get("seed", 0))
        self.run_inter_session_cv(val_size=val_size, seed=seed)

        summary = pd.DataFrame(self.cv_summaries)
        summary.to_csv(self.model_dire / "cv_summary.csv", index=False)

        # one folder per test condition, same layout as the mode-specific inter-session runs
        run_cfg = json.loads((self.model_dire / "run_cfg.json").read_text())
        for c in CONDITIONS:
            out = self.model_dire.parents[3] / c / self.model_name / self.model_name_id / self.model_dire.name
            out.mkdir(parents=True, exist_ok=True)
            summary[summary["condition"] == c].to_csv(out / "cv_summary.csv", index=False)
            cfg_c = {**run_cfg, "condition": c, "train_conditions": CONDITIONS, "joint_run": str(self.model_dire)}
            (out / "run_cfg.json").write_text(json.dumps(cfg_c, indent=4, sort_keys=True))
        return self.model_dire


def main():
    ap = argparse.ArgumentParser(description="Joint-modality training (inter-session)")
    ap.add_argument("--base_config", type=Path, default=REPO_ROOT / "config" / "paper_models_config.yaml")
    ap.add_argument(
        "--model_config",
        type=Path,
        default=REPO_ROOT / "config" / "models_configs" / "speechnet_config.yaml",
    )
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, default=Path("./artifacts_rebuttal"))
    ap.add_argument("--win_and_feats", type=str, default=None, help="Override paths.win_and_feats")
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed (e.g. 42, 52, 62). Default: experiment.seed of the base config",
    )
    args = ap.parse_args()

    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())
    base_cfg["data"]["data_directory"] = str(args.data_dir)
    base_cfg["data"]["models_main_directory"] = str(args.artifacts_dir)
    if args.win_and_feats is not None:
        base_cfg["paths"]["win_and_feats"] = args.win_and_feats
    if args.seed is not None:
        base_cfg["experiment"]["seed"] = args.seed
    configure_seed(base_cfg["experiment"]["seed"])

    for sub in args.subjects:
        cfg_run = deepcopy(base_cfg)
        cfg_run["data"]["subject_id"] = sub
        print("\n" + "=" * 80)
        print(f"=== JOINT-MODALITY INTER-SESSION | {sub} ===")
        print("=" * 80)
        out_dir = JointModality_Inter_Session_Trainer(base_config=cfg_run, model_config=model_cfg).main()
        print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
