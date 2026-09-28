#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Random-label control for the inter-session setting.

Sanity check against data/label leakage: the model is trained with the class labels
randomly permuted, while preprocessing, rest downsampling, data splits, model and
training pipeline are identical to II_inter_session_models.py.

- Train and validation labels are permuted (separately, so class counts are preserved).
  Validation labels are permuted too, because the validation loss drives checkpoint
  selection, early stopping and the LR scheduler; true validation labels would leak
  label information into the selected model.
- Test labels are left untouched: accuracy is measured against the true labels.

Expected result: balanced accuracy at chance level (1/9 = 11.1% with rest included).

Outputs are saved under:
    <artifacts_dir>/models/inter_session_random_labels/<subject>/<condition>/<model_name>/w<ms>ms/model_<k>/

Usage:
    python offline_experiments/V_random_label_control.py \
        --data_dir /path/to/data --artifacts_dir artifacts_rebuttal
"""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from models.seeds import configure_seed

LABEL_COLS = ["Label_int", "Label_str"]


def permute_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of df with the label columns jointly permuted across rows."""
    df = df.copy()
    perm = rng.permutation(len(df))
    for col in LABEL_COLS:
        df[col] = df[col].to_numpy()[perm]
    return df


class RandomLabel_Inter_Session_Trainer(Inter_Session_Model_Trainer):
    """Inter-session trainer that permutes train/val labels before each fold."""

    def __init__(self, base_config: dict, model_config: dict) -> None:
        super().__init__(
            base_config=base_config,
            model_config=model_config,
            experiment_subdir="inter_session_random_labels",
        )
        self.label_permutation_seed = int(
            self.base_config["experiment"]["label_permutation_seed"]
        )

    def _run_one_fold(
        self,
        fold_id: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mode: str,
        test_session_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Independent, reproducible permutation per fold
        rng = np.random.default_rng([self.label_permutation_seed, fold_id])
        train_perm = permute_labels(train_df, rng)
        val_perm = permute_labels(val_df, rng)

        frac_train_kept = float((train_perm["Label_int"] == train_df["Label_int"]).mean())
        frac_val_kept = float((val_perm["Label_int"] == val_df["Label_int"]).mean())
        print(
            f"[RANDOM LABELS] fold {fold_id+1}: labels unchanged by permutation "
            f"train={frac_train_kept:.1%} val={frac_val_kept:.1%} (expected ~1/num_classes)"
        )

        row_summary = super()._run_one_fold(
            fold_id=fold_id,
            train_df=train_perm,
            val_df=val_perm,
            test_df=test_df,
            mode=mode,
            test_session_id=test_session_id,
        )
        row_summary["label_permutation_seed"] = self.label_permutation_seed
        row_summary["frac_train_labels_unchanged"] = frac_train_kept
        row_summary["frac_val_labels_unchanged"] = frac_val_kept
        return row_summary


def main():
    ap = argparse.ArgumentParser(description="Random-label control (inter-session)")
    ap.add_argument(
        "--base_config",
        type=Path,
        default=REPO_ROOT / "config" / "rebuttal_random_labels_config.yaml",
    )
    ap.add_argument(
        "--model_config",
        type=Path,
        default=REPO_ROOT / "config" / "models_configs" / "speechnet_config.yaml",
    )
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, default=Path("./artifacts_rebuttal"))
    ap.add_argument(
        "--win_and_feats",
        type=str,
        default=None,
        help="Override paths.win_and_feats (name of the windows folder inside data_dir)",
    )
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed (e.g. 42, 52, 62); also used as label permutation seed. "
        "Default: values of the base config",
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
        base_cfg["experiment"]["label_permutation_seed"] = args.seed
    configure_seed(base_cfg["experiment"]["seed"])

    for sub in args.subjects:
        for cond in args.conditions:
            cfg_run = deepcopy(base_cfg)
            cfg_run["data"]["subject_id"] = sub
            cfg_run["condition"] = cond

            print("\n" + "=" * 80)
            print(f"=== RANDOM-LABEL INTER-SESSION | {sub} | {cond} ===")
            print("=" * 80)

            trainer = RandomLabel_Inter_Session_Trainer(base_config=cfg_run, model_config=model_cfg)
            out_dir = trainer.main()
            print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
