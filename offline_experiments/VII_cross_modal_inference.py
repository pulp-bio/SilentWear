#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Cross-modal evaluation of the inter-session models (inference only, no training).

Question
    Does a model trained on one articulation mode transfer to the other one?
    vocalized -> silent and silent -> vocalized.

Procedure
    For every subject, source condition C and fold k of the inter-session setting
    (II_inter_session_models.py, test session j held out), the saved checkpoint
    leave_one_session_out_fold_<k>.pt is evaluated on session j of the *other*
    condition C' (whole session, rest class not downsampled). Session j of C' is recorded
    in the same donning as session j of C, so it is unseen in both modes.
    Configs (base_cfg / model_cfg) are read from the source run_cfg.json.

    Sanity check: every checkpoint is also re-evaluated on session j of its own
    condition; the result must equal balanced_accuracy in the source cv_summary.csv.

Outputs (same layout and columns as the inter-session runs, so that
utils/III_results_analysis/I_global_intersession_analysis.py and 00_pool_seeds.py apply
unchanged; <condition> is the TEST condition, the model was trained on the other one):
    <artifacts_dir>/models/inter_session_cross_modal/<subject>/<test condition>/<model_name>/<model_name_id>/<model_run>/
        cv_summary.csv
        run_cfg.json

Usage:
    python offline_experiments/VII_cross_modal_inference.py \
        --data_dir /path/to/data --artifacts_dir artifacts_rebuttal/seed_42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.Model_Master import Model_Master
from models.TorchTrainer import evaluate_model
from utils.general_utils import load_all_h5files_from_folder

SUBJECTS = ["S01", "S02", "S03", "S04"]
CONDITIONS = ["silent", "vocalized"]
OTHER = {"silent": "vocalized", "vocalized": "silent"}


def load_condition(base_cfg: dict, sub: str, cond: str) -> pd.DataFrame:
    win_ms = int(float(base_cfg["window"]["window_size_s"]) * 1000)
    d = (
        Path(base_cfg["data"]["data_directory"])
        / base_cfg["paths"]["win_and_feats"]
        / sub
        / cond
        / f"WIN_{win_ms}"
    )
    if not d.exists():
        raise FileNotFoundError(f"Windows directory does not exist: {d}")
    return load_all_h5files_from_folder(d, key="wins_feats", print_statistics=False).reset_index(
        drop=True
    )


def build_master(base_cfg: dict, model_cfg: dict, df: pd.DataFrame) -> Model_Master:
    """Model skeleton + label mapping; df is only used to resolve the input columns."""
    mm = Model_Master(base_config=base_cfg, model_config=model_cfg)
    mm.df_train = df
    mm.df_val = pd.DataFrame()
    mm.df_test = pd.DataFrame()
    mm.generate_training_labels()
    mm.remap_all_datasets()
    mm.register_model()
    return mm


def load_weights(mm: Model_Master, ckpt_path: Path) -> None:
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    mm.model.load_state_dict(state_dict, strict=True)


def evaluate(mm: Model_Master, df_session: pd.DataFrame):
    d = mm.apply_label_mapping(df_session, orig_to_train=mm.orig_to_train)
    loader = mm.trainer_manager.create_dataloader_from_df(
        d[mm.data_col_to_consider + ["Label_train"]], shuffle=False
    )
    metrics, y_true, y_pred = evaluate_model(mm.model, loader)
    return metrics, y_true, y_pred, d.index


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


def run_subject(args, sub: str, train_cond: str) -> None:
    test_cond = OTHER[train_cond]
    src = (
        args.artifacts_dir / "models" / "inter_session" / sub / train_cond
        / args.model_name / args.model_name_id / args.model_run
    )
    src_cfg = json.loads((src / "run_cfg.json").read_text())
    base_cfg, model_cfg = src_cfg["base_cfg"], src_cfg["model_cfg"]
    if args.data_dir is not None:
        base_cfg["data"]["data_directory"] = str(args.data_dir)
    if args.win_and_feats is not None:
        base_cfg["paths"]["win_and_feats"] = args.win_and_feats
    src_summary = pd.read_csv(src / "cv_summary.csv")

    df_src = load_condition(base_cfg, sub, train_cond)
    df_tgt = load_condition(base_cfg, sub, test_cond)
    ch_src = sorted(c for c in df_src.columns if c.startswith("Ch_"))
    ch_tgt = sorted(c for c in df_tgt.columns if c.startswith("Ch_"))
    if ch_src != ch_tgt:
        raise ValueError(f"{sub}: channel columns differ between {train_cond} and {test_cond}")

    rows = []
    for _, fold in src_summary.iterrows():
        k, j = int(fold["fold_num"]), int(fold["test_session"])
        ckpt = src / f"{fold['cv_mode']}_fold_{k}.pt"
        mm = build_master(base_cfg, model_cfg, df_tgt)
        load_weights(mm, ckpt)

        # sanity: own condition, same held-out session -> must reproduce the source score
        m_own, _, _, _ = evaluate(mm, df_src[df_src["session_id"] == j])
        if not np.isclose(m_own["balanced_accuracy"], fold["balanced_accuracy"], rtol=0, atol=1e-9):
            raise RuntimeError(
                f"{sub} {train_cond} fold {k}: re-evaluation {m_own['balanced_accuracy']:.6f} "
                f"!= cv_summary {fold['balanced_accuracy']:.6f}"
            )

        metrics, y_true, y_pred, idx = evaluate(mm, df_tgt[df_tgt["session_id"] == j])
        print(
            f"{sub} | train {train_cond:9s} -> test {test_cond:9s} | session {j} | "
            f"bal.acc {metrics['balanced_accuracy']:.4f} (own condition {m_own['balanced_accuracy']:.4f})"
        )
        rows.append(
            {
                "cv_mode": fold["cv_mode"],
                "fold_num": k,
                "test_session": j,
                **to_row(metrics),
                "train_condition": train_cond,
                "own_condition_balanced_accuracy": float(m_own["balanced_accuracy"]),
                "test_idx": idx.tolist(),
                "y_true": np.asarray(y_true).tolist(),
                "y_pred": np.asarray(y_pred).tolist(),
                "checkpoint": str(ckpt),
            }
        )

    out = (
        args.artifacts_dir / "models" / "inter_session_cross_modal" / sub / test_cond
        / args.model_name / args.model_name_id / args.model_run
    )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "cv_summary.csv", index=False)
    run_cfg = {
        "experiment_type": "inter_session_cross_modal",
        "subject": sub,
        "condition": test_cond,
        "train_condition": train_cond,
        "source_run": str(src),
        "base_cfg": base_cfg,
        "model_cfg": model_cfg,
        "seeds": src_cfg.get("seeds"),
    }
    (out / "run_cfg.json").write_text(json.dumps(run_cfg, indent=4, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(description="Cross-modal inference of the inter-session models")
    ap.add_argument("--artifacts_dir", type=Path, required=True, help="e.g. artifacts_rebuttal/seed_42")
    ap.add_argument("--data_dir", type=Path, default=None, help="Override data.data_directory")
    ap.add_argument("--win_and_feats", type=str, default=None, help="Override paths.win_and_feats")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--train_conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--model_name", default="speechnet")
    ap.add_argument("--model_name_id", default="w1400ms")
    ap.add_argument("--model_run", default="model_1")
    args = ap.parse_args()

    for sub in args.subjects:
        for cond in args.train_conditions:
            run_subject(args, sub, cond)
    print(f"[DONE] {args.artifacts_dir}/models/inter_session_cross_modal")


if __name__ == "__main__":
    main()
