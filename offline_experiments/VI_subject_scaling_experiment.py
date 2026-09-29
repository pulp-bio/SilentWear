#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Subject-scaling analysis (inter-session setting, SpeechNet, 1400 ms windows).

Question
    Does pre-training on data from other subjects improve accuracy on a new subject,
    both zero-shot and after fine-tuning on one or two of its sessions?

Definitions
    Cohort              the n subjects of the study (--subjects; default S01..S04, n = 4).
    Target subject T    the subject being evaluated (leave-one-subject-out).
    Pool P              the set of subjects used for pre-training, P ⊆ cohort \\ {T}.
    x = |P|             number of pre-training subjects, x ∈ {0, 1, ..., n-1}.
    N(x)                number of pools of size x per target: C(n-1, x); for n = 4: 1, 3, 3, 1.
    
                        x	C(3, x)	Pools
                        0	1	    { } (empty: no pre-training)
                        1	3	    {S01}, {S02}, {S03}
                        2	3	    {S01,S02}, {S01,S03}, {S02,S03}
                        3	1	    {S01,S02,S03}
    Condition           silent or vocalized; every step is run separately per condition.

Pre-training (x >= 1)
    Data: all 3 sessions of every subject in P. The rest class is downsampled to the
    size of the smallest word class, separately for each subject. Train/val split:
    stratified by label, val_size from the config (0.2). Training: base config
    (100 epochs, early stopping on the validation loss).
    For each target subject, the model is pre-trained on each of the 2^(n-1) - 1
    non-empty subsets of the remaining subjects; for n = 4: 7 pools per target,
    28 (target, pool) combinations in total.

Evaluated settings ("bars")
    Every score is the balanced accuracy on one complete held-out session of subject T
    (rest class not downsampled).
    zero_shot   pre-trained model on other subjects, no target data; scored on each of the 3 sessions.  x >= 1
    ft_1sess    trained on 1 session of T; scored on each of the other 2 sessions.
    ft_2sess    trained on 2 sessions of T; scored on the remaining session.
    For x >= 1, training starts from the pre-trained weights of P and uses the paper's
    fine-tuning config (all layers, lr 1e-3, 50 epochs, early stopping). For x = 0,
    the model is trained from random initialisation with the base config; x = 0 with
    ft_2sess therefore equals the paper's inter-session setting.

Aggregation, tables and figures
    utils/III_results_analysis/IV_subject_scaling_analysis.py (reads results.csv).

Outputs:
    <artifacts_dir>/models/subject_scaling/<condition>/w<ms>ms/
        pretrain/<S01+S02>/model.pt
        finetune/<target>/<pool or none>/<bar>_train<sessions>.pt
        results.csv   (one row per test session score; resumable)
        run_cfg.json

Usage:
    python offline_experiments/VI_subject_scaling_experiment.py \
        --data_dir /path/to/data --artifacts_dir artifacts/seed_42 --conditions silent --seed 42
    python utils/III_results_analysis/IV_subject_scaling_analysis.py \
        --artifacts_dir artifacts/seed_42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.IV_inter_session_with_ft import build_ft_model_cfg
from offline_experiments.Model_Master import Model_Master
from models.seeds import configure_seed, get_seed_info
from models.TorchTrainer import evaluate_model
from utils.general_utils import load_all_h5files_from_folder

SUBJECTS = ["S01", "S02", "S03", "S04"]  # default cohort (--subjects)
SESSIONS = [1, 2, 3]  # every subject must have exactly these sessions
BARS = ["zero_shot", "ft_1sess", "ft_2sess"]
META_COLS = ["Label_int", "Label_str", "session_id", "batch_id"]
FT_SETTINGS = {"ft_lr": 1e-3, "num_ft_epochs": 50}  # paper Setting III-a


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_subject(base_cfg: dict, sub: str, cond: str) -> pd.DataFrame:
    win_ms = int(round(float(base_cfg["window"]["window_size_s"]) * 1000))
    d = (
        Path(base_cfg["data"]["data_directory"])
        / base_cfg["paths"]["win_and_feats"]
        / sub
        / cond
        / f"WIN_{win_ms}"
    )
    if not d.exists():
        raise FileNotFoundError(f"Windows directory does not exist: {d}")
    df = load_all_h5files_from_folder(d, key="wins_feats")
    ch_cols = [c for c in df.columns if c.startswith("Ch_") and c.endswith("_filt")]
    df = df[META_COLS + ch_cols].copy()
    df["subject"] = sub
    return df.reset_index(drop=True)


def downsample_rest(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Balance the rest class to the smallest word class, separately per subject."""
    parts = []
    for _, d in df.groupby("subject", sort=True):
        min_samples = d["Label_int"].value_counts().min()
        rest = d[d["Label_str"] == "rest"]
        keep = rest.sample(n=min_samples, random_state=seed).index
        parts.append(d.drop(index=rest.index.difference(keep)))
    return pd.concat(parts)


def split_train_val(df: pd.DataFrame, val_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df, test_size=val_size, shuffle=True, random_state=seed, stratify=df["Label_int"]
    )


# ----------------------------------------------------------------------------
# Model helpers (thin wrappers around Model_Master)
# ----------------------------------------------------------------------------
def build_master(
    base_cfg: dict, model_cfg: dict, df_train: pd.DataFrame, df_val: pd.DataFrame
) -> Model_Master:
    """Build a Model_Master with label mapping, fresh seeds and a new model."""
    mm = Model_Master(base_config=base_cfg, model_config=model_cfg)
    mm.df_train = df_train
    mm.df_val = df_val
    mm.df_test = pd.DataFrame()
    mm.generate_training_labels()
    mm.remap_all_datasets()
    mm.register_model()  # resets seeds before building the model
    return mm


def load_weights(mm: Model_Master, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = mm.model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing} unexpected={unexpected}")


def ckpt_info(ckpt_path: Path) -> Dict[str, Optional[int]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return {"best_epoch": ckpt.get("best_epoch"), "epochs_ran": ckpt.get("epochs_ran")}


def eval_sessions(
    mm: Model_Master, df_target: pd.DataFrame, sessions: Sequence[int]
) -> Dict[int, Dict[str, float]]:
    """Balanced accuracy of mm.model on each whole target session (rest not downsampled)."""
    cols = mm.data_col_to_consider + ["Label_train"]
    out = {}
    for s in sessions:
        d = df_target[df_target["session_id"] == s]
        d = mm.apply_label_mapping(d, orig_to_train=mm.orig_to_train)
        loader = mm.trainer_manager.create_dataloader_from_df(d[cols], shuffle=False)
        metrics, _, _ = evaluate_model(mm.model, loader)
        out[int(s)] = {
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "accuracy": float(metrics["accuracy"]),
            "n_test": int(len(d)),
        }
    return out


def pool_name(pool: Tuple[str, ...]) -> str:
    return "+".join(pool) if pool else "none"


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
class SubjectScalingExperiment:
    def __init__(
        self, base_cfg: dict, model_cfg: dict, cond: str, artifacts_dir: Path, subjects, targets
    ):
        self.cond = cond
        self.base_cfg = deepcopy(base_cfg)
        self.base_cfg["condition"] = cond
        self.model_cfg = model_cfg
        # Fine-tuning configuration. 
        self.ft_model_cfg = build_ft_model_cfg(model_cfg, FT_SETTINGS)
        self.seed = int(self.base_cfg["experiment"]["seed"])
        self.val_size = float(self.base_cfg["cv"]["val_size"])
        self.subjects = list(subjects)
        self.targets = list(targets)
        unknown = set(self.targets) - set(self.subjects)
        if unknown:
            raise ValueError(f"targets {sorted(unknown)} are not in the cohort {self.subjects}")
        self.win_ms = int(round(float(self.base_cfg["window"]["window_size_s"]) * 1000))

        self.out_dir = artifacts_dir / "models" / "subject_scaling" / cond / f"w{self.win_ms}ms"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.out_dir / "results.csv"
        self.done = set()
        if self.results_path.exists():
            prev = pd.read_csv(self.results_path, dtype={"pretrain_subjects": str})
            self.done = set(
                zip(prev.target, prev.pretrain_subjects, prev.bar, prev.unit.astype(int))
            )
            print(f"[RESUME] {len(self.done)} (target, pool, bar, unit) entries already done")

        self.data = {s: load_subject(self.base_cfg, s, cond) for s in self.subjects}
        for s, df in self.data.items():
            found = sorted(int(v) for v in df["session_id"].unique())
            if found != SESSIONS:
                raise ValueError(f"{s}: sessions {found}, expected {SESSIONS}")

    # -- bookkeeping --------------------------------------------------------
    def _cfg_for(self, name: str) -> dict:
        cfg = deepcopy(self.base_cfg)
        cfg["data"]["subject_id"] = name
        return cfg

    def _append(self, rows: List[dict]) -> None:
        df = pd.DataFrame(rows)
        df.to_csv(self.results_path, mode="a", header=not self.results_path.exists(), index=False)

    def save_run_cfg(self) -> None:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            commit = None
        run_cfg = {
            "experiment_type": "subject_scaling",
            "condition": self.cond,
            "window_size_ms": self.win_ms,
            "subjects": self.subjects,
            "targets": self.targets,
            "val_size": self.val_size,
            "split_seed": self.seed,
            "ft_settings": FT_SETTINGS,
            "base_cfg": self.base_cfg,
            "model_cfg": self.model_cfg,
            "ft_model_cfg": self.ft_model_cfg,
            "seeds": get_seed_info(),
            "git_commit": commit,
        }
        with open(self.out_dir / "run_cfg.json", "w") as f:
            json.dump(run_cfg, f, indent=4, sort_keys=True)

    # -- stages -------------------------------------------------------------
    def pretrain(self, pool: Tuple[str, ...]) -> Path:
        ckpt = self.out_dir / "pretrain" / pool_name(pool) / "model.pt"
        if ckpt.exists():
            return ckpt
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n=== PRETRAIN | {self.cond} | pool={pool_name(pool)} ===")
        df = downsample_rest(pd.concat([self.data[s] for s in pool], ignore_index=True), self.seed)
        df_train, df_val = split_train_val(df, self.val_size, self.seed)
        mm = build_master(self._cfg_for(pool_name(pool)), self.model_cfg, df_train, df_val)
        tmp = ckpt.parent / "model.partial.pt"  # TorchTrainer forces a .pt suffix
        mm.trainer_manager.fit(save_model_path=tmp)
        tmp.rename(ckpt)  # only a completed training produces model.pt
        with open(ckpt.parent / "pretrain_info.json", "w") as f:
            json.dump(
                {"pool": list(pool), "n_train": len(df_train), "n_val": len(df_val), **ckpt_info(ckpt)},
                f,
                indent=4,
            )
        return ckpt

    def zero_shot(self, target: str, pool: Tuple[str, ...], ckpt: Path) -> None:
        pname = pool_name(pool)
        todo = [s for s in SESSIONS if (target, pname, "zero_shot", s) not in self.done]
        if not todo:
            return
        df_t = self.data[target]
        # build only for label mapping / model skeleton; no training happens here
        mm = build_master(self._cfg_for(target), self.model_cfg, df_t, pd.DataFrame())
        load_weights(mm, ckpt)
        scores = eval_sessions(mm, df_t, todo)
        self._append(
            [
                self._row(target, pool, "zero_shot", unit=s, train_sessions=[], test_session=s,
                          score=scores[s], n_train=0, n_val=0, info={}, ckpt=ckpt)
                for s in todo
            ]
        )

    def finetune(self, target: str, pool: Tuple[str, ...], ckpt: Optional[Path], bar: str, unit: int) -> None:
        pname = pool_name(pool)
        if (target, pname, bar, unit) in self.done:
            return
        if bar == "ft_1sess":
            train_sessions, test_sessions = [unit], [s for s in SESSIONS if s != unit]
        else:  # ft_2sess
            train_sessions, test_sessions = [s for s in SESSIONS if s != unit], [unit]

        print(f"\n=== {bar.upper()} | {self.cond} | target={target} | pool={pname} | train sess={train_sessions} ===")
        df_t = self.data[target]
        df = downsample_rest(df_t[df_t["session_id"].isin(train_sessions)], self.seed)
        df_train, df_val = split_train_val(df, self.val_size, self.seed)
        model_cfg = self.ft_model_cfg if ckpt is not None else self.model_cfg
        mm = build_master(self._cfg_for(target), model_cfg, df_train, df_val)
        if ckpt is not None:
            load_weights(mm, ckpt)

        save_path = self.out_dir / "finetune" / target / pname / f"{bar}_train{''.join(map(str, train_sessions))}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mm.trainer_manager.fit(save_model_path=save_path)  # restores best-val weights
        scores = eval_sessions(mm, df_t, test_sessions)
        info = ckpt_info(save_path)
        self._append(
            [
                self._row(target, pool, bar, unit=unit, train_sessions=train_sessions, test_session=s,
                          score=scores[s], n_train=len(df_train), n_val=len(df_val), info=info, ckpt=save_path)
                for s in test_sessions
            ]
        )

    def _row(self, target, pool, bar, unit, train_sessions, test_session, score, n_train, n_val, info, ckpt):
        return {
            "condition": self.cond,
            "window_ms": self.win_ms,
            "target": target,
            "n_pretrain_subjects": len(pool),
            "pretrain_subjects": pool_name(pool),
            "bar": bar,
            "unit": int(unit),
            "train_sessions": "".join(map(str, train_sessions)),
            "test_session": int(test_session),
            "balanced_accuracy": score["balanced_accuracy"],
            "accuracy": score["accuracy"],
            "n_train": n_train,
            "n_val": n_val,
            "n_test": score["n_test"],
            "best_epoch": info.get("best_epoch"),
            "epochs_ran": info.get("epochs_ran"),
            "checkpoint": str(ckpt),
        }

    def run(self) -> None:
        self.save_run_cfg()
        # pools build the possible combinations of pre-training subjects 
        # all non-empty proper subsets of the cohort (a pool cannot contain every subject)
        pools = [p for k in range(1, len(self.subjects)) for p in combinations(self.subjects, k)]
        needed = [p for p in pools if any(t not in p for t in self.targets)]
        # Pre-training the needed models for all pools  
        ckpts = {p: self.pretrain(p) for p in needed}

        for target in self.targets:
            others = [s for s in self.subjects if s != target]
            for x in range(len(others) + 1):
                pools_x = [()] if x == 0 else list(combinations(others, x))
                for pool in pools_x:
                    ckpt = ckpts[pool] if pool else None

                    # Zero-shot evaluation on the current pre-traine model
                    if ckpt is not None:
                        self.zero_shot(target, pool, ckpt)
                    # now fine-tune on the target's subject sessions
                    for bar in ("ft_1sess", "ft_2sess"):
                        for unit in SESSIONS:
                            self.finetune(target, pool, ckpt, bar, unit)
        print(f"\n[DONE] {self.cond}: results in {self.results_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Subject-scaling analysis (inter-session, SpeechNet)")
    ap.add_argument("--base_config", type=Path, default=REPO_ROOT / "config" / "paper_models_config.yaml")
    ap.add_argument("--model_config", type=Path, default=REPO_ROOT / "config" / "models_configs" / "speechnet_config.yaml")
    ap.add_argument("--data_dir", type=Path, default=None)
    ap.add_argument("--win_and_feats", type=str, default=None, help="Override paths.win_and_feats")
    ap.add_argument("--artifacts_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS, help="Cohort (targets and pool candidates)")
    ap.add_argument("--targets", nargs="+", default=None, help="Subset of --subjects to evaluate (default: all)")
    ap.add_argument("--window_s", type=float, default=1.4)
    ap.add_argument("--max_epochs", type=int, default=None, help="Debug only: cap all epochs")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed (e.g. 42, 52, 62). Default: experiment.seed of the base config",
    )
    args = ap.parse_args()

    if args.data_dir is None:
        ap.error("--data_dir is required")
    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())
    base_cfg["data"]["data_directory"] = str(args.data_dir)
    base_cfg["data"]["models_main_directory"] = str(args.artifacts_dir)
    base_cfg["window"]["window_size_s"] = float(args.window_s)
    if args.win_and_feats is not None:
        base_cfg["paths"]["win_and_feats"] = args.win_and_feats
    if args.seed is not None:
        base_cfg["experiment"]["seed"] = args.seed
    configure_seed(base_cfg["experiment"]["seed"])
    if args.max_epochs is not None:
        model_cfg["model"]["kwargs"]["train_cfg"]["num_epochs"] = args.max_epochs
        FT_SETTINGS["num_ft_epochs"] = args.max_epochs

    for cond in args.conditions:
        SubjectScalingExperiment(
            base_cfg, model_cfg, cond, args.artifacts_dir, args.subjects, args.targets or args.subjects
        ).run()

    print("Aggregate with: python utils/III_results_analysis/IV_subject_scaling_analysis.py "
          f"--artifacts_dir {args.artifacts_dir}")


if __name__ == "__main__":
    main()
