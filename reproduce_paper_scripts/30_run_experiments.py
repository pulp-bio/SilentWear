#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Experiment Orchestrator for Paper Reproducibility
==================================================

This script runs the experiments reported in the Silent-Wear paper and
stores all outputs under the specified artifacts directory.

Supported experiment types
--------------------------
- global
- inter_session
- inter_session_ft
- train_from_scratch
- data_augmentation_ablation
- session_count_ablation

Reproducibility Guarantees
--------------------------
For deterministic experiment grouping and consistent result aggregation,
the script automatically assigns:

  model_name_id = f"w{window_ms}ms"

This identifier is injected consistently across:
- global
- inter_session
- inter_session_ft
- train_from_scratch

This ensures reproducible folder structure and stable experiment tracking.

Inter-Session Window Ablations
------------------------------
The argument `--inter_session_windows_s` supports three modes:

1) Two values → interpreted as range endpoints
   Example:
       --inter_session_windows_s 0.4 1.4

   Expands to:
       0.4, 0.6, 0.8, 1.0, 1.2, 1.4
   (step controlled by --window_step_s, default=0.2)

2) More than two values → treated as explicit list

3) No values → default paper sweep:
       0.4 .. 1.4 (step=0.2)

For Usage Example, please refer to the README.md
"""


from __future__ import annotations

import argparse
from pathlib import Path
import sys
from copy import deepcopy
from itertools import product
from typing import List, Optional, Tuple, Sequence
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.general_utils import window_ms_from_cfg
from offline_experiments.I_global_models import Global_Model_Trainer
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.III_train_from_scratch import TrainFromScratch_Model_Trainer
from offline_experiments.IV_inter_session_with_ft import FineTuning_Model_Trainer
from offline_experiments.V_sessions_count_ablation import Session_Count_Ablation_Trainer
from offline_experiments.VI_data_augmentation_ablation import Data_Augmentation_Ablation_Trainer
from offline_experiments.general_utils import discover_sessions


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_EXPERIMENTS = ["global", "inter_session", "inter_session_ft", "train_from_scratch", "data_augmentation_ablation", "session_count_ablation"]
DEFAULT_SUBJECTS = ["S01", "S02", "S03", "S04"]
DEFAULT_CONDITIONS = ["silent", "vocalized"]
AUGMENTATION_MODE_CHOICES = ["augmented_size", "original_size"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _apply_open_release_overrides(base_cfg: dict, data_dir: Path, artifacts_dir: Path) -> dict:
    """Override all path-like entries for open-source execution."""
    base_cfg = deepcopy(base_cfg)
    base_cfg.setdefault("data", {})
    base_cfg["data"]["data_directory"] = str(data_dir)
    base_cfg["data"]["models_main_directory"] = str(artifacts_dir)
    return base_cfg


def _set_model_name_id_everywhere(cfg: dict, model_name_id: str) -> dict:
    """
    Put model_name_id in multiple keys to maximize compatibility across trainers.
    """
    cfg = deepcopy(cfg)
    cfg.setdefault("experiment", {})
    cfg.setdefault("data", {})
    cfg["experiment"]["model_name_id"] = model_name_id
    cfg["data"]["model_name_id"] = model_name_id
    cfg["model_name_id"] = model_name_id
    return cfg


def _expand_windows_s(vals: List[float], step: float) -> List[float]:
    """
    If user passes two numbers: [start, end], expand start..end with 'step' (inclusive).
    If user passes >2 numbers: treat as explicit windows.
    If user passes []: return default paper sweep 0.4..1.4 with step 0.2
    """
    if len(vals) == 0:
        start, end = 0.4, 1.4
    elif len(vals) == 1:
        return [float(vals[0])]
    elif len(vals) == 2:
        start, end = float(vals[0]), float(vals[1])
    else:
        return [float(v) for v in vals]

    if end < start:
        start, end = end, start

    # build inclusive range with rounding to avoid float drift
    out = []
    x = start
    # guard against step=0
    if step <= 0:
        raise ValueError("--window_step_s must be > 0")
    while x <= end + 1e-9:
        out.append(round(x, 3))
        x += step

    # ensure end included (within tolerance)
    if abs(out[-1] - end) > 1e-6:
        out.append(round(end, 3))

    return out


# ---------------------------------------------------------------------------
# Experiment drivers
# ---------------------------------------------------------------------------


def _run_one_subject_condition(
    experiment: str,
    base_cfg: dict,
    model_cfg: dict,
    sub: str,
    cond: str,
    ft_cfg: Optional[dict],
    tfs_cfg: Optional[dict],
) -> None:
    cfg_run = deepcopy(base_cfg)
    cfg_run["data"]["subject_id"] = sub
    cfg_run["condition"] = cond

    window_ms = window_ms_from_cfg(cfg_run)
    model_name_id = f"w{window_ms}ms"
    cfg_run = _set_model_name_id_everywhere(cfg_run, model_name_id)

    if experiment == "global":
        print(f"\n=== GLOBAL | {sub} | {cond} | {model_name_id} ===")
        trainer = Global_Model_Trainer(base_config=cfg_run, model_config=model_cfg)
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "inter_session":
        print(f"\n=== INTER-SESSION | {sub} | {cond} | {model_name_id} ===")
        trainer = Inter_Session_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, experiment_subdir="inter_session"
        )
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "session_count_ablation":
        print(f"\n=== SESSION COUNT ABLATION | {sub} | {cond} | {model_name_id} ===")
        trainer = Session_Count_Ablation_Trainer(base_config=cfg_run, model_config=model_cfg)
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "data_augmentation_ablation":
        print(f"\n=== DATA AUGMENTATION ABLATION | {sub} | {cond} | {model_name_id} ===")
        trainer = Data_Augmentation_Ablation_Trainer(base_config=cfg_run, model_config=model_cfg)
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "inter_session_ft":
        if ft_cfg is None:
            raise ValueError("inter_session_ft requested but ft_cfg is None")
        ft_cfg_local = deepcopy(ft_cfg)
        ft_cfg_local["model_name_id"] = model_name_id
        print(f"\n=== INTER-SESSION + FT | {sub} | {cond} | {model_name_id} ===")
        trainer = FineTuning_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, ft_cfg=ft_cfg_local
        )
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "train_from_scratch":
        if tfs_cfg is None:
            raise ValueError("train_from_scratch requested but tfs_cfg is None")
        tfs_cfg_local = deepcopy(tfs_cfg)
        tfs_cfg_local["model_name_id"] = model_name_id
        print(f"\n=== TRAIN-FROM-SCRATCH | {sub} | {cond} | {model_name_id} ===")
        trainer = TrainFromScratch_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, tfs_cfg=tfs_cfg_local
        )
        trainer.main()
        return

    raise ValueError(f"Unknown experiment: {experiment}")


def run_all_subjects(
    experiment: str,
    base_cfg: dict,
    model_cfg: dict,
    subjects: List[str],
    cond: str,
    ft_cfg: Optional[dict],
    tfs_cfg: Optional[dict],
) -> None:
    """
    Run an experiment pooling all specified subjects together into a single dataset.
    Passing a Python list as subject_id sets all_subjects_models = True inside the trainers.
    """
    cfg_run = deepcopy(base_cfg)
    cfg_run["data"]["subject_id"] = subjects
    cfg_run["condition"] = cond

    window_ms = window_ms_from_cfg(cfg_run)
    model_name_id = f"w{window_ms}ms"
    cfg_run = _set_model_name_id_everywhere(cfg_run, model_name_id)

    if experiment == "global":
        print(f"\n=== GLOBAL | all_subjects (pooled) | {cond} | {model_name_id} ===")
        trainer = Global_Model_Trainer(base_config=cfg_run, model_config=model_cfg)
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "inter_session":
        print(f"\n=== INTER-SESSION | all_subjects (pooled) | {cond} | {model_name_id} ===")
        trainer = Inter_Session_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, experiment_subdir="inter_session"
        )
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "inter_session_ft":
        if ft_cfg is None:
            raise ValueError("inter_session_ft requested but ft_cfg is None")
        ft_cfg_local = deepcopy(ft_cfg)
        ft_cfg_local["model_name_id"] = model_name_id
        print(f"\n=== INTER-SESSION + FT | all_subjects (pooled) | {cond} | {model_name_id} ===")
        trainer = FineTuning_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, ft_cfg=ft_cfg_local
        )
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "train_from_scratch":
        if tfs_cfg is None:
            raise ValueError("train_from_scratch requested but tfs_cfg is None")
        tfs_cfg_local = deepcopy(tfs_cfg)
        tfs_cfg_local["model_name_id"] = model_name_id
        print(f"\n=== TRAIN-FROM-SCRATCH | all_subjects (pooled) | {cond} | {model_name_id} ===")
        trainer = TrainFromScratch_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, tfs_cfg=tfs_cfg_local
        )
        trainer.main()
        return

    raise ValueError(f"Experiment '{experiment}' does not support pooled mode via run_all_subjects.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_config", type=Path, required=True)
    ap.add_argument("--model_config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, default=Path("./artifacts"))

    ap.add_argument(
        "--experiment",
        nargs="+",
        choices=DEFAULT_EXPERIMENTS,
        default=["inter_session"],
    )

    ap.add_argument("--ft_config", type=Path, default=None)
    ap.add_argument("--tfs_config", type=Path, default=None)
    ap.add_argument("--window_config", type=Path, default=None, help="Window config for ablation")

    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)

    # Flag for running pooled models across all subjects
    ap.add_argument(
        "--pool_subjects",
        action="store_true",
        help="Pool all specified subjects together into a single dataset (all_subjects mode).",
    )

    # Inter-session ablation controls
    ap.add_argument(
        "--inter_session_windows_s",
        nargs="*",
        type=float,
        default=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        help="Either: <start end> or explicit list. Default: 0.4..1.4 step 0.2",
    )
    ap.add_argument("--window_step_s", type=float, default=0.2)

    # FT/TFS windows (paper defaults)
    ap.add_argument(
        "--ft_windows_s",
        nargs="*",
        type=float,
        default=[0.8, 1.4],
        help="Windows for inter_session_ft (default: 0.8 1.4)",
    )
    ap.add_argument(
        "--tfs_windows_s",
        nargs="*",
        type=float,
        default=[0.8, 1.4],
        help="Windows for train_from_scratch (default: 0.8 1.4)",
    )
    
    # Data augmentation ablation parameters
    ap.add_argument("--aug_experiments", nargs="+", choices=["global", "inter_session"], default=["global", "inter_session"])
    ap.add_argument("--aug_windows_s", nargs="*", type=float, default=[1.4])
    ap.add_argument("--stride_ms", nargs="*", type=int, default=[10])
    ap.add_argument("--num_strides", nargs="*", type=int, default=[2, 5, 10])
    ap.add_argument("--train_mode", choices=AUGMENTATION_MODE_CHOICES, default=None,
                    help="Optional override. If omitted, the value from window_config['data_augmentation']['train_mode'] is used.")
    ap.add_argument("--skip_baseline", action="store_true", help="Skip the baseline (no augmentation) run")
    
    # Session count ablation parameters
    ap.add_argument("--session_windows_s", nargs="*", type=float, default=[1.4])
    ap.add_argument("--min_sessions", type=int, default=1, help="Minimum number of sessions to start with")

    # Plotting flags
    ap.add_argument("--plot_loss", action="store_true", help="Save epoch-by-epoch training/val loss curves")
    ap.add_argument("--plot_scatter", action="store_true", help="Generate final scatter plots for ablation results")

    # Explainability flags (global / inter_session SpeechNet only)
    ap.add_argument("--explain_embeddings", action="store_true",
                    help="Extract layer embeddings and run UMAP/t-SNE/PCA + centroid-margin analysis per fold")
    ap.add_argument("--explain_methods", nargs="+", choices=["umap", "tsne", "pca"],
                    default=["umap", "tsne", "pca"], help="Dimensionality-reduction methods")
    ap.add_argument("--explain_layers", nargs="+", choices=["pre_bilstm", "pre_fc"],
                    default=["pre_bilstm", "pre_fc"], help="SpeechNet points at which to grab embeddings")

    args = ap.parse_args()

    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())
    base_cfg = _apply_open_release_overrides(base_cfg, args.data_dir, args.artifacts_dir)
    
    # Propagate the loss plotting flag to internal trainers
    base_cfg["plot_loss"] = args.plot_loss

    # Propagate explainability settings to internal trainers
    base_cfg.setdefault("experiment", {})["explainability"] = {
        "enabled": args.explain_embeddings,
        "methods": args.explain_methods,
        "layers": args.explain_layers,
    }

    ft_cfg = None
    if "inter_session_ft" in args.experiment:
        if args.ft_config is None:
            raise ValueError("inter_session_ft requested but --ft_config not provided")
        ft_cfg = yaml.safe_load(args.ft_config.read_text())

    tfs_cfg = None
    if "train_from_scratch" in args.experiment:
        if args.tfs_config is None:
            raise ValueError("train_from_scratch requested but --tfs_config not provided")
        tfs_cfg = yaml.safe_load(args.tfs_config.read_text())

    # Expand inter-session ablation windows
    inter_session_windows = _expand_windows_s(args.inter_session_windows_s, step=args.window_step_s)
    
    max_sessions_global = max(
        (len(discover_sessions(args.data_dir, subj, cond)) for subj in args.subjects for cond in args.conditions),
        default=0
    )
    
    if "session_count_ablation" in args.experiment:
        for n_sess in range(args.min_sessions, max_sessions_global + 1):
            for current_exp in ["global", "inter_session"]:
                
                if current_exp == "inter_session" and n_sess < 2:
                    continue
                
                for w_s in args.session_windows_s:
                    for sub in args.subjects:
                        for cond in args.conditions:
                            sessions = discover_sessions(args.data_dir, sub, cond)
                            if n_sess > len(sessions):
                                continue
                                
                            selected_sessions = sessions[:n_sess]

                            base_cfg_w = deepcopy(base_cfg)
                            base_cfg_w.setdefault("window", {})
                            base_cfg_w["window"]["window_size_s"] = float(w_s)

                            base_cfg_w.setdefault("experiment", {})
                            base_cfg_w["experiment"]["sessions_ablation"] = {
                                "num_sessions": n_sess,
                                "session_ids": selected_sessions,
                                "experiment_type": current_exp,
                            }
                            
                            _run_one_subject_condition(
                                "session_count_ablation", base_cfg_w, model_cfg, sub, cond, None, None
                            )

    if "data_augmentation_ablation" in args.experiment:
        if args.window_config is None:
            raise ValueError("data_augmentation_ablation requires --window_config to be set")
        window_cfg = yaml.safe_load(args.window_config.read_text())
        cli_train_mode = args.train_mode
        base_train_mode = base_cfg.get("experiment", {}).get("augmentation_train_mode")
        yaml_train_mode = window_cfg.get("data_augmentation", {}).get("train_mode")
        train_mode = str(cli_train_mode or base_train_mode or yaml_train_mode or "augmented_size").strip().lower()
        
        augmentation_combos: Sequence[Tuple[Optional[int], Optional[int]]] = list(product(args.stride_ms, args.num_strides))
        if not args.skip_baseline:
            augmentation_combos = [(None, None)] + augmentation_combos

        varies_stride = len(args.stride_ms) > 1
        varies_num = len(args.num_strides) > 1

        if varies_stride and not varies_num:
            ablation_folder_name = "data_augmentation_ablation_stride_dim"
        elif varies_num and not varies_stride:
            ablation_folder_name = "data_augmentation_ablation_num_strides"
        else:
            ablation_folder_name = "data_augmentation_ablation"
            
        # 1. Loop on augmentation combos (stride dimension and number of strides including baseline with no augmentation)
        for stride_ms, num_strides in augmentation_combos:
            if stride_ms is None or num_strides is None:
                data_augmentation = {"mode": "disabled"}
                run_label = "baseline"
            else:
                data_augmentation = {"mode": "sliding_window", "stride_ms": int(stride_ms), "num_strides": int(num_strides)}
                run_label = f"stride{stride_ms}_n{num_strides}"

            # 2. Loop on number of sessions for ablation (starting from min_sessions up to max available across all subjects/conditions)
            for n_sess in range(args.min_sessions, max_sessions_global + 1):
                
                # 3. Loop on Experiment / Window / Subj / Cond
                for current_exp in args.aug_experiments:
                    
                    if current_exp == "inter_session" and n_sess < 2:
                        continue

                    for w_s in args.aug_windows_s:
                        for sub in args.subjects:
                            for cond in args.conditions:
                                sessions = discover_sessions(args.data_dir, sub, cond)
                                if n_sess > len(sessions):
                                    continue
                                
                                selected_sessions = sessions[:n_sess]

                                base_cfg_w = deepcopy(base_cfg)
                                base_cfg_w.setdefault("window", {})
                                base_cfg_w["window"]["window_size_s"] = float(w_s)
                                base_cfg_w.setdefault("experiment", {})
                                
                                current_window_cfg = deepcopy(window_cfg)
                                current_window_cfg.setdefault("data", {})
                                current_window_cfg["data"]["subject_id"] = sub
                                base_cfg_w["experiment"]["window_config_template"] = current_window_cfg                               
                                
                                base_cfg_w["experiment"]["augmentation_train_mode"] = train_mode
                                base_cfg_w["experiment"]["augmentation_ablation"] = {
                                    "num_sessions": n_sess,
                                    "session_ids": selected_sessions,
                                    "experiment_type": current_exp,
                                    "data_augmentation": data_augmentation,
                                    "run_label": run_label,
                                    "ablation_folder_name": ablation_folder_name
                                }
                                
                                _run_one_subject_condition(
                                    "data_augmentation_ablation", base_cfg_w, model_cfg, sub, cond, None, None
                                )

    if "global" in args.experiment:
        if args.pool_subjects:
            for cond in args.conditions:
                run_all_subjects("global", base_cfg, model_cfg, args.subjects, cond, None, None)
        else:
            for sub in args.subjects:
                for cond in args.conditions:
                    _run_one_subject_condition("global", base_cfg, model_cfg, sub, cond, {}, {})

    if "inter_session" in args.experiment:
        for w_s in inter_session_windows:
            if args.pool_subjects:
                for cond in args.conditions:
                    base_cfg_w = deepcopy(base_cfg)
                    base_cfg_w.setdefault("window", {})
                    base_cfg_w["window"]["window_size_s"] = float(w_s)
                    run_all_subjects("inter_session", base_cfg_w, model_cfg, args.subjects, cond, None, None)
            else:
                for sub in args.subjects:
                    for cond in args.conditions:
                        sessions = discover_sessions(args.data_dir, sub, cond)
                        if len(sessions) < 2:
                            print(f"[SKIP] inter_session requires >=2 sessions. Found {len(sessions)} for {sub} {cond}.")
                            continue

                        base_cfg_w = deepcopy(base_cfg)
                        base_cfg_w.setdefault("window", {})
                        base_cfg_w["window"]["window_size_s"] = float(w_s)
                        _run_one_subject_condition(
                            "inter_session", base_cfg_w, model_cfg, sub, cond, {}, {}
                        )

    if "train_from_scratch" in args.experiment:
        for w_s in args.tfs_windows_s:
            if args.pool_subjects:
                for cond in args.conditions:
                    base_cfg_w = deepcopy(base_cfg)
                    base_cfg_w.setdefault("window", {})
                    base_cfg_w["window"]["window_size_s"] = float(w_s)
                    run_all_subjects("train_from_scratch", base_cfg_w, model_cfg, args.subjects, cond, None, tfs_cfg)
            else:
                for sub in args.subjects:
                    for cond in args.conditions:
                        base_cfg_w = deepcopy(base_cfg)
                        base_cfg_w.setdefault("window", {})
                        base_cfg_w["window"]["window_size_s"] = float(w_s)
                        _run_one_subject_condition(
                            "train_from_scratch", base_cfg_w, model_cfg, sub, cond, None, tfs_cfg
                        )

    if "inter_session_ft" in args.experiment:
        for w_s in args.ft_windows_s:
            if args.pool_subjects:
                for cond in args.conditions:
                    base_cfg_w = deepcopy(base_cfg)
                    base_cfg_w.setdefault("window", {})
                    base_cfg_w["window"]["window_size_s"] = float(w_s)
                    run_all_subjects("inter_session_ft", base_cfg_w, model_cfg, args.subjects, cond, ft_cfg, None)
            else:
                for sub in args.subjects:
                    for cond in args.conditions:
                        sessions = discover_sessions(args.data_dir, sub, cond)
                        if len(sessions) < 2:
                            print(f"[SKIP] inter_session_ft requires >=2 sessions. Found {len(sessions)} for {sub} {cond}.")
                            continue

                        base_cfg_w = deepcopy(base_cfg)
                        base_cfg_w.setdefault("window", {})
                        base_cfg_w["window"]["window_size_s"] = float(w_s)
                        _run_one_subject_condition(
                            "inter_session_ft", base_cfg_w, model_cfg, sub, cond, ft_cfg, None
                        )

if __name__ == "__main__":
    main()
