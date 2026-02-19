#!/usr/bin/env python3

# # SPDX-FileCopyrightText: 2026 ETH Zurich
# # SPDX-License-Identifier: Apache-2.0

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
from typing import List, Optional
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.I_global_models import Global_Model_Trainer
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer
from offline_experiments.III_train_from_scratch import TrainFromScratch_Model_Trainer
from offline_experiments.IV_inter_session_with_ft import FineTuning_Model_Trainer


def _apply_open_release_overrides(base_cfg: dict, data_dir: Path, artifacts_dir: Path) -> dict:
    """Override all path-like entries for open-source execution."""
    base_cfg = deepcopy(base_cfg)
    base_cfg.setdefault("data", {})
    base_cfg["data"]["data_directory"] = str(data_dir)
    base_cfg["data"]["models_main_directory"] = str(artifacts_dir)
    return base_cfg


def _window_ms_from_cfg(cfg: dict) -> int:
    w_s = float(cfg["window"]["window_size_s"])
    return int(round(w_s * 1000))


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
        # single value -> just that one
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

    window_ms = _window_ms_from_cfg(cfg_run)
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

    if experiment == "inter_session_ft":
        if ft_cfg is None:
            raise ValueError("inter_session_ft requested but ft_cfg is None")
        ft_cfg_local = deepcopy(ft_cfg)
        ft_cfg_local["model_name_id"] = model_name_id
        print(f"\n=== INTER-SESSION + FT | {sub} | {cond} | {model_name_id} ===")
        trainer = FineTuning_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, ft_cfg=ft_cfg_local
        )
        # your fixed version: trainer.main() exists and returns Path
        if hasattr(trainer, "main"):
            trainer.main()
        return

    if experiment == "train_from_scratch":
        if tfs_cfg is None:
            raise ValueError("train_from_scratch requested but tfs_cfg is None")
        tfs_cfg_local = deepcopy(tfs_cfg)
        tfs_cfg_local["model_name_id"] = model_name_id
        print(f"\n=== TRAIN-FROM-SCRATCH | {sub} | {cond} | {model_name_id} ===")
        # your fixed TFS trainer runs full sweep in __init__
        trainer = TrainFromScratch_Model_Trainer(
            base_config=cfg_run, model_config=model_cfg, tfs_cfg=tfs_cfg_local
        )
        trainer.main()
        return

    raise ValueError(f"Unknown experiment: {experiment}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_config", type=Path, required=True)
    ap.add_argument("--model_config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--artifacts_dir", type=Path, default=Path("./artifacts"))

    ap.add_argument(
        "--experiment",
        nargs="+",
        choices=["global", "inter_session", "inter_session_ft", "train_from_scratch"],
        default=["inter_session"],
    )

    ap.add_argument("--ft_config", type=Path, default=None)
    ap.add_argument("--tfs_config", type=Path, default=None)

    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])

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

    args = ap.parse_args()

    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())
    base_cfg = _apply_open_release_overrides(base_cfg, args.data_dir, args.artifacts_dir)

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

    if "global" in args.experiment:
        for sub in args.subjects:
            for cond in args.conditions:
                base_cfg["cv"]["mode"] = "leave_one_batch_out"
                _run_one_subject_condition("global", base_cfg, model_cfg, sub, cond, {}, {})

    if "inter_session" in args.experiment:
        for w_s in inter_session_windows:
            for sub in args.subjects:
                for cond in args.conditions:
                    base_cfg_w = deepcopy(base_cfg)
                    base_cfg_w.setdefault("window", {})
                    base_cfg_w["window"]["window_size_s"] = float(w_s)
                    _run_one_subject_condition(
                        "inter_session", base_cfg_w, model_cfg, sub, cond, {}, {}
                    )

    if "train_from_scratch" in args.experiment:
        for sub in args.subjects:
            for cond in args.conditions:
                for w_s in args.tfs_windows_s:
                    base_cfg_w = deepcopy(base_cfg)
                    base_cfg_w.setdefault("window", {})
                    base_cfg_w["window"]["window_size_s"] = float(w_s)
                    _run_one_subject_condition(
                        "train_from_scratch", base_cfg_w, model_cfg, sub, cond, ft_cfg, tfs_cfg
                    )

    if "inter_session_ft" in args.experiment:
        for sub in args.subjects:
            for cond in args.conditions:
                for w_s in args.ft_windows_s:
                    base_cfg_w = deepcopy(base_cfg)
                    base_cfg_w.setdefault("window", {})
                    base_cfg_w["window"]["window_size_s"] = float(w_s)
                    _run_one_subject_condition(
                        "inter_session_ft", base_cfg_w, model_cfg, sub, cond, ft_cfg, tfs_cfg
                    )

    # for sub in args.subjects:
    #     for cond in args.conditions:

    #         # GLOBAL: uses window from base_cfg (user sets it in YAML)
    #         if "global" in args.experiment:
    #             base_cfg["cv"]["mode"] = "leave_one_batch_out"
    #             _run_one_subject_condition("global", base_cfg, model_cfg, sub, cond, ft_cfg, tfs_cfg)

    #         if "inter_session" in args.experiment:
    #             for w_s in inter_session_windows:
    #                 base_cfg_w = deepcopy(base_cfg)
    #                 base_cfg_w.setdefault("window", {})
    #                 base_cfg_w["window"]["window_size_s"] = float(w_s)
    #                 _run_one_subject_condition("inter_session", base_cfg_w, model_cfg, sub, cond, ft_cfg, tfs_cfg)

    #         # INTER-SESSION + FT: run configured FT windows
    #         if "inter_session_ft" in args.experiment:
    #             for w_s in args.ft_windows_s:
    #                 base_cfg_w = deepcopy(base_cfg)
    #                 base_cfg_w.setdefault("window", {})
    #                 base_cfg_w["window"]["window_size_s"] = float(w_s)
    #                 _run_one_subject_condition("inter_session_ft", base_cfg_w, model_cfg, sub, cond, ft_cfg, tfs_cfg)

    #         # TRAIN-FROM-SCRATCH: run configured windows
    #         if "train_from_scratch" in args.experiment:
    #             for w_s in args.tfs_windows_s:
    #                 base_cfg_w = deepcopy(base_cfg)
    #                 base_cfg_w.setdefault("window", {})
    #                 base_cfg_w["window"]["window_size_s"] = float(w_s)
    #                 _run_one_subject_condition("train_from_scratch", base_cfg_w, model_cfg, sub, cond, ft_cfg, tfs_cfg)


if __name__ == "__main__":
    main()
