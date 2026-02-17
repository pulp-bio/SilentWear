#!/usr/bin/env python3
"""Run paper experiments (Global / Inter-Session) and save run folders into artifacts/.

This wrapper avoids editing `offline_experiments/*.py` and instead imports their trainer classes.

Outputs:
  artifacts/models/<experiment>/<subject>/<condition>/<model_name>/model_<k>/
    - cv_summary.csv
    - run_cfg.json
    - (optionally) checkpoints, confusion matrices, etc.

Example:
  python scripts/30_run_experiments.py \
    --base_config config/open_release_base_models_config.yaml \
    --model_config config/models_configs/speechnet_base_with_padding.yaml \
    --data_dir ./data \
    --artifacts_dir ./artifacts \
    --experiment global inter_session \
    --subjects S01 S02 S03 S04 \
    --conditions silent vocalized
"""

import argparse
from pathlib import Path
import yaml
import sys
from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.I_global_models import Global_Model_Trainer
from offline_experiments.II_inter_session_models import Inter_Session_Model_Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", type=Path, default=REPO_ROOT/"config/open_release_base_models_config.yaml")
    ap.add_argument("--model_config", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=Path("./data"))
    ap.add_argument("--artifacts_dir", type=Path, default=Path("./artifacts"))
    ap.add_argument("--experiment", nargs="+", choices=["global", "inter_session"], default=["global","inter_session"])
    ap.add_argument("--subjects", nargs="+", default=["S01","S02","S03","S04"])
    ap.add_argument("--conditions", nargs="+", default=["silent","vocalized"])
    args = ap.parse_args()

    base_cfg = yaml.safe_load(args.base_config.read_text())
    model_cfg = yaml.safe_load(args.model_config.read_text())

    # Override paths for open release
    base_cfg["data"]["data_directory"] = str(args.data_dir)
    base_cfg["data"]["models_main_directory"] = str(args.artifacts_dir)

    for sub in args.subjects:
        for cond in args.conditions:
            cfg_run = deepcopy(base_cfg)
            cfg_run["data"]["subject_id"] = sub
            cfg_run["condition"] = cond

            if "global" in args.experiment:
                print(f"\n=== GLOBAL | {sub} | {cond} ===")
                Global_Model_Trainer(base_config=cfg_run, model_config=model_cfg)

            if "inter_session" in args.experiment:
                print(f"\n=== INTER-SESSION | {sub} | {cond} ===")
                Inter_Session_Model_Trainer(base_config=cfg_run, model_config=model_cfg)

if __name__ == "__main__":
    main()
