#!/usr/bin/env python3
"""Generate windowed EMG and (optional) handcrafted features.

This is a thin wrapper around `utils/II_feature_extraction/win_feature_extraction_main.py`,
but it allows overriding paths and window size without editing repo files.

Example:
  python scripts/20_make_windows_and_features.py \
    --config config/open_release_create_windows.yaml \
    --data_dir ./data \
    --subject S01 \
    --window_s 1.6
"""

import argparse
from pathlib import Path
import yaml
import tempfile
import shutil
import sys

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.general_utils import SubjectConfig
from utils.II_feature_extraction.win_feature_extraction_main import Global_Windower_and_Feature_Extractor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=REPO_ROOT/"config/open_release_create_windows.yaml")
    ap.add_argument("--data_dir", type=Path, default=Path("./data"))
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--window_s", type=float, default=None)
    ap.add_argument("--manual_features", type=str, default=None, help="true/false to override")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    # override
    cfg["data"]["data_directory"] = str(args.data_dir)
    if args.subject is not None:
        cfg["data"]["subject_id"] = args.subject
    if args.window_s is not None:
        cfg["window"]["window_size_s"] = float(args.window_s)
    if args.manual_features is not None:
        cfg["feature_extraction"]["manual_feature_extraction"] = args.manual_features.lower() == "true"

    # write temp yaml so we don't mutate tracked configs
    with tempfile.TemporaryDirectory() as td:
        tmp_cfg = Path(td)/"create_windows.yaml"
        tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

        subject_cfg = SubjectConfig(tmp_cfg)
        extractor = Global_Windower_and_Feature_Extractor(subject_cfg)
        extractor.main()

if __name__ == "__main__":
    main()
