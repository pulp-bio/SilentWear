#!/usr/bin/env python3


# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Generate windowed EMG and (optional) handcrafted features.

This is a thin wrapper around:
  utils/II_feature_extraction/win_feature_extraction_main.py

Goals:
1) Can be run standalone (default: all subjects, all conditions, window sweep 0.4..1.4s).
2) Can be called from other scripts (override data_dir / subjects / conditions / windows).

Typical usage (standalone, full sweep):
  python scripts/20_make_windows_and_features.py --data_dir ./data

Custom subset:
  python scripts/20_make_windows_and_features.py --data_dir ./data --subjects S01 S02 --conditions silent --windows_s 0.8 1.2 --manual_features false

Notes:
- This script writes a temporary YAML per run so tracked configs are not modified.
- It expects your dataset layout such that the extractor can find raw data under data_dir.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys
import yaml

# Ensure repo root on path (repo_root/scripts/this_file.py -> repo_root)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.general_utils import SubjectConfig  # type: ignore
from utils.II_feature_extraction.win_feature_extraction_main import (
    Global_Windower_and_Feature_Extractor,
)


DEFAULT_WINDOWS_S = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
DEFAULT_SUBJECTS = ["S01", "S02", "S03", "S04"]
DEFAULT_CONDITIONS = ["silent", "vocalized"]


def _parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Expected boolean string, got: {s!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config/create_windows.yaml",
        help="Base YAML config to use as template (will not be modified).",
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data"),
        help="Dataset root directory.",
    )
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="Subjects to process (default: all).",
    )
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Conditions to process (default: silent vocalized).",
    )
    ap.add_argument(
        "--windows_s",
        nargs="+",
        type=float,
        default=DEFAULT_WINDOWS_S,
        help="Window sizes in seconds (default: 0.4 0.6 0.8 1.0 1.2 1.4).",
    )
    ap.add_argument(
        "--manual_features",
        type=str,
        default=None,
        help="Override manual feature extraction: true/false. If omitted, uses YAML value.",
    )

    args = ap.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    cfg_template = yaml.safe_load(args.config.read_text())

    manual_features_override = None
    if args.manual_features is not None:
        manual_features_override = _parse_bool(args.manual_features)

    # Run sweep
    for window_s in args.windows_s:
        for sub in args.subjects:
            for cond in args.conditions:
                cfg = yaml.safe_load(args.config.read_text())  # fresh copy each run

                # Override minimal fields for this run
                cfg.setdefault("data", {})
                cfg["data"]["data_directory"] = str(args.data_dir)
                cfg["data"]["subject_id"] = str(sub)

                # Some pipelines keep condition in config; set if present/expected
                cfg["condition"] = str(cond)

                cfg.setdefault("window", {})
                cfg["window"]["window_size_s"] = float(window_s)

                if manual_features_override is not None:
                    cfg.setdefault("feature_extraction", {})
                    cfg["feature_extraction"]["manual_feature_extraction"] = bool(
                        manual_features_override
                    )

                print("\n" + "=" * 80)
                print(f"[WINDOWING] subject={sub} | condition={cond} | window_s={window_s}")
                print("=" * 80)

                with tempfile.TemporaryDirectory() as td:
                    tmp_cfg = Path(td) / "create_windows_tmp.yaml"
                    tmp_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

                    subject_cfg = SubjectConfig(tmp_cfg)
                    extractor = Global_Windower_and_Feature_Extractor(subject_cfg)
                    print("extractor initialized")
                    extractor.main()

                    print("done!")


if __name__ == "__main__":
    main()
