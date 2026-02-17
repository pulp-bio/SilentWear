#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   SILENTWEAR_ARTIFACTS_DIR=./artifacts bash scripts/40_make_results.sh
#
# This calls the analysis scripts that read per-run `cv_summary.csv` files and generate:
#   - artifacts/tables/*.csv
#   - artifacts/figures/*.svg

export SILENTWEAR_ARTIFACTS_DIR="${SILENTWEAR_ARTIFACTS_DIR:-./artifacts}"

python utils/III_results_analysis/I_global_intersession_analysis.py
python utils/III_results_analysis/III_ft_results.py || true
python utils/III_results_analysis/II_infotransrate.py || true
