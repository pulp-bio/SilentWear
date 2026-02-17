#!/usr/bin/env bash
set -euo pipefail

# Reproduce paper results end-to-end.
#
# Fast path (recommended):
#   1) download the Hugging Face dataset
#   2) set DATA_DIR to the dataset root folder
#   3) run this script
#
# It will:
#   - (optionally) regenerate windows/features if you set REGEN_WINDOWS=1
#   - run Global + Inter-Session experiments
#   - generate tables/figures from saved summaries
#
# Environment variables:
#   DATA_DIR          dataset root (default: ./data)
#   ARTIFACTS_DIR     output folder (default: ./artifacts)
#   MODEL_CONFIG      model yaml (default: config/models_configs/speechnet_base_with_padding.yaml)
#   REGEN_WINDOWS     if 1, rerun windowing/features (default: 0)
#   WINDOW_S          window size in seconds when REGEN_WINDOWS=1 (default: 1.6)

DATA_DIR="${DATA_DIR:-./data}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-./artifacts}"
MODEL_CONFIG="${MODEL_CONFIG:-config/models_configs/speechnet_base_with_padding.yaml}"
REGEN_WINDOWS="${REGEN_WINDOWS:-0}"
WINDOW_S="${WINDOW_S:-1.6}"

mkdir -p "${ARTIFACTS_DIR}"

if [ "${REGEN_WINDOWS}" = "1" ]; then
  echo "== Regenerating windows/features =="
  python scripts/20_make_windows_and_features.py \
    --config config/open_release_create_windows.yaml \
    --data_dir "${DATA_DIR}" \
    --window_s "${WINDOW_S}" \
    --manual_features false
fi

echo "== Running experiments =="
python scripts/30_run_experiments.py \
  --base_config config/open_release_base_models_config.yaml \
  --model_config "${MODEL_CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --artifacts_dir "${ARTIFACTS_DIR}" \
  --experiment global inter_session

echo "== Generating tables/figures =="
SILENTWEAR_ARTIFACTS_DIR="${ARTIFACTS_DIR}" bash scripts/40_make_results.sh

echo "Done. Outputs in:"
echo "  ${ARTIFACTS_DIR}/tables"
echo "  ${ARTIFACTS_DIR}/figures"
