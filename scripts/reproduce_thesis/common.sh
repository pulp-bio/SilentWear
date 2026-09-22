#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Shared settings and helpers for the thesis reproduction scripts.
# ================================================================
#
# Every configuration referenced here lives under config_thesis/, which is the
# published configuration folder of the thesis; see config_thesis/README.md.
#
# Environment overrides (all optional):
#   SUBJECTS         participant list                  (default: S01..S07)
#   CONDITIONS       speaking conditions               (default: silent vocalized)
#   EXPERIMENTS      protocols to run                  (default: global inter_session)
#   ARTIFACTS_BASE   root under which artefacts land   (default: artifacts_thesis)
#   JOBS             workers for the offline sweeps    (default: 8)
#   SKIP_TRAIN       set to 1 to reuse existing runs   (default: 0)
#   PYTHON           interpreter                       (default: ./venv/bin/python)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-./venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------


DATA_SENTENCES="${DATA_SENTENCES:-data_sentences}"
DATA_WORDS="${DATA_WORDS:-data_words}"
DATA_WORDS_PUBLISHED="${DATA_WORDS_PUBLISHED:-/baltic/users/ml_datasets/iis_bio_internal_datasets/2026_spacone_speech_classification_hmi}"

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


SUBJECTS="${SUBJECTS:-S01 S02 S03 S04 S05 S06 S07}"
CONDITIONS="${CONDITIONS:-silent vocalized}"
EXPERIMENTS="${EXPERIMENTS:-global inter_session}"
ARTIFACTS_BASE="${ARTIFACTS_BASE:-artifacts_thesis}"
JOBS="${JOBS:-8}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

SUBJECTS_3="${SUBJECTS_3:-S01 S03 S04}"
SUBJECTS_4="${SUBJECTS_4:-S01 S02 S03 S04}"

BEAM_WIDTHS="${BEAM_WIDTHS:-1 5 10 25}"
TEMPERATURES="${TEMPERATURES:-1.0 1.3 1.6 2.0}"
BLANK_PENALTIES="${BLANK_PENALTIES:-0.0 1.0 2.0 4.0}"
LENGTH_BONUSES="${LENGTH_BONUSES:-0.0 0.5 1.0 2.0}"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


start_log() {
    mkdir -p "$ARTIFACTS_BASE/logs"
    local log="$ARTIFACTS_BASE/logs/$1_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$log") 2>&1
    echo "====================================================================="
    echo " $1"
    echo " log: $log"
    echo "====================================================================="
}

banner() {
    echo ""
    echo "--- $* ---"
}

# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------


# train  <base_cfg> <model_cfg> <data_dir> <artifacts_root> <experiment>
#        <window_id_seconds> [extra args...]
#
train() {
    local base_cfg="$1" model_cfg="$2" data_dir="$3" root="$4" exp="$5" win_s="$6"
    shift 6
    local extra=()
    [ "$exp" = "inter_session" ] && extra+=(--inter_session_windows_s "$win_s")
    "$PYTHON" reproduce_paper_scripts/30_run_experiments.py \
        --base_config "$base_cfg" \
        --model_config "$model_cfg" \
        --data_dir "$data_dir" \
        --artifacts_dir "$root" \
        --experiment "$exp" \
        --subjects $SUBJECTS_RUN \
        --conditions $CONDITIONS \
        "${extra[@]}" "$@"
}

# analyse  <artifacts_root> <experiment> <model_name> <model_name_id> [extra...]
analyse() {
    local root="$1" exp="$2" model_name="$3" win_id="$4"
    shift 4
    "$PYTHON" utils/III_results_analysis/I_global_intersession_analysis.py \
        --artifacts_dir "$root" \
        --experiment "$exp" \
        --model_name "$model_name" \
        --model_name_id "$win_id" \
        --subjects $SUBJECTS_RUN \
        --conditions $CONDITIONS \
        --model_run model_1 \
        "$@"
}

# train_and_analyse  <base_cfg> <model_cfg> <data_dir> <root> <model_name>
#                    <window_seconds> <window_id> [pool_flag]
train_and_analyse() {
    local base_cfg="$1" model_cfg="$2" data_dir="$3" root="$4"
    local model_name="$5" win_s="$6" win_id="$7"
    shift 7
    SUBJECTS_RUN="${SUBJECTS_RUN:-$SUBJECTS}"
    for exp in $EXPERIMENTS; do
        if [ "$SKIP_TRAIN" != "1" ]; then
            banner "train | $exp | $model_name | $root"
            train "$base_cfg" "$model_cfg" "$data_dir" "$root" "$exp" "$win_s" \
                  --plot_loss "$@"
        fi
        banner "tables | $exp | $model_name | $root"
        analyse "$root" "$exp" "$model_name" "$win_id" "$@" \
            || echo "  [warn] analysis failed for $exp on $root"
    done
}

# beam_sweep  <artifacts_root> <experiment>
beam_sweep() {
    local root="$1" exp="$2"
    banner "offline beam sweep | $exp | $root"
    "$PYTHON" offline_experiments/VII_beam_sweep.py \
        --dumps "$root/models/$exp" \
        --out   "$root/beam_sweep_$exp.csv" \
        --beam_widths $BEAM_WIDTHS \
        --temperatures $TEMPERATURES \
        --blank_penalties $BLANK_PENALTIES \
        --length_bonuses $LENGTH_BONUSES \
        --jobs "$JOBS"
}

# beam_tables  <artifacts_root> <experiment> <model_name> <window_id> [extra...]
beam_tables() {
    local root="$1" exp="$2" model_name="$3" win_id="$4"
    shift 4
    SUBJECTS_RUN="${SUBJECTS_RUN:-$SUBJECTS}"
    banner "beam tables | $exp | $root"
    "$PYTHON" utils/III_results_analysis/VII_beam_sweep_tables.py \
        --artifacts_dir "$root" \
        --experiment "$exp" \
        --model_name "$model_name" \
        --model_name_id "$win_id" \
        --model_run model_1 \
        --subjects $SUBJECTS_RUN \
        --conditions $CONDITIONS \
        --sweep_csv "$root/beam_sweep_$exp.csv" \
        --jobs "$JOBS" \
        --dump_predictions "$@" \
        || echo "  [warn] beam tables failed for $exp on $root"
}

done_msg() {
    echo ""
    echo "DONE: $1"
    echo "  artefacts : $2"
}
