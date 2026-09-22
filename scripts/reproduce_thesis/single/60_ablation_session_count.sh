#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 60 ablation session count
# =========================
#
# Retrains on the first 1 to 6 sessions of each participant. Not an axis of the chain: 
# it varies what the training split contains, not what the network is.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "60_ablation_session_count"

$PYTHON reproduce_paper_scripts/30_run_experiments.py \
    --base_config config_thesis/thesis_base_words_w1400_rest.yaml \
    --model_config config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
    --data_dir "$DATA_WORDS" \
    --artifacts_dir "$ARTIFACTS_BASE/60_ablations/session_count" \
    --experiment session_count_ablation \
    --subjects $SUBJECTS_3 --conditions $CONDITIONS \
    --session_windows_s 1.4 --min_sessions 1

done_msg "60_ablation_session_count" "$ARTIFACTS_BASE/60_ablations/session_count"
