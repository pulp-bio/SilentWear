#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 60 ablation augmentation
# ========================
#
# Sweeps the sliding-window stride and the number of shifts per side against an un-augmented baseline. 
# Each point of the sweep is one windowed dataset.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "60_ablation_augmentation"

$PYTHON reproduce_paper_scripts/30_run_experiments.py \
    --base_config config_thesis/thesis_base_words_w1400_rest.yaml \
    --model_config config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
    --data_dir "$DATA_WORDS" \
    --window_config config_thesis/create_windows_words.yaml \
    --artifacts_dir "$ARTIFACTS_BASE/60_ablations/augmentation" \
    --experiment data_augmentation_ablation \
    --subjects $SUBJECTS_3 --conditions $CONDITIONS \
    --aug_windows_s 1.4 --stride_ms 10 20 50 100 --num_strides 2 5 10

done_msg "60_ablation_augmentation" "$ARTIFACTS_BASE/60_ablations/augmentation"
