#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Ablation: sliding-window augmentation, augmented-size arm
# =========================================================
#
# 60_ablations.sh runs this sweep under `original_size`, which resamples the
# augmented pool back to the cardinality of the base split. This script runs 
# the same grid under `augmented_size`.
#
# Runtime: 12 sweep points x 6 session counts x 3 subjects x 2 conditions
#          x 2 protocols, plus the shared baseline (skipped here, 60 has it).
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "61_ablation_augmentation_augmented_size"

ROOT="$ARTIFACTS_BASE/61_ablation_augmentation_augmented_size"

$PYTHON reproduce_paper_scripts/30_run_experiments.py \
    --base_config config_thesis/thesis_base_words_w1400_rest.yaml \
    --model_config config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
    --data_dir "$DATA_WORDS" \
    --window_config config_thesis/create_windows_words.yaml \
    --artifacts_dir "$ROOT" \
    --experiment data_augmentation_ablation \
    --subjects $SUBJECTS_3 --conditions $CONDITIONS \
    --aug_windows_s 1.4 --stride_ms 10 20 50 100 --num_strides 2 5 10 \
    --train_mode augmented_size \
    --skip_baseline

done_msg "61_ablation_augmentation_augmented_size" "$ROOT"
