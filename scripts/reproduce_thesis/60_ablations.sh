#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Ablations on the training data, and the pooled models
# =====================================================
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "60_ablations"

SUBJECTS_RUN="$SUBJECTS_3"

# Enrolment sessions: retrain on the first 1 to 6 sessions of each participant.
$PYTHON reproduce_paper_scripts/30_run_experiments.py \
    --base_config config_thesis/thesis_base_words_w1400_rest.yaml \
    --model_config config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
    --data_dir "$DATA_WORDS" \
    --artifacts_dir "$ARTIFACTS_BASE/60_ablations/session_count" \
    --experiment session_count_ablation \
    --subjects $SUBJECTS_3 --conditions $CONDITIONS \
    --session_windows_s 1.4 --min_sessions 1

# Sliding-window augmentation: one windowed dataset per point of the sweep.
# Set data_augmentation.stride_ms and num_strides in the windowing configuration
# before each point; see config_thesis/create_windows_words.yaml.
$PYTHON reproduce_paper_scripts/30_run_experiments.py \
    --base_config config_thesis/thesis_base_words_w1400_rest.yaml \
    --model_config config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
    --data_dir "$DATA_WORDS" \
    --window_config config_thesis/create_windows_words.yaml \
    --artifacts_dir "$ARTIFACTS_BASE/60_ablations/augmentation" \
    --experiment data_augmentation_ablation \
    --subjects $SUBJECTS_3 --conditions $CONDITIONS \
    --aug_windows_s 1.4 --stride_ms 10 20 50 100 --num_strides 2 5 10

# Pooled multi-subject models, with and without amplitude normalization.
SUBJECTS_RUN="$SUBJECTS"
for NORM in "" _zscore _minmax; do
    BASE="config_thesis/thesis_base_sentences_w2000_rest${NORM}.yaml"
    train_and_analyse "$BASE" \
        config_thesis/models_configs/sentences_stft_ctc_transformer_classification_greedy.yaml \
        "$DATA_SENTENCES" "$ARTIFACTS_BASE/60_ablations/pooled${NORM:-_none}" \
        speechnet_transformer 2.0 w2000ms --pool_subjects
done

done_msg "60_ablations" "$ARTIFACTS_BASE/60_ablations"
