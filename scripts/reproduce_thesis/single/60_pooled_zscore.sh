#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 60 pooled zscore
# ================
#
# Pooled multi-subject model, amplitude normalization: zscore. Not an axis of the chain.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "60_pooled_zscore"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_sentences_w2000_rest_zscore.yaml \
                  config_thesis/models_configs/sentences_stft_ctc_transformer_classification_greedy.yaml \
                  "$DATA_SENTENCES" "$ARTIFACTS_BASE/60_ablations/pooled_zscore" \
                  speechnet_transformer 2.0 w2000ms --pool_subjects

done_msg "60_pooled_zscore" "$ARTIFACTS_BASE/60_ablations/pooled_zscore"
