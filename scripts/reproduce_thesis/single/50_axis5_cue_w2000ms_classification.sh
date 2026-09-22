#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 50 axis5 cue w2000ms classification
# ===================================
#
# One unit of axis 5, the window anchor. Runs both protocols and both speaking conditions on
# a single model configuration, so that units may be assigned to separate
# accelerators. Restrict further with EXPERIMENTS, CONDITIONS or SUBJECTS.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "50_axis5_cue_w2000ms_classification"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_sentences_w2000_norest.yaml \
                  config_thesis/models_configs/sentences_stft_ctc_transformer_classification_greedy.yaml \
                  "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/cue_w2000ms_classification" \
                  speechnet_transformer 2.0 w2000ms

done_msg "50_axis5_cue_w2000ms_classification" "$ARTIFACTS_BASE/50_axis5_trigger_free/cue_w2000ms_classification"
