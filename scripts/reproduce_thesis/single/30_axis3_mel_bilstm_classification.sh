#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 30 axis3 mel bilstm classification
# ==================================
#
# One unit of axis 3, the EMG-fitted mel cepstrum. Runs both protocols and both speaking conditions on
# a single model configuration, so that units may be assigned to separate
# accelerators. Restrict further with EXPERIMENTS, CONDITIONS or SUBJECTS.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "30_axis3_mel_bilstm_classification"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_sentences_w2000_rest.yaml \
                  config_thesis/models_configs/sentences_mfcc_b15_q10_ctc_bilstm_classification_greedy.yaml \
                  "$DATA_SENTENCES" "$ARTIFACTS_BASE/30_axis3_sequence_stage/mel_bilstm_classification" \
                  speechnet 2.0 w2000ms

done_msg "30_axis3_mel_bilstm_classification" "$ARTIFACTS_BASE/30_axis3_sequence_stage/mel_bilstm_classification"
