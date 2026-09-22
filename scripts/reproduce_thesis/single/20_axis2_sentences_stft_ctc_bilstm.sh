#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 20 axis2 sentences stft ctc bilstm
# ==================================
#
# One unit of axis 2, the training objective. Runs both protocols and both speaking conditions on
# a single model configuration, so that units may be assigned to separate
# accelerators. Restrict further with EXPERIMENTS, CONDITIONS or SUBJECTS.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "20_axis2_sentences_stft_ctc_bilstm"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_sentences_w2000_rest.yaml \
                  config_thesis/models_configs/sentences_stft_ctc_bilstm_classification_greedy.yaml \
                  "$DATA_SENTENCES" "$ARTIFACTS_BASE/20_axis2_objective/sentences_stft_ctc_bilstm" \
                  speechnet 2.0 w2000ms

done_msg "20_axis2_sentences_stft_ctc_bilstm" "$ARTIFACTS_BASE/20_axis2_objective/sentences_stft_ctc_bilstm"
