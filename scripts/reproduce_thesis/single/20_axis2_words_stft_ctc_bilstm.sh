#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 20 axis2 words stft ctc bilstm
# ==============================
#
# One unit of axis 2, the training objective. Runs both protocols and both speaking conditions on
# a single model configuration, so that units may be assigned to separate
# accelerators. Restrict further with EXPERIMENTS, CONDITIONS or SUBJECTS.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "20_axis2_words_stft_ctc_bilstm"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_words_w1400_rest.yaml \
                  config_thesis/models_configs/words_stft_ctc_bilstm_classification_greedy.yaml \
                  "$DATA_WORDS" "$ARTIFACTS_BASE/20_axis2_objective/words_stft_ctc_bilstm" \
                  speechnet 1.4 w1400ms

done_msg "20_axis2_words_stft_ctc_bilstm" "$ARTIFACTS_BASE/20_axis2_objective/words_stft_ctc_bilstm"
