#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 50 axis5 onset w2400ms recognition
# ==================================
#
# One unit of axis 5, the window anchor. Runs both protocols and both speaking conditions on
# a single model configuration, so that units may be assigned to separate
# accelerators. Restrict further with EXPERIMENTS, CONDITIONS or SUBJECTS.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "50_axis5_onset_w2400ms_recognition"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_sentences_onset_w2400_norest.yaml \
                  config_thesis/models_configs/sentences_stft_ctc_transformer_recognition_greedy.yaml \
                  "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/onset_w2400ms_recognition" \
                  speechnet_transformer 2.4 w2400ms

done_msg "50_axis5_onset_w2400ms_recognition" "$ARTIFACTS_BASE/50_axis5_trigger_free/onset_w2400ms_recognition"
