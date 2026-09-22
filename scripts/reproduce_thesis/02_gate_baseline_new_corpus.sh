#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Gate: baseline on the corpus of this thesis
# ===========================================
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "02_gate_baseline_new_corpus"

SUBJECTS_RUN="$SUBJECTS_3"
ROOT="$ARTIFACTS_BASE/02_gate_baseline_new_corpus"

# The published backbone, unchanged
train_and_analyse config_thesis/thesis_base_words_w1400_rest.yaml \
                  config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
                  "$DATA_WORDS" "$ROOT" speechnet 1.4 w1400ms

# The STFT backbone with a recurrent stage
train_and_analyse config_thesis/thesis_base_words_w1400_rest.yaml \
                  config_thesis/models_configs/words_stft_ce_bilstm.yaml \
                  "$DATA_WORDS" "$ROOT/stft_bilstm" speechnet 1.4 w1400ms

done_msg "02_gate_baseline_new_corpus" "$ROOT"
