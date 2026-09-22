#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Gate: baseline on the published corpus
# ======================================
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "01_gate_baseline_published"

SUBJECTS_RUN="$SUBJECTS_4"
ROOT="$ARTIFACTS_BASE/01_gate_baseline_published"

BASE_PUB="config_thesis/thesis_base_words_published_w1400_rest.yaml"

# Cross-entropy
train_and_analyse "$BASE_PUB" \
                  config_thesis/models_configs/speechnet_baseline_words_ce.yaml \
                  "$DATA_WORDS_PUBLISHED" "$ROOT/ce" speechnet 1.4 w1400ms

# CTC on the un-resized backbone
train_and_analyse "$BASE_PUB" \
                  config_thesis/models_configs/speechnet_baseline_words_ctc.yaml \
                  "$DATA_WORDS_PUBLISHED" "$ROOT/ctc" speechnet 1.4 w1400ms

# The STFT backbone with a recurrent stage
train_and_analyse "$BASE_PUB" \
                  config_thesis/models_configs/words_stft_ce_bilstm.yaml \
                  "$DATA_WORDS_PUBLISHED" "$ROOT/stft_bilstm" speechnet 1.4 w1400ms

done_msg "01_gate_baseline_published" "$ROOT"
