#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 2: the training objective
# ==============================
#
# CE against CTC on the input representation axis 1 selected.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "20_axis2_objective"

SUBJECTS_RUN="$SUBJECTS"

for CORPUS in words sentences; do
    if [ "$CORPUS" = "words" ]; then
        BASE=config_thesis/thesis_base_words_w1400_rest.yaml
        DATA="$DATA_WORDS"; WIN=1.4; WID=w1400ms
    else
        BASE=config_thesis/thesis_base_sentences_w2000_rest.yaml
        DATA="$DATA_SENTENCES"; WIN=2.0; WID=w2000ms
    fi
    MODEL="config_thesis/models_configs/${CORPUS}_stft_ctc_bilstm_classification_greedy.yaml"
    ROOT="$ARTIFACTS_BASE/20_axis2_objective/${CORPUS}_stft_ctc_bilstm"
    train_and_analyse "$BASE" "$MODEL" "$DATA" "$ROOT" speechnet "$WIN" "$WID"
done

done_msg "20_axis2_objective" "$ARTIFACTS_BASE/20_axis2_objective"
