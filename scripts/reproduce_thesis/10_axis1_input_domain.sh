#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 1: the input domain
# ========================
#
# The four domains are crossed with the two sequence stages on all seven 
# participants and on both label sets. Every run decodes greedily: the decoder
# is axis 4, and comparing input representations under a search that has not been chosen 
# would confound the two.
#
# The mel cepstrum is measured here at both parameterizations, the audio one at
# 64 filters and 40 coefficients and the one refitted to the EMG bandwidth at 15
# and 10, on the same participants and under the same objective as every other
# domain, so that the input representation question is closed by this script on one footing
# and never reopened later in the chain.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "10_axis1_input_domain"

SUBJECTS_RUN="$SUBJECTS"

for CORPUS in words sentences; do
    if [ "$CORPUS" = "words" ]; then
        BASE=config_thesis/thesis_base_words_w1400_rest.yaml
        DATA="$DATA_WORDS"; WIN=1.4; WID=w1400ms
    else
        BASE=config_thesis/thesis_base_sentences_w2000_rest.yaml
        DATA="$DATA_SENTENCES"; WIN=2.0; WID=w2000ms
    fi
    for DOMAIN in time stft mfcc_b64_q40 mfcc_b15_q10; do
        for SEQ in none bilstm; do
            MODEL="config_thesis/models_configs/${CORPUS}_${DOMAIN}_ce_${SEQ}.yaml"
            [ -f "$MODEL" ] || continue
            ROOT="$ARTIFACTS_BASE/10_axis1_input_domain/${CORPUS}_${DOMAIN}_ce_${SEQ}"
            train_and_analyse "$BASE" "$MODEL" "$DATA" "$ROOT" speechnet "$WIN" "$WID"
        done
    done
done

done_msg "10_axis1_input_domain" "$ARTIFACTS_BASE/10_axis1_input_domain"
