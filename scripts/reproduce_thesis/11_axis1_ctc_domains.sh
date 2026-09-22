#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 1, completion: the CTC cells of the input-domain matrix
# ============================================================
#
# Time + CTC + BiLSTM, MFCC 64/40 + CTC + BiLSTM
# Runtime: 2 configurations x 7 subjects x 2 conditions x 2 protocols.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "11_axis1_ctc_domains"

ROOT="$ARTIFACTS_BASE/11_axis1_ctc_domains"
BASE="config_thesis/thesis_base_sentences_w2000_rest.yaml"

for CFG in sentences_time_ctc_bilstm_classification_greedy \
           sentences_mfcc_b64_q40_ctc_bilstm_classification_greedy; do
    train_and_analyse "$BASE" \
                      "config_thesis/models_configs/${CFG}.yaml" \
                      "$DATA_SENTENCES" "$ROOT/$CFG" speechnet 2.0 w2000ms
done

done_msg "11_axis1_ctc_domains" "$ROOT"
