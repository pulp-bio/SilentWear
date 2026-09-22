#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 40 axis4 bilstm recognition sweep
# ==============================
#
# One unit of axis 4, the decoder. Trains once with the log-probability dump
# enabled, then re-decodes the whole grid offline from that saving.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "40_axis4_bilstm_recognition_sweep"

SUBJECTS_RUN="$SUBJECTS"
BASE=config_thesis/thesis_base_sentences_w2000_rest.yaml
MODEL=config_thesis/models_configs/sentences_stft_ctc_bilstm_recognition_sweep.yaml
ROOT="$ARTIFACTS_BASE/40_axis4_decoder_sweep/bilstm_recognition"

for EXP in $EXPERIMENTS; do
    [ "$SKIP_TRAIN" != "1" ] && train "$BASE" "$MODEL" "$DATA_SENTENCES" "$ROOT" "$EXP" 2.0
    beam_sweep  "$ROOT" "$EXP"
    beam_tables "$ROOT" "$EXP" speechnet w2000ms
done

done_msg "40_axis4_bilstm_recognition_sweep" "$ROOT"
