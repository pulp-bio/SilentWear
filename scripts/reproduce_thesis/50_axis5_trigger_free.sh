#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 5: the window anchor
# =========================
#
# Cue-anchored windows against trigger-free ones, run on the single
# configuration with the Transformer on the STFT map under greedy decoding. 
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "50_axis5_trigger_free"

SUBJECTS_RUN="$SUBJECTS"

for TASK in classification recognition; do
    MODEL="config_thesis/models_configs/sentences_stft_ctc_transformer_${TASK}_greedy.yaml"

    train_and_analyse config_thesis/thesis_base_sentences_w2000_norest.yaml "$MODEL" \
        "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/cue_w2000ms_${TASK}" \
        speechnet_transformer 2.0 w2000ms

    train_and_analyse config_thesis/thesis_base_sentences_onset_w2000_norest.yaml "$MODEL" \
        "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/onset_w2000ms_${TASK}" \
        speechnet_transformer 2.0 w2000ms

    train_and_analyse config_thesis/thesis_base_sentences_onset_w2400_norest.yaml "$MODEL" \
        "$DATA_SENTENCES" "$ARTIFACTS_BASE/50_axis5_trigger_free/onset_w2400ms_${TASK}" \
        speechnet_transformer 2.4 w2400ms
done

done_msg "50_axis5_trigger_free" "$ARTIFACTS_BASE/50_axis5_trigger_free"
