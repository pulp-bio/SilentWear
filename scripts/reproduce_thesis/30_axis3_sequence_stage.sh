#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 3: the sequence stage
# ==========================
#
# The convolutional backbone alone, then BiLSTM against Transformer at equal
# parameter count, on all seven participants, under both readings of the same
# posteriors.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "30_axis3_sequence_stage"

# Restrict the step:
#   ARCHS="none"  only the backbone floor        (default: all three)
#   MEL=0         skip the mel cepstrum block    (default: 1, run it)
ARCHS="${ARCHS:-none bilstm transformer}"
MEL="${MEL:-1}"

SUBJECTS_RUN="$SUBJECTS"
BASE=config_thesis/thesis_base_sentences_w2000_rest.yaml

for ARCH in $ARCHS; do
    [ "$ARCH" = "transformer" ] && NAME=speechnet_transformer || NAME=speechnet
    for TASK in classification recognition; do
        MODEL="config_thesis/models_configs/sentences_stft_ctc_${ARCH}_${TASK}_greedy.yaml"
        ROOT="$ARTIFACTS_BASE/30_axis3_sequence_stage/${ARCH}_${TASK}"
        train_and_analyse "$BASE" "$MODEL" "$DATA_SENTENCES" "$ROOT" "$NAME" 2.0 w2000ms
    done
done

for ARCH in $([ "$MEL" = "1" ] && echo "bilstm transformer"); do
    [ "$ARCH" = "bilstm" ] && NAME=speechnet || NAME=speechnet_transformer
    for TASK in classification recognition; do
        MODEL="config_thesis/models_configs/sentences_mfcc_b15_q10_ctc_${ARCH}_${TASK}_greedy.yaml"
        ROOT="$ARTIFACTS_BASE/30_axis3_sequence_stage/mel_${ARCH}_${TASK}"
        train_and_analyse "$BASE" "$MODEL" "$DATA_SENTENCES" "$ROOT" "$NAME" 2.0 w2000ms
    done
done

done_msg "30_axis3_sequence_stage" "$ARTIFACTS_BASE/30_axis3_sequence_stage"
