#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Axis 4: the decoder, swept once per surviving architecture
# ==========================================================
#
# Because axis 3 carried both architectures forward, the sweep runs twice. It is
# not two experiments but one axis determined separately for each survivor, and the
# fact that the grid selects genuinely different operating points is itself the
# evidence that the two models shape their posteriors differently.
#
# Each model is trained once with the log-probability dump enabled, and the whole 
# grid is re-decoded offline from that saved outputs, so the acoustic model is held fixed 
# and the decoder is the only variable. The grid is scoped to one protocol at a 
# time so that the global and the inter-session saved outputs are never pooled.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "40_axis4_decoder_sweep"

SUBJECTS_RUN="$SUBJECTS"
BASE=config_thesis/thesis_base_sentences_w2000_rest.yaml

for ARCH in bilstm transformer; do
    [ "$ARCH" = "bilstm" ] && NAME=speechnet || NAME=speechnet_transformer
    for TASK in classification recognition; do
        MODEL="config_thesis/models_configs/sentences_stft_ctc_${ARCH}_${TASK}_sweep.yaml"
        ROOT="$ARTIFACTS_BASE/40_axis4_decoder_sweep/${ARCH}_${TASK}"
        for EXP in $EXPERIMENTS; do
            [ "$SKIP_TRAIN" != "1" ] && train "$BASE" "$MODEL" "$DATA_SENTENCES" "$ROOT" "$EXP" 2.0
            beam_sweep  "$ROOT" "$EXP"
            beam_tables "$ROOT" "$EXP" "$NAME" w2000ms
        done
    done
done

$PYTHON utils/III_results_analysis/aggregate_rest_sentence_results.py \
    --root "$ARTIFACTS_BASE/40_axis4_decoder_sweep" \
    --csv  "$ARTIFACTS_BASE/40_axis4_decoder_sweep/decoder_summary.csv" \
    || echo "  [warn] aggregation expects the run names of config_thesis"

done_msg "40_axis4_decoder_sweep" "$ARTIFACTS_BASE/40_axis4_decoder_sweep"
