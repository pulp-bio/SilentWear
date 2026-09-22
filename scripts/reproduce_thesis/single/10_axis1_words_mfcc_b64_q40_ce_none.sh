#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 10 axis1 words mfcc b64 q40 ce none
# ===================================
#
# One unit of axis 1, the input domain. Runs both protocols and both speaking conditions on
# a single model configuration, so that units may be assigned to separate
# accelerators. Restrict further with EXPERIMENTS, CONDITIONS or SUBJECTS.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "10_axis1_words_mfcc_b64_q40_ce_none"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_words_w1400_rest.yaml \
                  config_thesis/models_configs/words_mfcc_b64_q40_ce_none.yaml \
                  "$DATA_WORDS" "$ARTIFACTS_BASE/10_axis1_input_domain/words_mfcc_b64_q40_ce_none" \
                  speechnet 1.4 w1400ms

done_msg "10_axis1_words_mfcc_b64_q40_ce_none" "$ARTIFACTS_BASE/10_axis1_input_domain/words_mfcc_b64_q40_ce_none"
