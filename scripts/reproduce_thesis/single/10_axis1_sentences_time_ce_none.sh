#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# 10 axis1 sentences time ce none
# ===============================
#
# One unit of axis 1, the input domain. Runs both protocols and both speaking conditions on
# a single model configuration, so that units may be assigned to separate
# accelerators. Restrict further with EXPERIMENTS, CONDITIONS or SUBJECTS.
#
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
start_log "10_axis1_sentences_time_ce_none"

SUBJECTS_RUN="$SUBJECTS"

train_and_analyse config_thesis/thesis_base_sentences_w2000_rest.yaml \
                  config_thesis/models_configs/sentences_time_ce_none.yaml \
                  "$DATA_SENTENCES" "$ARTIFACTS_BASE/10_axis1_input_domain/sentences_time_ce_none" \
                  speechnet 2.0 w2000ms

done_msg "10_axis1_sentences_time_ce_none" "$ARTIFACTS_BASE/10_axis1_input_domain/sentences_time_ce_none"
