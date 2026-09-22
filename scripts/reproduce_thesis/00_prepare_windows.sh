#!/bin/bash
#
# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Windowed datasets
# =================
#
# Produces: the three windowed datasets every later script reads.
#
# Trigger-anchored words at 1.4 s, trigger-anchored sentences at 2.0 s, and the
# trigger-free sentence windows, which the detector of Section 3.2 anchors on
# the measured speech onset rather than on the cue.
#
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
start_log "00_prepare_windows"

$PYTHON reproduce_paper_scripts/20_make_windows_and_features.py \
    --config config_thesis/create_windows_words.yaml \
    --data_dir "$DATA_WORDS" --windows_s 1.4 --label_mode word

$PYTHON reproduce_paper_scripts/20_make_windows_and_features.py \
    --config config_thesis/create_windows_sentences.yaml \
    --data_dir "$DATA_SENTENCES" --windows_s 2.0 --label_mode sentence

$PYTHON reproduce_paper_scripts/20_make_windows_and_features.py \
    --config config_thesis/create_windows_sentences_onset.yaml \
    --data_dir "$DATA_SENTENCES" --windows_s 2.4 --label_mode sentence

done_msg "00_prepare_windows" "$DATA_WORDS, $DATA_SENTENCES"
