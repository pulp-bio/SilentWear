# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
This file includes constants used during the data collection.
"""

from typing import Dict, Tuple

FS = 500

# ---------------------------------------------------------------------------
# Canonical dataset folder names
# ---------------------------------------------------------------------------


# These names must match the Hugging Face release layout documented in the README.
# Every module that needs to read/write these folders must import these constants.
RAW_DIRNAME = "raw"                                 # unfiltered .bio recordings (only present for self-collected data)
RAW_AND_FILTERED_DIRNAME = "raw_and_processed"      # filtered .h5 (output of data preparation)
WINS_AND_FEATURES_DIRNAME = "wins_and_features"     # windows + features (output of window extraction)
WINDOW_DIR_PREFIX = "WIN_"                          # per-window-size subfolder prefix, e.g. WIN_1400
SILENT_DIRNAME = "silent"
VOCALIZED_DIRNAME = "vocalized"

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------


ORIGINAL_LABELS_WORDS = {
    0: "rest",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
    5: "forward",
    6: "backward",
    7: "start",
    8: "stop",
    9: "advance",
    10: "reverse",
    11: "rotate",
    12: "halt",
    13: "begin",
    14: "grab",
    15: "place",
}

# ORIGINAL_LABELS_WORDS = {
#     0: "rest",
#     1: "up",
#     2: "down",
#     3: "left",
#     4: "right",
#     5: "forward",
#     6: "backward",
#     7: "start",
#     8: "stop"
# }

ORIGINAL_LABELS_SENTENCES = {
    0: "rest",
    1: "move forward",
    2: "turn left",
    3: "turn right",
    4: "stop the mission",
    5: "go to the center",
    6: "move inside",
    7: "go to the wall",
    8: "move backward",
    9: "proceed to the door",
    10: "back to initial position",
    11: "enter the room",
    12: "proceed two meters",
    13: "pick up the object",
    14: "put the object down",
    15: "turn on the light",
    16: "switch off the light",
    17: "turn off the volume",
    18: "stop right there",
    19: "start the task",
    20: "start the mission"
}


def get_active_labels(mode: str = "word") -> Dict[int, str]:
    """Returns the active labels based on the mode."""
    if mode == "sentence":
        return ORIGINAL_LABELS_SENTENCES.copy()
    return ORIGINAL_LABELS_WORDS.copy()


def build_label_maps(
    label_mode: str = "word", include_rest: bool = True
) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, int]]:
    """Build the training label maps from the active label set.

    Returns
    -------
    (train_label_map, train_to_orig, orig_to_train)
        train_label_map: {train_id: word | sentence}
        train_to_orig:   {train_id: original_id}
        orig_to_train:   {original_id: train_id}

    When include_rest is False, the rest class (original id 0) is removed and the
    remaining labels are re-indexed contiguously starting from 0.
    """
    original_map = get_active_labels(label_mode)

    if include_rest:
        train_label_map = original_map.copy()
        train_to_orig = {k: k for k in original_map}
        orig_to_train = {k: k for k in original_map}
        return train_label_map, train_to_orig, orig_to_train

    # Remove rest (original label 0) and re-index contiguously.
    filtered_items = [(k, v) for k, v in original_map.items() if k != 0]
    train_label_map = {new_k: text for new_k, (_, text) in enumerate(filtered_items)}
    train_to_orig = {new_k: orig_k for new_k, (orig_k, _) in enumerate(filtered_items)}
    orig_to_train = {orig_k: new_k for new_k, (orig_k, _) in enumerate(filtered_items)}
    return train_label_map, train_to_orig, orig_to_train