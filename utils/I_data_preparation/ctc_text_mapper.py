# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Text mapper for CTC labels and token IDs.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import editdistance
import torch

from utils.I_data_preparation.text_transform import CTCTextTransform, ENGLISH_ALPHABET

DEFAULT_BLANK_ID = 0


class CTCTextMapper(CTCTextTransform):
    """Map class labels to CTC token targets and decode token IDs back to texts."""

    def __init__(
        self,
        lexicon_path: str | None = None,
        lexicon_texts: List[str] | None = None,
        train_label_map: Dict[int, str] | None = None,
        blank_id: int = DEFAULT_BLANK_ID,
        use_full_alphabet: bool = False,
    ):
        self.blank_id = int(blank_id)
        self.train_label_map = train_label_map or {}

        self.label_to_text_map = {
            int(k): self.clean_text(v) for k, v in self.train_label_map.items()
        }
        self.text_to_label_map = {
            text: label for label, text in self.label_to_text_map.items()
        }

        self.lexicon_texts = []
        if lexicon_path:
            if not os.path.exists(lexicon_path):
                raise FileNotFoundError(f"CTC lexicon_path does not exist: {lexicon_path}.")
            with open(lexicon_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    phrase_parts = []
                    for part in parts:
                        if len(part) == 1 and part != "|":
                            break
                        phrase_parts.append(part)
                    
                    phrase = " ".join(phrase_parts)
                    self.lexicon_texts.append(self.clean_text(phrase))
        elif lexicon_texts:
            self.lexicon_texts = [self.clean_text(w) for w in lexicon_texts]
        else:
            raise ValueError("CTCTextMapper requires either lexicon_path or lexicon_texts.")

        self.lexicon_texts = list(dict.fromkeys(self.lexicon_texts))

        # If training labels are provided, keep only lexicon texts that belong to the active label set.
        if self.text_to_label_map:
            active_texts = set(self.text_to_label_map.keys())
            self.lexicon_texts = [text for text in self.lexicon_texts if text in active_texts]

            missing_active_texts = sorted(active_texts.difference(self.lexicon_texts))
            if missing_active_texts:
                raise ValueError(f"CTC lexicon is missing active label texts: {missing_active_texts}.")

        if not self.lexicon_texts:
            raise ValueError("CTC lexicon is empty after loading.")

        self.word_vocabulary = tuple(
            sorted({word for text in self.lexicon_texts for word in text.split()})
        )
        
        super().__init__(
            vocab_texts=self.lexicon_texts,
            blank_id=self.blank_id,
            alphabet=ENGLISH_ALPHABET if use_full_alphabet else None,
        )

    def label_int_to_texts(self, labels: torch.Tensor) -> List[str]:
        """Convert label IDs to texts using label_to_text_map."""
        return [
            self.label_to_text_map.get(int(label), str(int(label)))
            for label in labels.detach().cpu().tolist()
        ]

    def ctc_targets_from_label_int(
        self, targets: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode target labels into CTC character token stream and per-sample lengths."""
        target_tokens, target_lengths = [], []
        for text in self.label_int_to_texts(targets):
            encoded = self.text_to_int(text)
            if not encoded:
                raise ValueError(f"CTC target text '{text}' produced an empty token sequence.")
            target_tokens.extend(encoded)
            target_lengths.append(len(encoded))

        return (
            torch.tensor(target_tokens, dtype=torch.long, device=device),
            torch.tensor(target_lengths, dtype=torch.long, device=device),
        )

    def token_int_to_texts(self, pred_ids: torch.Tensor) -> List[str]:
        """Decode predicted token IDs to texts with CTC collapse (exact decode)."""
        if pred_ids.ndim == 1:
            pred_ids = pred_ids.unsqueeze(0)

        texts = []
        for seq in pred_ids.detach().cpu().tolist():
            collapsed, prev = [], None
            for token_id in seq:
                if token_id == self.blank_id:
                    prev = token_id
                    continue
                if token_id != prev:
                    collapsed.append(token_id)
                prev = token_id

            texts.append(self.clean_text(self.int_to_text(collapsed)))

        return texts

    def _closest_known_text(self, text: str) -> str | None:
        """Find nearest train label text by edit distance for lexicon-constrained decoding."""
        if not text or not self.text_to_label_map:
            return None
        return min(self.text_to_label_map, key=lambda candidate: editdistance.eval(text, candidate))

    def texts_to_label_int(
        self, texts: List[str], allow_nearest: bool = True
    ) -> Tuple[List[int], float]:
        """Map decoded texts to class IDs, with optional nearest-text fallback."""
        if not texts:
            return [], 0.0

        preds = []
        unknown = 0

        for raw_text in texts:
            text = self.clean_text(raw_text)

            if text in self.text_to_label_map:
                preds.append(int(self.text_to_label_map[text]))
                continue

            if text == "":
                preds.append(-1)
                unknown += 1
                continue

            if allow_nearest:
                nearest = self._closest_known_text(text)
                if nearest is not None:
                    preds.append(int(self.text_to_label_map[nearest]))
                    continue

            preds.append(-1)
            unknown += 1

        return preds, unknown / float(len(texts))