# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Text transformation utilities for CTC lexicon/token conversion.
"""

from __future__ import annotations
from typing import Iterable, Optional
import jiwer
from unidecode import unidecode


_TEXT_NORMALIZER = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
ENGLISH_ALPHABET = list("abcdefghijklmnopqrstuvwxyz ")


class CTCTextTransform:
    """Utility class for text transformation, handling character-level tokenization and cleaning."""

    def __init__(
        self,
        vocab_texts: Iterable[str],
        blank_id: int = 0,
        alphabet: Optional[Iterable[str]] = None):
        self.blank_id = int(blank_id)
        self.transformation = _TEXT_NORMALIZER

        if alphabet is not None:
            chars = list(dict.fromkeys(alphabet))
        else:
            seen = set()
            chars = []
            for text in vocab_texts:
                clean_text = CTCTextTransform.clean_text(text)
                for ch in clean_text:
                    if ch not in seen:
                        seen.add(ch)
                        chars.append(ch)

        self.chars = chars
        self.char_to_int = {ch: idx + 1 for idx, ch in enumerate(self.chars)}
        self.int_to_char = {idx + 1: ch for idx, ch in enumerate(self.chars)}

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans text by normalizing punctuation, converting to lowercase, and removing accents."""
        text = unidecode(str(text))
        text = text.replace("-", " ")
        text = text.replace(":", " ")
        result = _TEXT_NORMALIZER(text)
        text = result if isinstance(result, str) else result[0]
        return text.strip()

    def text_to_int(self, text: str) -> list[int]:
        """Converts text to a list of integer token IDs based on the character vocabulary."""
        clean = self.clean_text(text)
        return [self.char_to_int[c] for c in clean if c in self.char_to_int]

    def int_to_text(self, ints: list[int]) -> str:
        """Converts a list of integer token IDs back to text using the character vocabulary."""
        return "".join(self.int_to_char.get(int(i), "") for i in ints)
