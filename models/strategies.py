# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Defines task-specific strategies for computing loss and making predictions, such as CrossEntropy and CTC.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List

from models.ctc_decoding import ctc_prefix_beam_search


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------


class TaskStrategy(ABC):

    @abstractmethod
    def compute_loss(self, outputs, targets, device: torch.device) -> torch.Tensor:
        pass

    def backward(self, loss: torch.Tensor) -> None:
        """Run one backward pass for the current strategy."""
        loss.backward()

    @staticmethod
    def _extract_logits(outputs) -> torch.Tensor:
        """Extracts logits from model outputs, handling both tensors and tuple/list outputs."""
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    @staticmethod
    def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
        """Greedy decode: pick the maximum-probability class at each position."""
        return torch.argmax(logits, dim=-1)

    @abstractmethod
    def predict_labels(self, outputs, use_score_fallback: bool = True) -> np.ndarray:
        """Return predicted class labels as a numpy array.

        use_score_fallback: strategies with expensive decoding steps 
        skip them when False and may return -1 for samples they cannot map.
        """
        pass


# ---------------------------------------------------------------------------
# CE
# ---------------------------------------------------------------------------


class CrossEntropyStrategy(TaskStrategy):
    """Standard cross-entropy strategy for classification tasks."""

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def _prepare_logits(self, outputs) -> torch.Tensor:
        logits = self._extract_logits(outputs)
        if logits.ndim == 3:
            return logits.mean(dim=1)
        if logits.ndim != 2:
            raise ValueError(f"Unsupported logits shape for CE: {tuple(logits.shape)}")
        return logits

    def compute_loss(self, outputs, targets, device: torch.device) -> torch.Tensor:
        return self.criterion(self._prepare_logits(outputs), targets.long())

    def predict_labels(self, outputs, use_score_fallback: bool = True) -> np.ndarray:
        return self.greedy_decode(self._prepare_logits(outputs)).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# CTC
# ---------------------------------------------------------------------------


class CTCStrategy(TaskStrategy):
    """CTC strategy for sequence-to-sequence tasks.

    Works with any label_mode ('word' or 'sentence'): the text_mapper
    receives the active train_label_map and handles tokenization accordingly.

    Character decoding (before the lexicon/label mapping) is selected by
    `decode_strategy`:
      - "greedy": best-path argmax + CTC collapse.
      - "beam":   CTC prefix beam search of width `beam_width`. 
    Beam search only runs on the final test decode (use_score_fallback=True).
    """

    def __init__(
        self,
        text_mapper,
        allow_nearest: bool = True,
        decode_strategy: str = "greedy",
        beam_width: int = 10,
        beam_temperature: float = 1.0,
        beam_blank_penalty: float = 0.0,
        beam_length_bonus: float = 0.0,
        label_smoothing: float = 0.0,
        lexicon_decision: str = "nearest",
    ):
        self.text_mapper = text_mapper
        self.criterion = nn.CTCLoss(blank=self.text_mapper.blank_id, zero_infinity=True)
        self.allow_nearest = allow_nearest
        self.decode_strategy = str(decode_strategy).lower().strip()
        if self.decode_strategy not in ("greedy", "beam"):
            raise ValueError(
                f"decode_strategy must be 'greedy' or 'beam', got '{decode_strategy}'."
            )
        self.beam_width = int(beam_width)
        self.beam_temperature = float(beam_temperature)
        self.beam_blank_penalty = float(beam_blank_penalty)
        self.beam_length_bonus = float(beam_length_bonus)
        self.lexicon_decision = str(lexicon_decision).lower().strip()
        if self.lexicon_decision not in ("nearest", "score"):
            raise ValueError(
                f"lexicon_decision must be 'nearest' or 'score', got '{lexicon_decision}'."
            )
        self.label_smoothing = float(label_smoothing)
        self._label_token_map = {
            int(label): self.text_mapper.text_to_int(text)
            for label, text in self.text_mapper.label_to_text_map.items()
        }
        self._ctc_len_warned = False

    @property
    def word_vocabulary(self) -> tuple:
        """Closed word vocabulary of the task: the words of all the lexicon texts.

        Passed to `compute_wer_metrics` so the vocabulary-constrained WER snaps
        each hypothesis word onto the lexicon's word set, independently of which
        utterances a given evaluation split contains.
        """
        return self.text_mapper.word_vocabulary

    def _warn_if_targets_too_long(self, target_lengths: torch.Tensor, time_steps: int) -> None:
        """Surface samples that CTC cannot align (target longer than input frames).

        `nn.CTCLoss(zero_infinity=True)` returns 0 loss/gradient for any
        sample whose target length exceeds the number of input frames T'.
        """
        if self._ctc_len_warned:
            return
        too_long = int((target_lengths > time_steps).sum().item())
        if too_long > 0:
            self._ctc_len_warned = True
            max_target = int(target_lengths.max().item())
            print(
                f"[CTC WARNING] {too_long} sample(s) have target length > input frames "
                f"(T'={time_steps}, longest target={max_target}). Increase the window "
                f"size or reduce temporal pooling so T' >= longest target."
            )

    def compute_loss(self, outputs, targets, device: torch.device) -> torch.Tensor:
        """Expects logits of shape (B, T, C)."""
        logits = self._extract_logits(outputs)
        bsz, time_steps, _ = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)          # (B, T, C)
        pred = log_probs.transpose(0, 1)                   # (T, B, C) for nn.CTCLoss
        input_lengths = torch.full((bsz,), fill_value=time_steps, dtype=torch.long, device=device)
        target_tokens, target_lengths = self.text_mapper.ctc_targets_from_label_int(targets, device)
        self._warn_if_targets_too_long(target_lengths, time_steps)
        loss = self.criterion(pred, target_tokens, input_lengths, target_lengths)

        # Optional entropy regularisation for CTC.
        if self.label_smoothing > 0.0:
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
            loss = loss - self.label_smoothing * entropy
        return loss

    def _best_label_from_ctc_scores(self, sample_logits: torch.Tensor) -> int:
        """Pick class label (text) with minimum CTC loss."""
        device = sample_logits.device
        time_steps = int(sample_logits.shape[0])

        log_probs = F.log_softmax(sample_logits, dim=-1).unsqueeze(1)  # (T, 1, C)
        input_lengths = torch.tensor([time_steps], dtype=torch.long, device=device) # Number of time steps for CTC input
        best_label = -1
        best_loss = float("inf")

        for label, token_ids in self._label_token_map.items():
            if not token_ids:
                continue

            target_tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)

            loss = self.criterion(log_probs, target_tokens, input_lengths, target_lengths)
            loss_value = float(loss.detach().cpu().item())

            if loss_value < best_loss:
                best_loss = loss_value
                best_label = int(label)

        if best_label < 0:
            raise ValueError("CTC scores decoding failed: no valid class tokenization found.")

        return best_label

    def _labels_from_ctc_scores_batch(self, logits: torch.Tensor) -> np.ndarray:
        """Exact closed-set decision for a whole batch: argmax_k P_CTC(s_k | X).

        Every lexicon text s_k is scored with the CTC forward algorithm (which
        marginalises over all alignments) and the most likely one is returned per
        sample.
        """
        candidates = [
            (int(label), token_ids)
            for label, token_ids in self._label_token_map.items()
            if token_ids
        ]
        if not candidates:
            raise ValueError("CTC score decision failed: no valid class tokenization found.")

        device = logits.device
        bsz, time_steps, num_tokens = logits.shape
        num_cands = len(candidates)

        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, C)
        expanded = (
            log_probs.unsqueeze(1)
            .expand(bsz, num_cands, time_steps, num_tokens)
            .reshape(bsz * num_cands, time_steps, num_tokens)
            .transpose(0, 1)  # (T, B*K, C) as expected by ctc_loss
        )
        input_lengths = torch.full((bsz * num_cands,), time_steps, dtype=torch.long, device=device)
        cand_tokens = torch.cat(
            [torch.as_tensor(toks, dtype=torch.long, device=device) for _, toks in candidates]
        )
        cand_lengths = torch.tensor([len(toks) for _, toks in candidates], dtype=torch.long, device=device)
        targets = cand_tokens.repeat(bsz)
        target_lengths = cand_lengths.repeat(bsz)

        losses = F.ctc_loss(
            expanded,
            targets,
            input_lengths,
            target_lengths,
            blank=self.text_mapper.blank_id,
            reduction="none",
            zero_infinity=True,
        ).view(bsz, num_cands)

        label_ids = torch.tensor([label for label, _ in candidates], device=device)
        best = label_ids[losses.argmin(dim=1)]
        return best.detach().cpu().numpy().astype(np.int64)

    def _decode_texts(self, logits: torch.Tensor, force_greedy: bool = False) -> List[str]:
        """Decode logits to character strings with the configured decoder."""
        if self.decode_strategy == "greedy" or force_greedy:
            return self.text_mapper.token_int_to_texts(self.greedy_decode(logits))

        # Beam search: run per sample on log-softmax probabilities.
        log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
        blank_id = self.text_mapper.blank_id
        texts: List[str] = []
        for seq in log_probs:
            ids = ctc_prefix_beam_search(
                seq.tolist(),
                self.beam_width,
                blank_id,
                temperature=self.beam_temperature,
                blank_penalty=self.beam_blank_penalty,
                length_bonus=self.beam_length_bonus,
            )
            texts.append(self.text_mapper.clean_text(self.text_mapper.int_to_text(ids)))
        return texts

    def predict_labels(self, outputs, use_score_fallback: bool = True) -> np.ndarray:
        with torch.no_grad():
            logits = self._extract_logits(outputs)
            texts = self._decode_texts(logits, force_greedy=not use_score_fallback)
            preds, _ = self.text_mapper.texts_to_label_int(
                texts, allow_nearest=self.allow_nearest
            )

            if use_score_fallback and any(int(p) < 0 for p in preds):
                for i, pred in enumerate(preds):
                    if int(pred) < 0:
                        preds[i] = self._best_label_from_ctc_scores(logits[i])

        return np.asarray(preds, dtype=np.int64)

    def backward(self, loss: torch.Tensor) -> None:
        """Temporarily disable deterministic algorithms on CUDA for CTC backward."""
        deterministic = loss.device.type == "cuda" and torch.are_deterministic_algorithms_enabled()
        if deterministic:
            torch.use_deterministic_algorithms(False)
        loss.backward()
        if deterministic:
            torch.use_deterministic_algorithms(True)


class CTCRecognitionStrategy(CTCStrategy):
    """CTC recognition strategy (WER/CER scored).

    Same CNN+BiLSTM+CTC training as CTCStrategy, but decoding returns the raw
    CTC-collapsed character string over the fixed English alphabet,
    without mapping it back to a numeric class or applying the lexicon
    nearest-match. It is scored with WER/CER metrics.

    The decoder is open (it may emit any string over a-z+space),
    but the reference/target texts are drawn from the closed command/sentence
    set.

    Decoding uses the shared CTCStrategy decoders (`decode_strategy`: 'greedy' or
    'beam').

    `primary_metric` is the WER/CER key used for validation-based model selection
    and headline reporting: "cer" for word-level runs (where WER degenerates to
    exact match on a single token) and "wer" for sentence-level runs.
    """

    def __init__(
        self,
        text_mapper,
        decode_strategy: str = "greedy",
        beam_width: int = 10,
        beam_temperature: float = 1.0,
        beam_blank_penalty: float = 0.0,
        beam_length_bonus: float = 0.0,
        label_mode: str = "sentence",
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            text_mapper,
            allow_nearest=False,
            decode_strategy=decode_strategy,
            beam_width=beam_width,
            beam_temperature=beam_temperature,
            beam_blank_penalty=beam_blank_penalty,
            beam_length_bonus=beam_length_bonus,
            label_smoothing=label_smoothing,
        )
        self.label_mode = str(label_mode).lower().strip()
        self.primary_metric = "cer" if self.label_mode == "word" else "wer"

    def predict_texts(self, outputs, force_greedy: bool = False) -> List[str]:
        """Decode to collapsed character strings (no lexicon mapping).

        Uses the configured decoder ('greedy'/'beam'); `force_greedy` overrides to
        greedy (used for per-epoch validation monitoring even when the final
        test decode is beam search).
        """
        return self._decode_texts(self._extract_logits(outputs), force_greedy=force_greedy)

    def reference_texts(self, targets: torch.Tensor) -> List[str]:
        """Ground-truth texts for the target labels."""
        return self.text_mapper.label_int_to_texts(targets)

    def predict_labels(self, outputs, use_score_fallback: bool = True) -> np.ndarray:
        """Exact-match mapping only (no lexicon nearest or CTC-score fallback).
        """
        texts = self.predict_texts(outputs, force_greedy=True)
        preds, _ = self.text_mapper.texts_to_label_int(texts, allow_nearest=False)
        return np.asarray(preds, dtype=np.int64)