# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
CTC decoding strategies
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import math


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _rescore_rows(
    log_probs: Sequence[Sequence[float]],
    blank_id: int,
    temperature: float,
    blank_penalty: float,
) -> Sequence[Sequence[float]]:
    """Apply temperature rescaling and a blank penalty to per-frame log-probs."""
    need_temp = temperature is not None and abs(temperature - 1.0) > 1e-9
    need_blank = abs(blank_penalty) > 1e-12
    if not need_temp and not need_blank:
        return log_probs

    inv_t = 1.0 / temperature if need_temp else 1.0
    rows = []
    for row in log_probs:
        if need_temp:
            scaled = [lp * inv_t for lp in row]
            m = max(scaled)
            lse = m + math.log(sum(math.exp(v - m) for v in scaled))
            new_row = [v - lse for v in scaled]
        else:
            new_row = list(row)
        if need_blank:
            new_row[blank_id] -= blank_penalty
        rows.append(new_row)
    return rows


def ctc_prefix_beam_search(
    log_probs: Sequence[Sequence[float]],
    beam_width: int,
    blank_id: int = 0,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    length_bonus: float = 0.0,
) -> List[int]:
    """Prefix beam search over a single utterance.

    Args:
        log_probs: (T, C) log-softmax probabilities (list/ndarray-like).
        beam_width: number of prefixes kept after each frame.
        blank_id: CTC blank index.
        temperature: softmax temperature applied to the per-frame posteriors
            before the search (T > 1 flattens peaky CTC distributions). Default
            1.0 leaves them unchanged.
        blank_penalty: constant subtracted from the blank log-prob every frame to
            counter CTC's blank/deletion bias. Default 0.0 is a no-op.
        length_bonus: reward added per emitted symbol to the prefix score used for
            pruning and final selection (a.k.a. word/insertion bonus). Counters
            CTC's bias towards short outputs. Default 0.0 is a no-op.

    Returns:
        The most probable collapsed token-id sequence.
    """
    T = len(log_probs)
    if T == 0:
        return []
    C = len(log_probs[0])

    log_probs = _rescore_rows(log_probs, blank_id, temperature, blank_penalty)

    NEG = -math.inf

    def _score(entry, prefix_len: int) -> float:
        return _logsumexp(entry[0], entry[1]) + length_bonus * prefix_len

    beams = {(): [0.0, NEG]}

    for t in range(T):
        row = log_probs[t]
        next_beams: dict = {}

        def _get(prefix):
            e = next_beams.get(prefix)
            if e is None:
                e = [NEG, NEG]
                next_beams[prefix] = e
            return e

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _logsumexp(p_b, p_nb)
            last = prefix[-1] if prefix else -1

            for c in range(C):
                lp = row[c]
                if lp == NEG:
                    continue

                if c == blank_id:
                    e = _get(prefix)
                    e[0] = _logsumexp(e[0], p_total + lp)
                    continue

                if c == last:
                    e_new = _get(prefix + (c,))
                    e_new[1] = _logsumexp(e_new[1], p_b + lp)
                    e_same = _get(prefix)
                    e_same[1] = _logsumexp(e_same[1], p_nb + lp)
                else:
                    e_new = _get(prefix + (c,))
                    e_new[1] = _logsumexp(e_new[1], p_total + lp)

        # Keep the beam_width most probable prefixes (length bonus included).
        scored = sorted(
            next_beams.items(),
            key=lambda kv: _score(kv[1], len(kv[0])),
            reverse=True,
        )
        beams = dict(scored[: max(1, beam_width)])

    best_prefix = max(beams.items(), key=lambda kv: _score(kv[1], len(kv[0])))[0]
    return list(best_prefix)


def ctc_greedy_decode(argmax_ids: Sequence[int], blank_id: int = 0) -> List[int]:
    """Best-path decode: collapse repeats then remove blanks from an argmax path."""
    collapsed: List[int] = []
    prev: Optional[int] = None
    for tok in argmax_ids:
        if tok == blank_id:
            prev = tok
            continue
        if tok != prev:
            collapsed.append(int(tok))
        prev = tok
    return collapsed
