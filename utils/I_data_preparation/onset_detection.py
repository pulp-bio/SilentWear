# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Trigger-free speech onset detection from multichannel sEMG
==========================================================

This module locates the intervals in which the subject speaks **without ever
reading the trigger channel**.

Algorithm
---------
Four stages, all causal:

1. **Envelope** - rectify each channel and smooth with a *causal* moving
   average over ``smooth_ms`` (sample ``n`` uses only ``n-w+1 .. n``).

2. **Normalisation** - per channel, ``z = (env - LO) / (HI - LO)`` where
   ``LO``/``HI`` are the 20th/80th percentiles of that channel's envelope over
   the preceding ``baseline_window_s`` seconds, refreshed every
   ``baseline_update_s``. ``LO`` tracks the resting level and ``HI - LO`` the
   channel's own dynamic range, so ``z`` is **dimensionless**.

3. **Channel aggregation** - the activity trace is the mean of the ``topk``
   largest ``z`` values *at each sample*.

4. **Dual threshold with backward search** - a run of activity above ``t_low``
   is only accepted once it reaches ``t_high``; the reported onset is then the
   *start of that run*, i.e. the algorithm walks back from the confirmation
   point to where the rise began, bounded by ``max_backtrack_s``. The two
   thresholds decouple the two decisions: ``t_high`` alone governs the false
   alarm rate, ``t_low`` alone governs how early the onset is placed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.I_data_preparation.experimental_config import FS


@dataclass
class OnsetConfig:
    """Constants of the detector."""

    fs: int = FS

    # Stage 1: envelope
    smooth_ms: float = 30.0

    # Stage 2: causal baseline
    baseline_window_s: float = 20.0
    baseline_update_s: float = 1.0
    baseline_decim_hz: float = 25.0
    low_pct: float = 20.0
    high_pct: float = 80.0
    warmup_s: float = 5.0

    # Stage 3: channel aggregation
    topk: int = 4

    # Stage 4: dual threshold
    t_high: float = 2.5
    t_low_ratio: float = 0.4
    min_duration_s: float = 0.15
    merge_gap_s: float = 0.30
    max_backtrack_s: float = 0.30

    match_tolerance_s: float = 0.50

    @property
    def t_low(self) -> float:
        return self.t_high * self.t_low_ratio

    def __post_init__(self) -> None:
        if not 0.0 < self.t_low_ratio <= 1.0:
            raise ValueError(f"t_low_ratio must be in (0, 1], got {self.t_low_ratio}")
        if self.topk < 1:
            raise ValueError(f"topk must be >= 1, got {self.topk}")
        if self.baseline_window_s < self.warmup_s:
            raise ValueError("baseline_window_s must be >= warmup_s")


@dataclass(frozen=True)
class Baseline:
    """``(low, high)`` envelope percentiles, stored per block."""

    lo: np.ndarray  # (n_blocks, n_channels)
    hi: np.ndarray  # (n_blocks, n_channels)
    block_samples: int
    n_samples: int
    first_valid_block: int

    def blocks(self) -> Iterator[Tuple[int, int, int]]:
        """Yield the ``(block, start, stop)`` sample span of every block past the warm-up."""
        for b in range(self.first_valid_block, self.lo.shape[0]):
            start = b * self.block_samples
            yield b, start, min(start + self.block_samples, self.n_samples)

    def expand(self) -> Tuple[np.ndarray, np.ndarray]:
        """Expand into per-sample ``(LO, HI)`` arrays, for plotting and debugging.

        This is the representation the detector used to carry around.
        """
        idx = np.arange(self.n_samples) // self.block_samples
        return self.lo[idx], self.hi[idx]


@dataclass(frozen=True)
class SpeechEvent:
    """One detected speech interval, in sample indices of the recording."""

    onset: int
    offset: int
    peak_activity: float

    @property
    def duration_samples(self) -> int:
        return self.offset - self.onset


# ---------------------------------------------------------------------------
# Stage 1-3
# ---------------------------------------------------------------------------


def causal_envelope(X: np.ndarray, cfg: OnsetConfig) -> np.ndarray:
    """Compute the rectified signal smoothed by a causal moving average.

    ``out[n]`` averages ``|X|`` over ``[n - w + 1, n]``
    """
    w = max(1, int(round(cfg.smooth_ms * cfg.fs / 1000.0)))
    kernel = np.ones(w) / w
    n = X.shape[0]
    out = np.empty_like(X, dtype=float)
    for ch in range(X.shape[1]):
        out[:, ch] = np.convolve(np.abs(X[:, ch]), kernel)[:n]
    return out


def causal_baseline(E: np.ndarray, cfg: OnsetConfig) -> Baseline:
    """Calculate the ``(low, high)`` percentiles of the envelope over past samples only."""
    n, n_ch = E.shape
    decim = max(1, int(round(cfg.fs / cfg.baseline_decim_hz)))
    E_dec = E[::decim]

    update = max(1, int(round(cfg.baseline_update_s * cfg.fs)))
    win_dec = max(1, int(round(cfg.baseline_window_s * cfg.fs / decim)))
    min_dec = max(1, int(round(cfg.warmup_s * cfg.fs / decim)))

    n_blocks = (n + update - 1) // update
    lo = np.full((n_blocks, n_ch), np.nan)
    hi = np.full((n_blocks, n_ch), np.nan)

    first_valid = n_blocks
    for b in range(n_blocks):
        end_dec = (b * update) // decim
        beg_dec = max(0, end_dec - win_dec)
        if end_dec - beg_dec < min_dec:
            continue  # still warming up
        lo[b], hi[b] = np.percentile(E_dec[beg_dec:end_dec], [cfg.low_pct, cfg.high_pct], axis=0)
        first_valid = min(first_valid, b)

    return Baseline(lo=lo, hi=hi, block_samples=update, n_samples=n, first_valid_block=first_valid)


def compute_activity(X: np.ndarray, cfg: Optional[OnsetConfig] = None) -> np.ndarray:
    """Compute the dimensionless activity trace the thresholds are applied to.

    Parameters
    ----------
    X : (n_samples, n_channels) filtered EMG.

    Returns
    -------
    (n_samples,) activity, 0 wherever the baseline is not yet available.
    """
    cfg = cfg or OnsetConfig()
    E = causal_envelope(X, cfg)
    base = causal_baseline(E, cfg)

    topk = min(cfg.topk, E.shape[1])

    activity = np.zeros(E.shape[0], dtype=float)
    for b, start, stop in base.blocks():
        lo = base.lo[b]
        scale = np.maximum(base.hi[b] - lo, 1e-12)
        Z = (E[start:stop] - lo) / scale
        # Per-sample top-k across channels.
        part = np.partition(Z, -topk, axis=1)[:, -topk:]
        activity[start:stop] = part.mean(axis=1)
    return np.nan_to_num(activity, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------


def detect_events(activity: np.ndarray, cfg: Optional[OnsetConfig] = None) -> List[SpeechEvent]:
    """Run a dual-threshold state machine over the activity trace."""
    cfg = cfg or OnsetConfig()
    t_high, t_low = cfg.t_high, cfg.t_low

    above_low = activity > t_low
    if not above_low.any():
        return []

    edges = np.diff(above_low.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    stops = np.flatnonzero(edges == -1) + 1
    if above_low[0]:
        starts = np.r_[0, starts]
    if above_low[-1]:
        stops = np.r_[stops, len(above_low)]

    max_back = int(round(cfg.max_backtrack_s * cfg.fs))
    confirmed: List[Tuple[int, int]] = []
    for start, stop in zip(starts, stops):
        hits = np.flatnonzero(activity[start:stop] > t_high)
        if hits.size == 0:
            continue
        confirm_at = start + int(hits[0])
        confirmed.append((max(start, confirm_at - max_back), stop))

    if not confirmed:
        return []

    gap = int(round(cfg.merge_gap_s * cfg.fs))
    merged: List[List[int]] = [list(confirmed[0])]
    for start, stop in confirmed[1:]:
        if start - merged[-1][1] < gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])

    min_len = int(round(cfg.min_duration_s * cfg.fs))
    return [
        SpeechEvent(onset=s, offset=e, peak_activity=float(activity[s:e].max()))
        for s, e in merged
        if e - s >= min_len
    ]


def detect_speech_events(
    X: np.ndarray, cfg: Optional[OnsetConfig] = None
) -> List[SpeechEvent]:
    """Detect speech intervals from filtered channels."""
    cfg = cfg or OnsetConfig()
    return detect_events(compute_activity(X, cfg), cfg)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def filtered_channel_columns(df: pd.DataFrame) -> List[str]:
    """Extract filtered channel columns, i.e. what the models consume."""
    return [c for c in df.columns if c.startswith("Ch_") and c.endswith("_filt")]


def detect_events_in_dataframe(
    df: pd.DataFrame, cfg: Optional[OnsetConfig] = None
) -> List[SpeechEvent]:
    """Run the detector on a processed recording."""
    cols = filtered_channel_columns(df)
    if not cols:
        raise ValueError("No filtered channel columns (Ch_*_filt) in the DataFrame.")
    return detect_speech_events(df[cols].to_numpy(dtype=float), cfg)


def label_boxes(df: pd.DataFrame, label_col: str = "Label_int") -> List[Tuple[int, int, int]]:
    """Extract contiguous non-rest trigger runs as ``(start, stop, label_int)``."""
    lab = df[label_col].to_numpy()
    if lab.size == 0:
        return []
    edges = np.r_[True, lab[1:] != lab[:-1]]
    starts = np.flatnonzero(edges)
    stops = np.r_[starts[1:], len(lab)]
    return [
        (int(s), int(e), int(lab[s])) for s, e in zip(starts, stops) if lab[s] != 0
    ]


def match_events_to_boxes(
    events: Sequence[SpeechEvent],
    boxes: Sequence[Tuple[int, int, int]],
    tolerance_s: float = OnsetConfig.match_tolerance_s,
    fs: int = FS,
) -> List[Optional[int]]:
    """Find for each event the index of the trigger box it belongs to (or ``None``).

    An event matches the box whose span contains its onset, allowing the onset to
    anticipate the cue by ``tolerance_s``.
    """
    tol = int(round(tolerance_s * fs))
    out: List[Optional[int]] = []
    for ev in events:
        match = None
        for bi, (bs, be, _) in enumerate(boxes):
            if bs - tol <= ev.onset < be:
                match = bi
                break
        out.append(match)
    return out


def rest_intervals_between_events(
    events: Sequence[SpeechEvent], n_samples: int, min_gap_samples: int
) -> List[Tuple[int, int]]:
    """Find gaps between detected events that are long enough to host a rest window."""
    gaps: List[Tuple[int, int]] = []
    cursor = 0
    for ev in events:
        if ev.onset - cursor >= min_gap_samples:
            gaps.append((cursor, ev.onset))
        cursor = max(cursor, ev.offset)
    if n_samples - cursor >= min_gap_samples:
        gaps.append((cursor, n_samples))
    return gaps


def rest_spans_from_boxes(
    boxes: Sequence[Tuple[int, int, int]], n_samples: int
) -> List[Tuple[int, int]]:
    """Compute the complement of the cue boxes, i.e. the rest spans of the recording."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for bs, be, _ in boxes:
        if bs > cursor:
            spans.append((cursor, bs))
        cursor = max(cursor, be)
    if n_samples > cursor:
        spans.append((cursor, n_samples))
    return spans


def binary_rates_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Calculate precision, recall and the rest of the rates a 2x2 table defines."""
    div = lambda a, b: float(a) / b if b else float("nan")
    recall = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": div(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": div(fp, fp + tn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else float("nan"),
    }


def _binary_rates(tp: int, fp: int, fn: int, tn: int, prefix: str) -> dict:
    """Compute :func:`binary_rates_from_counts` under a key prefix."""
    return {f"{prefix}_{k}": v for k, v in binary_rates_from_counts(tp, fp, fn, tn).items()}


def binary_scores(
    events: Sequence[SpeechEvent],
    boxes: Sequence[Tuple[int, int, int]],
    n_samples: int,
    tolerance_s: float = OnsetConfig.match_tolerance_s,
    fs: int = FS,
) -> dict:
    """Read the detector as a binary speech-against-rest classifier."""
    matches = match_events_to_boxes(events, boxes, tolerance_s, fs)

    seg_tp = len({m for m in matches if m is not None})
    seg_fn = len(boxes) - seg_tp
    rest = rest_spans_from_boxes(boxes, n_samples)
    woken = {
        ri
        for ev, m in zip(events, matches)
        if m is None
        for ri, (rs, re_) in enumerate(rest)
        if rs <= ev.onset < re_
    }
    seg_fp, seg_tn = len(woken), len(rest) - len(woken)

    pred = np.zeros(n_samples, dtype=bool)
    for ev in events:
        pred[ev.onset:ev.offset] = True
    true = np.zeros(n_samples, dtype=bool)
    for bs, be, _ in boxes:
        true[bs:be] = True
    smp_tp = int(np.count_nonzero(pred & true))
    smp_fp = int(np.count_nonzero(pred & ~true))
    smp_fn = int(np.count_nonzero(~pred & true))
    smp_tn = int(np.count_nonzero(~pred & ~true))

    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    return {
        "n_rest_spans": len(rest),
        "box_duration_median_s": med([(be - bs) / fs for bs, be, _ in boxes]),
        "rest_span_median_s": med([(re_ - rs) / fs for rs, re_ in rest]),
        "flagged_frac": float(np.count_nonzero(pred)) / n_samples if n_samples else float("nan"),
        **_binary_rates(seg_tp, seg_fp, seg_fn, seg_tn, "seg"),
        **_binary_rates(smp_tp, smp_fp, smp_fn, smp_tn, "smp"),
    }


def score_against_trigger(
    events: Sequence[SpeechEvent],
    boxes: Sequence[Tuple[int, int, int]],
    n_samples: int,
    tolerance_s: float = OnsetConfig.match_tolerance_s,
    fs: int = FS,
) -> dict:
    """Evaluate the detector against the trigger boxes.

    ``n_detected`` cue boxes out of ``n_boxes`` received at least one onset;
    ``detection_rate`` is their ratio. ``false_alarms`` counts detected events
    that overlap no box at all, so an utterance split in two is not punished
    twice.
    """
    matches = match_events_to_boxes(events, boxes, tolerance_s, fs)
    covered = {m for m in matches if m is not None}

    latencies, durations, spans = [], [], []
    for bi, (bs, _, _) in enumerate(boxes):
        own = [i for i, m in enumerate(matches) if m == bi]
        if not own:
            continue
        first = events[own[0]]
        last_offset = max(events[i].offset for i in own)
        latencies.append((first.onset - bs) / fs)
        durations.append((last_offset - first.onset) / fs)
        spans.append((last_offset - bs) / fs)

    matched_peak = [ev.peak_activity for ev, m in zip(events, matches) if m is not None]
    unmatched_peak = [ev.peak_activity for ev, m in zip(events, matches) if m is None]
    matched_dur = [(ev.offset - ev.onset) / fs for ev, m in zip(events, matches) if m is not None]
    unmatched_dur = [(ev.offset - ev.onset) / fs for ev, m in zip(events, matches) if m is None]

    false_alarms = sum(
        1
        for ev in events
        if not any(ev.onset < be and ev.offset > bs for bs, be, _ in boxes)
    )
    pct = lambda v, p: float(np.percentile(v, p)) if v else float("nan")
    return {
        "n_boxes": len(boxes),
        "n_events": len(events),
        "n_detected": len(covered),
        "detection_rate": len(covered) / len(boxes) if boxes else float("nan"),
        "false_alarms": false_alarms,
        "false_alarms_per_min": false_alarms / (n_samples / fs / 60) if n_samples else float("nan"),
        "onset_latency_median_s": pct(latencies, 50),
        "onset_latency_p90_s": pct(latencies, 90),
        "duration_median_s": pct(durations, 50),
        "duration_p90_s": pct(durations, 90),
        "duration_p95_s": pct(durations, 95),
        # Fraction of detected trials a 2.0 s cue-anchored window would truncate.
        "span_from_cue_median_s": pct(spans, 50),
        "span_from_cue_p90_s": pct(spans, 90),
        "truncated_at_2000ms_frac": (
            float(np.mean([s > 2.0 for s in spans])) if spans else float("nan")),
        "truncated_at_2400ms_frac": (
            float(np.mean([s > 2.4 for s in spans])) if spans else float("nan")),
        # Matched against unmatched events, on amplitude and on duration.
        "peak_matched_median": pct(matched_peak, 50),
        "peak_unmatched_median": pct(unmatched_peak, 50),
        "duration_matched_median_s": pct(matched_dur, 50),
        "duration_unmatched_median_s": pct(unmatched_dur, 50),
        "n_unmatched_events": len(unmatched_peak),
        **binary_scores(events, boxes, n_samples, tolerance_s, fs),
    }
