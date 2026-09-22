#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Offline CTC prefix-beam-search parameter sweep.
===============================================

Why this exists
---------------
Decode parameters do not affect training: the trained CTC model is fixed, only the 
decoder changes. So this tool sweeps decode parameters over **saved test-set log-probs**, 
dumped once during a normal run.

Producing the dumps
-------------------
Add ``dump_logprobs: true`` under ``model.kwargs.train_cfg.ctc`` in the ablation
config and run the experiment once. Each CV fold then writes
``<mode>_fold_<k>_logprobs.npz`` (+ a ``.meta.json`` sibling) next to its
checkpoint. Point ``--dumps`` at those files / their directory.

Usage
-----
    python offline_experiments/VII_beam_sweep.py \
        --dumps artifacts_ablation/<run>/.../model_1 \
        --out   artifacts_ablation/<run>/beam_sweep.csv \
        --beam_widths 1 5 10 25 \
        --temperatures 1.0 1.3 1.6 2.0 \
        --blank_penalties 0.0 1.0 2.0 \
        --length_bonuses 0.0 0.5 1.0 \
        --jobs 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import editdistance

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.ctc_decoding import ctc_prefix_beam_search, ctc_greedy_decode
from models.utils import compute_wer_metrics
from utils.I_data_preparation.text_transform import CTCTextTransform

DEFAULT_BEAM_WIDTH=[5, 10, 25]
DEFAULT_TEMPERATURES=[1.0, 1.3, 1.6, 2.0]
DEFAULT_BLANK_PENALTIES=[0.0, 1.0, 2.0]
DEFAULT_LENGTH_BONUSES=[0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Offline mapper
# ---------------------------------------------------------------------------


class OfflineCTCMapper:
    """Minimal token<->text mapper reconstructed from a dump's ``.meta.json``."""

    def __init__(self, meta: dict):
        self.blank_id = int(meta["blank_id"])
        self.int_to_char = {int(k): v for k, v in meta["int_to_char"].items()}
        self.label_to_text_map = {
            int(k): CTCTextTransform.clean_text(v) for k, v in meta["label_to_text_map"].items()
        }
        self.text_to_label_map = {text: label for label, text in self.label_to_text_map.items()}
        self.label_mode = meta.get("label_mode")
        self.task = meta.get("task", "recognition")
        self.word_vocabulary = tuple(
            sorted({w for text in self.label_to_text_map.values() for w in text.split()})
        )

    @staticmethod
    def clean_text(text: str) -> str:
        return CTCTextTransform.clean_text(text)

    def ids_to_text(self, ids: Sequence[int]) -> str:
        raw = "".join(self.int_to_char.get(int(i), "") for i in ids)
        return self.clean_text(raw)

    def label_int_to_texts(self, labels: Sequence[int]) -> List[str]:
        return [self.label_to_text_map.get(int(lbl), str(int(lbl))) for lbl in labels]

    def _closest_known_text(self, text: str) -> Optional[str]:
        if not text or not self.text_to_label_map:
            return None
        if editdistance is None:
            raise ImportError("editdistance is required for allow_nearest classification mapping.")
        ed = editdistance
        return min(self.text_to_label_map, key=lambda cand: ed.eval(text, cand))

    def texts_to_label_int(self, texts: List[str], allow_nearest: bool = True) -> List[int]:
        """Same rule as CTCTextMapper.texts_to_label_int (exact -> nearest -> -1)."""
        preds: List[int] = []
        for raw_text in texts:
            text = self.clean_text(raw_text)
            if text in self.text_to_label_map:
                preds.append(int(self.text_to_label_map[text]))
            elif text == "":
                preds.append(-1)
            elif allow_nearest:
                nearest = self._closest_known_text(text)
                preds.append(int(self.text_to_label_map[nearest]) if nearest is not None else -1)
            else:
                preds.append(-1)
        return preds


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------


def _discover_dumps(paths: Sequence[Path]) -> List[Path]:
    """Expand files / directories / globs into a sorted list of *_logprobs.npz."""
    found: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found.extend(sorted(p.rglob("*logprobs*.npz")))
        elif any(ch in str(p) for ch in "*?["):
            found.extend(sorted(Path().glob(str(p))))
        elif p.suffix == ".npz":
            found.append(p)
        else:
            raise FileNotFoundError(f"Unrecognised dump path: {p}")
    seen, uniq = set(), []
    for f in found:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    if not uniq:
        raise FileNotFoundError(f"No *logprobs*.npz dumps found under: {[str(p) for p in paths]}")
    return uniq


def load_dumps(paths: Sequence[Path]) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    """Load one or more fold dumps into a flat sample list + labels + shared meta.

    Returns (seqs, labels, meta) where ``seqs`` is a list of (T, C) float32
    log-prob arrays (ragged-safe across folds) and ``labels`` is (N,) int64.
    """
    files = _discover_dumps(paths)
    seqs: List[np.ndarray] = []
    labels: List[int] = []
    meta0: Optional[dict] = None

    for f in files:
        meta_path = f.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta sidecar for dump {f}: expected {meta_path}")
        meta = json.loads(meta_path.read_text())
        if meta0 is None:
            meta0 = meta
        elif meta.get("int_to_char") != meta0.get("int_to_char") or meta.get("blank_id") != meta0.get("blank_id"):
            raise ValueError(f"Incompatible token table in {f}; cannot pool folds with different vocabularies.")

        data = np.load(f)
        lp = data["log_probs"].astype(np.float32)   # (n, T, C)
        lbl = data["labels"].astype(np.int64)       # (n,)
        for i in range(lp.shape[0]):
            seqs.append(lp[i])
        labels.extend(lbl.tolist())
        print(f"  loaded {f.name}: {lp.shape[0]} samples, T={lp.shape[1]}, C={lp.shape[2]}")

    assert meta0 is not None
    print(f"[LOAD] {len(files)} dump(s) -> {len(seqs)} pooled samples | task={meta0.get('task')}")
    return seqs, np.asarray(labels, dtype=np.int64), meta0


# ---------------------------------------------------------------------------
# Decoding (single sample)
# ---------------------------------------------------------------------------


_SEQS: List[np.ndarray] = []
_BLANK_ID: int = 0


def _decode_seq(seq: np.ndarray, combo: "DecodeCombo", blank_id: int) -> List[int]:
    if combo.decode == "greedy":
        argmax_ids = seq.argmax(axis=-1).tolist()
        return ctc_greedy_decode(argmax_ids, blank_id)
    return ctc_prefix_beam_search(
        seq.tolist(),
        combo.beam_width,
        blank_id,
        temperature=combo.temperature,
        blank_penalty=combo.blank_penalty,
        length_bonus=combo.length_bonus,
    )


def _decode_index(args: Tuple[int, "DecodeCombo"]) -> List[int]:
    idx, combo = args
    return _decode_seq(_SEQS[idx], combo, _BLANK_ID)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


class DecodeCombo:
    __slots__ = ("decode", "beam_width", "temperature", "blank_penalty", "length_bonus")

    def __init__(self, decode, beam_width=0, temperature=1.0, blank_penalty=0.0, length_bonus=0.0):
        self.decode = decode
        self.beam_width = int(beam_width)
        self.temperature = float(temperature)
        self.blank_penalty = float(blank_penalty)
        self.length_bonus = float(length_bonus)

    def as_row(self) -> dict:
        return {
            "decode": self.decode,
            "beam_width": self.beam_width if self.decode == "beam" else "",
            "temperature": round(self.temperature, 4) if self.decode == "beam" else "",
            "blank_penalty": round(self.blank_penalty, 4) if self.decode == "beam" else "",
            "length_bonus": round(self.length_bonus, 4) if self.decode == "beam" else "",
        }

    def label(self) -> str:
        if self.decode == "greedy":
            return "greedy"
        return f"beam(w={self.beam_width},T={self.temperature},bp={self.blank_penalty},lb={self.length_bonus})"


def build_combos(args) -> List[DecodeCombo]:
    combos: List[DecodeCombo] = []
    if args.include_greedy:
        combos.append(DecodeCombo("greedy"))
    for bw, temp, bp, lb in itertools.product(
        args.beam_widths, args.temperatures, args.blank_penalties, args.length_bonuses
    ):
        combos.append(DecodeCombo("beam", bw, temp, bp, lb))
    return combos


def evaluate_combo(
    combo: DecodeCombo,
    seqs: List[np.ndarray],
    labels: np.ndarray,
    mapper: OfflineCTCMapper,
    allow_nearest: bool,
    pool=None,
) -> dict:
    """Decode every sample with ``combo`` and score recognition + classification."""
    n = len(seqs)
    if pool is None:
        decoded_ids = [_decode_seq(seqs[i], combo, mapper.blank_id) for i in range(n)]
    else:
        decoded_ids = pool.map(_decode_index, ((i, combo) for i in range(n)), chunksize=16)

    hyps = [mapper.ids_to_text(ids) for ids in decoded_ids]
    refs = mapper.label_int_to_texts(labels.tolist())

    rec_metrics, _, _ = compute_wer_metrics(refs, hyps, verbose=False, vocabulary=mapper.word_vocabulary)

    # Classification:
    preds = np.asarray(mapper.texts_to_label_int(hyps, allow_nearest=allow_nearest), dtype=np.int64)
    acc = float(np.mean(preds == labels))
    try:
        import warnings

        from sklearn.metrics import balanced_accuracy_score

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            bal_acc = float(balanced_accuracy_score(labels, preds))
    except Exception:
        bal_acc = float("nan")
    empty_rate = float(np.mean([h == "" for h in hyps]))

    row = combo.as_row()
    row.update(
        {
            "n_samples": int(n),
            "wer": round(rec_metrics["wer"], 4),
            "balanced_wer": round(rec_metrics["balanced_wer"], 4),
            "cer": round(rec_metrics["cer"], 4),
            "balanced_cer": round(rec_metrics["balanced_cer"], 4),
            "vocab_wer": round(rec_metrics["vocab_wer"], 4),
            "balanced_vocab_wer": round(rec_metrics["balanced_vocab_wer"], 4),
            "cls_accuracy": round(acc, 4),
            "cls_balanced_accuracy": round(bal_acc, 4),
            "empty_rate": round(empty_rate, 4),
        }
    )
    return row


def run_sweep(seqs, labels, mapper, combos, allow_nearest, jobs) -> List[dict]:
    global _SEQS, _BLANK_ID
    pool = None
    if jobs and jobs > 1:
        import multiprocessing as mp

        _SEQS = seqs
        _BLANK_ID = mapper.blank_id
        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=jobs)

    rows: List[dict] = []
    try:
        for i, combo in enumerate(combos, 1):
            row = evaluate_combo(combo, seqs, labels, mapper, allow_nearest, pool=pool)
            rows.append(row)
            print(
                f"[{i:>3}/{len(combos)}] {combo.label():<45} "
                f"WER={row['wer']:.4f} CER={row['cer']:.4f} acc={row['cls_accuracy']:.4f}"
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_csv(rows: List[dict], out_path: Path) -> None:
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "decode", "beam_width", "temperature", "blank_penalty", "length_bonus",
        "n_samples", "wer", "balanced_wer", "cer", "balanced_cer",
        "vocab_wer", "balanced_vocab_wer",
        "cls_accuracy", "cls_balanced_accuracy", "empty_rate",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[WRITE] {len(rows)} rows -> {out_path}")


def _print_top(rows: List[dict], key: str, maximize: bool, top_n: int, title: str) -> None:
    ordered = sorted(rows, key=lambda r: r[key], reverse=maximize)
    print(f"\n=== Best by {title} ({'max' if maximize else 'min'} {key}) ===")
    for r in ordered[:top_n]:
        tag = r["decode"] if r["decode"] == "greedy" else (
            f"beam w={r['beam_width']} T={r['temperature']} bp={r['blank_penalty']} lb={r['length_bonus']}"
        )
        print(f"  {key}={r[key]:<8} | acc={r['cls_accuracy']:<7} wer={r['wer']:<7} cer={r['cer']:<7} | {tag}")


def report(rows: List[dict], mapper: OfflineCTCMapper, top_n: int) -> None:
    if not rows:
        return
    rec_key = "cer" if mapper.label_mode == "word" else "wer"
    _print_top(rows, rec_key, maximize=False, top_n=top_n, title=f"recognition ({rec_key.upper()})")
    _print_top(rows, "cls_accuracy", maximize=True, top_n=top_n, title="classification (accuracy)")

    greedy = [r for r in rows if r["decode"] == "greedy"]
    if greedy:
        g = greedy[0]
        print(f"\n[baseline] greedy: {rec_key}={g[rec_key]} acc={g['cls_accuracy']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", nargs="+", type=Path, required=True,
                    help="Dump .npz files, directories to search, or globs (*_logprobs.npz).")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path.")

    ap.add_argument("--beam_widths", nargs="+", type=int, default=DEFAULT_BEAM_WIDTH)
    ap.add_argument("--temperatures", nargs="+", type=float, default=DEFAULT_TEMPERATURES)
    ap.add_argument("--blank_penalties", nargs="+", type=float, default=DEFAULT_BLANK_PENALTIES)
    ap.add_argument("--length_bonuses", nargs="+", type=float, default=DEFAULT_LENGTH_BONUSES)

    ap.add_argument("--no_greedy", dest="include_greedy", action="store_false",
                    help="Skip the greedy baseline row (included by default).")
    ap.add_argument("--allow_nearest", dest="allow_nearest", action="store_true", default=True,
                    help="Map decoded text to nearest lexicon label for classification (default on).")
    ap.add_argument("--no_nearest", dest="allow_nearest", action="store_false",
                    help="Disable nearest-match; only exact text matches score a class.")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Randomly subsample this many pooled samples (speed). Default: all.")
    ap.add_argument("--seed", type=int, default=0, help="Subsampling seed.")
    ap.add_argument("--jobs", type=int, default=1, help="Worker processes for decoding (Linux fork).")
    ap.add_argument("--top_n", type=int, default=8, help="How many best configs to print per task.")

    args = ap.parse_args()

    seqs, labels, meta = load_dumps(args.dumps)
    mapper = OfflineCTCMapper(meta)

    if args.max_samples is not None and args.max_samples < len(seqs):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(seqs), size=args.max_samples, replace=False)
        seqs = [seqs[i] for i in idx]
        labels = labels[idx]
        print(f"[SUBSAMPLE] using {len(seqs)} of pooled samples (seed={args.seed})")

    combos = build_combos(args)
    print(f"[SWEEP] {len(combos)} decode configs over {len(seqs)} samples "
          f"(jobs={args.jobs}) | task hint: {meta.get('task')}")

    rows = run_sweep(seqs, labels, mapper, combos, args.allow_nearest, args.jobs)
    write_csv(rows, args.out)
    report(rows, mapper, args.top_n)


if __name__ == "__main__":
    main()
