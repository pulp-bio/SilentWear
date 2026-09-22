#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Per-subject result tables at the best-beam CTC decode.
=========================================================================

This is the beam-search counterpart of
``utils/III_results_analysis/I_global_intersession_analysis.py``.

Why a separate script
---------------------
``I_global_intersession_analysis.py`` builds its per-subject/condition tables from
``cv_summary.csv``, whose metrics were computed at training time with the decode
baked into the config -- ``decode_strategy: greedy`` for the beam-sweep runs. So
those tables are locked to greedy and can never show a beam operating point.

``offline_experiments/VII_beam_sweep.py`` sweeps the decode parameters over the cached
per-fold log-probs, but only reports **pooled** aggregate metrics (one number over
all samples), with no per-subject breakdown.

This script combines the two: it re-decodes the same cached per-fold dumps and
emits the **exact same per-subject/condition table** as I_global_intersession, but
for the beam configuration that scored best in ``beam_sweep_<experiment>.csv`` --
with no retraining. Recognition runs produce a WER/CER table, classification runs a
balanced-accuracy table, matching the original format and file naming (with a
``_beam`` decode suffix). Only the best-beam table is written by default; pass
``--greedy`` to also emit the greedy-decode table for comparison.

Usage
-----
    python utils/III_results_analysis/VII_beam_sweep_tables.py \
        --artifacts_dir artifacts_beam_sweep/transformer_stft_ctc_recognition_beam_sweep \
        --experiment global \
        --model_name speechnet_transformer \
        --model_name_id w2000ms \
        --model_run model_1 \
        --subjects S01 S02 S03 S04 S05 S07 \
        --conditions silent vocalized

The best beam config is read from ``<artifacts_dir>/beam_sweep_<experiment>.csv``
(or ``--sweep_csv``); pass ``--select_metric`` to override the ranking metric. Add
``--dump_predictions`` to also write a per-sample ``*_predictions_beam.txt`` next
to each fold dump (same format as the training-time dump), for qualitative
inspection of the beam output (e.g. to decide whether to add a language model).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.VII_beam_sweep import DecodeCombo, OfflineCTCMapper, _decode_seq, load_dumps
from models.utils import compute_wer_metrics

RECOGNITION_METRICS = [
    "wer", "balanced_wer", "cer", "balanced_cer", "vocab_wer", "balanced_vocab_wer",
]


# ---------------------------------------------------------------------------
# Parallel per-fold decoding
# ---------------------------------------------------------------------------


_POOL_SEQS: List[np.ndarray] = []
_POOL_COMBO: Optional[DecodeCombo] = None
_POOL_BLANK: int = 0


# ---------------------------------------------------------------------------
# Decoding pool
# ---------------------------------------------------------------------------


def _pool_decode(idx: int) -> List[int]:
    assert _POOL_COMBO is not None
    return _decode_seq(_POOL_SEQS[idx], _POOL_COMBO, _POOL_BLANK)


def _decode_fold(seqs: List[np.ndarray], combo: DecodeCombo, blank_id: int, jobs: int) -> List[List[int]]:
    if jobs and jobs > 1 and len(seqs) > 1:
        import multiprocessing as mp

        global _POOL_SEQS, _POOL_COMBO, _POOL_BLANK
        _POOL_SEQS, _POOL_COMBO, _POOL_BLANK = seqs, combo, blank_id
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=jobs) as pool:
            return pool.map(_pool_decode, range(len(seqs)), chunksize=16)
    return [_decode_seq(s, combo, blank_id) for s in seqs]


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def _fold_dumps(run_path: Path) -> List[Path]:
    """Sorted per-fold log-prob dumps under a single model_<k> folder."""
    return sorted(run_path.glob("*logprobs*.npz"))


def _find_runs(
    artifacts_dir: Path,
    experiment: str,
    subjects: Sequence[str],
    conditions: Sequence[str],
    model_name: str,
    model_name_id: str,
    model_run: Optional[str],
) -> List[Tuple[str, str, Path]]:
    """Return (subject, condition, run_path) triples that hold fold dumps."""
    root = artifacts_dir / "models" / experiment
    out: List[Tuple[str, str, Path]] = []
    for sub in subjects:
        for cond in conditions:
            base = root / sub / cond / model_name / model_name_id
            if not base.exists():
                continue
            mr = model_run or _latest_model_run(base)
            if mr is None:
                continue
            run_path = base / mr
            if _fold_dumps(run_path):
                out.append((sub, cond, run_path))
    return out


def _latest_model_run(folder: Path) -> Optional[str]:
    ks = []
    for p in folder.iterdir():
        if p.is_dir() and p.name.startswith("model_"):
            try:
                ks.append((int(p.name.split("_")[-1]), p.name))
            except ValueError:
                continue
    return sorted(ks)[-1][1] if ks else None


# ---------------------------------------------------------------------------
# Best-beam-config selection from beam_sweep_<experiment>.csv
# ---------------------------------------------------------------------------


def _metric_maximize(metric: str) -> bool:
    """accuracy-like metrics are maximised; wer/cer/empty_rate minimised."""
    return "accuracy" in metric or "acc" in metric


def _default_select_metric(task: str, label_mode: Optional[str]) -> str:
    if task == "recognition":
        # word-level -> CER (WER degenerates on a single token), else WER.
        return "cer" if label_mode == "word" else "wer"
    return "cls_accuracy"


def _best_beam_from_csv(sweep_csv: Path, metric: str) -> DecodeCombo:
    """Pick the best BEAM row (greedy excluded) from a sweep results CSV."""
    rows = list(csv.DictReader(sweep_csv.open()))
    beam_rows = [r for r in rows if r.get("decode") == "beam"]
    if not beam_rows:
        raise ValueError(f"No beam rows in {sweep_csv}; run the sweep first.")
    if metric not in beam_rows[0]:
        raise ValueError(f"Metric '{metric}' not in {sweep_csv}. Columns: {list(beam_rows[0])}")
    best = (max if _metric_maximize(metric) else min)(beam_rows, key=lambda r: float(r[metric]))
    combo = DecodeCombo(
        "beam",
        beam_width=int(float(best["beam_width"])),
        temperature=float(best["temperature"]),
        blank_penalty=float(best["blank_penalty"]),
        length_bonus=float(best["length_bonus"]),
    )
    print(f"[SELECT] best beam by {'max' if _metric_maximize(metric) else 'min'} {metric}"
          f" = {best[metric]}  ->  {combo.label()}")
    return combo


# ---------------------------------------------------------------------------
# Per-fold scoring
# ---------------------------------------------------------------------------


def _score_fold(
    fold_file: Path, combo: DecodeCombo, task: str, jobs: int,
    dump_pred_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Decode one fold's dump with ``combo`` and return its metric dict."""
    seqs, labels, meta = load_dumps([fold_file])
    mapper = OfflineCTCMapper(meta)

    decoded_ids = _decode_fold(seqs, combo, mapper.blank_id, jobs)
    hyps = [mapper.ids_to_text(ids) for ids in decoded_ids]

    if task == "recognition":
        refs = mapper.label_int_to_texts(labels.tolist())
        m, _, _ = compute_wer_metrics(refs, hyps, verbose=False, vocabulary=mapper.word_vocabulary)
        out = {k: float(m[k]) for k in RECOGNITION_METRICS}
    else:
        preds = np.asarray(mapper.texts_to_label_int(hyps, allow_nearest=True), dtype=np.int64)
        try:
            import warnings

            from sklearn.metrics import balanced_accuracy_score

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                bal = float(balanced_accuracy_score(labels, preds))
        except Exception:
            bal = float("nan")
        out = {"balanced_accuracy": bal, "accuracy": float(np.mean(preds == labels))}

    if dump_pred_path is not None:
        _dump_predictions(dump_pred_path, task, mapper, labels, hyps)
    return out


def _dump_predictions(path: Path, task: str, mapper: OfflineCTCMapper, labels, hyps: List[str]) -> None:
    """Per-sample dump (.txt + .csv).

    Task-aware columns: recognition writes ``index, reference, recognition_output``;
    classification also writes ``classification_output`` (the predicted class text).
    """
    import csv as _csv

    task = str(task).lower().strip()
    is_cls = task == "classification"
    refs = mapper.label_int_to_texts(labels.tolist())
    n = len(refs)
    cls_texts = []

    if is_cls:
        cls_ints = mapper.texts_to_label_int(hyps, allow_nearest=True)
        cls_texts = [mapper.label_to_text_map.get(int(i), "<unmatched>") if int(i) >= 0 else "<unmatched>"
                     for i in cls_ints]
        header = ["index", "reference", "recognition_output", "classification_output"]
        rows = [[i, r, h, c] for i, (r, h, c) in enumerate(zip(refs, hyps, cls_texts))]
    else:
        header = ["index", "reference", "recognition_output"]
        rows = [[i, r, h] for i, (r, h) in enumerate(zip(refs, hyps))]

    correct_rec = sum(1 for r, h in zip(refs, hyps) if r == h)
    summary = [f"recognition={correct_rec}/{n}"]
    if is_cls:
        correct_cls = sum(1 for r, c in zip(refs, cls_texts) if r == c)
        summary.append(f"classification={correct_cls}/{n}")

    comments = [
        f"# CTC prediction dump (offline beam) | task={task} | n_samples={n}",
        "# reference = ground-truth text | recognition_output = free-character CTC decode with the swept beam config",
    ]
    if is_cls:
        comments.append("# classification_output = closed-set class text after the lexicon constraint (predicted class)")
    comments.append(f"# exact-match: {'  '.join(summary)}")

    path = Path(path)
    txt_lines = comments + ["\t".join(header)] + ["\t".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------


def _pct(vals: np.ndarray) -> str:
    return f"{np.round(np.mean(vals) * 100, 1)}±{np.round(np.std(vals) * 100, 1)}"


def _print_table(df: "pd.DataFrame", task: str, experiment: str, model_name: str,
                 model_name_id: str, condition: str, combo: DecodeCombo) -> None:
    """Print the per-subject table."""
    print("\n" + "=" * 110)
    print(
        f"Experiment: {experiment} | Model: {model_name} | model_name_id: {model_name_id} "
        f"| Condition: {condition} | Decode: {combo.label()}"
    )
    print("=" * 110)

    if task == "recognition":
        perc_cols = [f"{m}_mean_std_perc" for m in RECOGNITION_METRICS]
    else:
        perc_cols = ["mean_std_perc"]
    print_cols = ["subject", "model_run"] + [c for c in perc_cols if c in df.columns]
    print(df[print_cols].to_string())


def build_table(
    runs: List[Tuple[str, str, Path]],
    subjects: Sequence[str],
    condition: str,
    combo: DecodeCombo,
    task: str,
    model_name: str,
    model_name_id: str,
    model_run_tag: str,
    pool_subjects: bool,
    jobs: int,
    dump_predictions: bool,
) -> Optional["pd.DataFrame"]:
    metric_cols = RECOGNITION_METRICS if task == "recognition" else ["balanced_accuracy"]
    rows: List[dict] = []
    for sub in subjects:
        cond_runs = [rp for (s, c, rp) in runs if s == sub and c == condition]
        if not cond_runs:
            continue
        run_path = cond_runs[-1]
        per_fold: Dict[str, List[float]] = {m: [] for m in metric_cols}
        for fold_file in _fold_dumps(run_path):
            pred_path = (fold_file.with_name(fold_file.name.replace("_logprobs.npz", "_predictions_beam.txt"))
                         if dump_predictions and combo.decode == "beam" else None)
            fold_metrics = _score_fold(fold_file, combo, task, jobs, dump_pred_path=pred_path)
            for m in metric_cols:
                per_fold[m].append(fold_metrics[m])

        row: Dict[str, Any] = {
            "subject": sub, "condition": condition, "model_name": model_name,
            "model_name_id": model_name_id, "model_run": run_path.name, "run_path": str(run_path),
        }
        if task == "recognition":
            for m in metric_cols:
                v = np.asarray(per_fold[m], dtype=float)
                row[f"{m}_mean"] = float(v.mean())
                row[f"{m}_std"] = float(v.std())
                row[f"{m}_vals"] = json.dumps(v.tolist())
            for m in metric_cols:
                row[f"{m}_mean_std_perc"] = _pct(np.asarray(per_fold[m], dtype=float))
        else:
            v = np.asarray(per_fold["balanced_accuracy"], dtype=float)
            row["balanced_acc_mean"] = float(v.mean())
            row["balanced_acc_std"] = float(v.std())
            row["balanced_acc_vals"] = json.dumps(v.tolist())
            row["mean_std_perc"] = _pct(v)
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # "All" row = mean/std over per-subject means (skipped for a single pooled model).
    if not pool_subjects and len(df) > 1:
        all_row: Dict[str, Any]= {
            "subject": "All", "condition": condition, "model_name": model_name,
            "model_name_id": model_name_id, "model_run": model_run_tag, "run_path": "",
        }
        if task == "recognition":
            for m in metric_cols:
                means = df[f"{m}_mean"].to_numpy(dtype=float)
                all_row[f"{m}_mean"] = float(means.mean())
                all_row[f"{m}_std"] = float(means.std())
                all_row[f"{m}_vals"] = ""
            for m in metric_cols:
                means = df[f"{m}_mean"].to_numpy(dtype=float)
                all_row[f"{m}_mean_std_perc"] = f"{np.round(means.mean()*100, 2)}±{np.round(means.std()*100, 2)}"
        else:
            means = df["balanced_acc_mean"].to_numpy(dtype=float)
            all_row["balanced_acc_mean"] = float(means.mean())
            all_row["balanced_acc_std"] = float(means.std())
            all_row["balanced_acc_vals"] = ""
            all_row["mean_std_perc"] = f"{np.round(means.mean()*100, 2)}±{np.round(means.std()*100, 2)}"
        df = pd.concat([df, pd.DataFrame([all_row])], ignore_index=True)

    # Record the decode used so the table is self-describing.
    df["decode"] = combo.decode
    df["decode_config"] = combo.label()
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _detect_task(artifacts_dir: Path, experiment: str) -> str:
    """Read the 'task' field from any fold meta under the artifacts root."""
    for meta in (artifacts_dir / "models" / experiment).rglob("*logprobs*.meta.json"):
        return json.loads(meta.read_text()).get("task", "recognition")
    raise FileNotFoundError(f"No *logprobs*.meta.json under {artifacts_dir/'models'/experiment}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts_dir", type=Path, required=True,
                    help="Root that holds models/<exp>/... dumps and beam_sweep_<experiment>.csv.")
    ap.add_argument("--experiment", choices=["global", "inter_session"], required=True)
    ap.add_argument("--model_name", required=True, help="e.g. speechnet_transformer or speechnet")
    ap.add_argument("--model_name_id", required=True, help="e.g. w2000ms")
    ap.add_argument("--model_run", default=None, help="e.g. model_1 (default: latest per folder)")
    ap.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03", "S04", "S05", "S07"])
    ap.add_argument("--conditions", nargs="+", default=["silent", "vocalized"])
    ap.add_argument("--pool_subjects", action="store_true",
                    help="Score the pooled 'all_subjects' model instead of per-subject folders.")

    ap.add_argument("--sweep_csv", type=Path, default=None,
                    help="Sweep CSV to pick the best beam from (default: <artifacts_dir>/beam_sweep_<experiment>.csv).")
    ap.add_argument("--select_metric", default=None,
                    help="Metric to rank beam configs by (default: wer/cer for recognition, cls_accuracy for classification).")
    ap.add_argument("--beam_width", type=int, default=None, help="Override: use this beam config instead of the CSV best.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--blank_penalty", type=float, default=0.0)
    ap.add_argument("--length_bonus", type=float, default=0.0)

    ap.add_argument("--greedy", dest="greedy", action="store_true", default=False,
                    help="Also emit the greedy-decode table (off by default; beam-only otherwise).")
    ap.add_argument("--dump_predictions", action="store_true",
                    help="Also write per-sample *_predictions_beam.txt next to each fold dump.")
    ap.add_argument("--tables_dir", type=Path, default=None,
                    help="Output dir (default: <artifacts_dir>/tables_beam).")
    ap.add_argument("--jobs", type=int, default=4, help="Worker processes for decoding (Linux fork).")

    args = ap.parse_args()

    subjects = ["all_subjects"] if args.pool_subjects else args.subjects
    task = _detect_task(args.artifacts_dir, args.experiment)

    runs = _find_runs(
        args.artifacts_dir, args.experiment, subjects, args.conditions,
        args.model_name, args.model_name_id, args.model_run,
    )
    if not runs:
        print(f"[WARN] No fold dumps found under {args.artifacts_dir}/models/{args.experiment} "
              f"for model={args.model_name}, mid={args.model_name_id}, subjects={subjects}. Nothing to do.")
        return
    print(f"[SCAN] task={task} | {len(runs)} (subject,condition) run(s) with dumps.")

    label_mode = json.loads((_fold_dumps(runs[0][2])[0]).with_suffix(".meta.json").read_text()).get("label_mode")

    # Determine the beam config.
    if args.beam_width is not None:
        beam_combo = DecodeCombo("beam", args.beam_width, args.temperature, args.blank_penalty, args.length_bonus)
        print(f"[SELECT] using explicit beam config -> {beam_combo.label()}")
    else:
        sweep_csv = args.sweep_csv or (args.artifacts_dir / f"beam_sweep_{args.experiment}.csv")
        if not sweep_csv.exists():
            raise FileNotFoundError(
                f"{sweep_csv} not found. Run VII_beam_sweep.py first, or pass --beam_width to set the config explicitly.")
        metric = args.select_metric or _default_select_metric(task, label_mode)
        beam_combo = _best_beam_from_csv(sweep_csv, metric)

    tables_dir = args.tables_dir or (args.artifacts_dir / "tables_beam")
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_run_tag = args.model_run or "latest"

    combos = ([("greedy", DecodeCombo("greedy"))] if args.greedy else []) + [("beam", beam_combo)]

    for cond in args.conditions:
        for decode_tag, combo in combos:
            df = build_table(
                runs, subjects, cond, combo, task, args.model_name, args.model_name_id,
                model_run_tag, args.pool_subjects, args.jobs,
                dump_predictions=args.dump_predictions,
            )
            if df is None:
                print(f"[WARN] no data for condition={cond}, decode={decode_tag}.")
                continue
            out_csv = tables_dir / (
                f"{args.model_name}_{model_run_tag}_{cond}_{args.model_name_id}_{args.experiment}_{decode_tag}.csv")
            df.to_csv(out_csv, index=False)
            _print_table(df, task, args.experiment, args.model_name, args.model_name_id, cond, combo)
            print(f"\n[SAVED] {out_csv}")

    print("\nDONE. Beam tables in:", tables_dir)


if __name__ == "__main__":
    main()
