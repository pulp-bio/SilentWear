#!/usr/bin/env python3

# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Aggregate the sentence results into the summary table the report quotes.

Usage:  
      python3 utils/III_results_analysis/aggregate_rest_sentence_results.py
      python3 ... --root artifacts_beam_sweep_no_rest
      python3 ... --csv rest_sentence_results.csv
      python3 ... --window w2400ms
      python3 ... --per-subject
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "artifacts_beam_sweep")

ARCHITECTURES = ("bilstm", "transformer")
DEFAULT_WINDOW = "w2000ms"
TASKS = {
    "classification": dict(greedy=["balanced_accuracy"],
                           beam=["balanced_acc_mean"]),
    "recognition": dict(greedy=["balanced_wer", "balanced_cer", "wer", "cer"],
                        beam=["balanced_wer_mean", "balanced_cer_mean",
                              "wer_mean", "cer_mean",
                              "balanced_vocab_wer_mean"]),
}


def greedy_figures(run_dir, metrics, window=DEFAULT_WINDOW):
    """Mean over participants of the per-fold mean, per protocol and condition."""
    acc = {}
    pattern = os.path.join(run_dir, "models", "*", "*", "*", "*", window, "*",
                           "cv_summary.csv")
    for path in glob.glob(pattern):
        parts = path.split(os.sep)
        i = parts.index("models")
        key = (parts[i + 1], parts[i + 3])          # protocol, condition
        d = pd.read_csv(path)
        for m in metrics:
            if m in d.columns:
                acc.setdefault((key, m), []).append(d[m].mean())
    return {k: 100.0 * float(np.mean(v)) for k, v in acc.items()}


def beam_figures(run_dir, columns, window=DEFAULT_WINDOW):
    """The ``All`` row of every per-condition table the sweep selected."""
    out = {}
    for path in glob.glob(os.path.join(run_dir, "tables_beam",
                                       f"*_{window}_*_beam.csv")):
        name = os.path.basename(path)
        condition = "silent" if "silent" in name else "vocalized"
        protocol = "inter_session" if "inter_session" in name else "global"
        d = pd.read_csv(path)
        row = d[d["subject"] == "All"]
        if row.empty:
            continue
        for c in columns:
            if c in row.columns:
                out[((protocol, condition), c)] = 100.0 * float(row[c].iloc[0])
        out[((protocol, condition), "decode_config")] = row["decode_config"].iloc[0]
    return out


def per_subject(run_dir, column, window=DEFAULT_WINDOW):
    """Per-participant beam figures."""
    rows = []
    for path in glob.glob(os.path.join(run_dir, "tables_beam",
                                       f"*_{window}_*_beam.csv")):
        name = os.path.basename(path)
        condition = "silent" if "silent" in name else "vocalized"
        protocol = "inter_session" if "inter_session" in name else "global"
        d = pd.read_csv(path)
        if column not in d.columns:
            continue
        for _, r in d.iterrows():
            rows.append(dict(protocol=protocol, condition=condition,
                             subject=r["subject"],
                             value=100.0 * float(r[column])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--csv", default=None, help="write the table here as well")
    ap.add_argument("--window", default=DEFAULT_WINDOW,
                    help="window token both sides are read at, as it appears in "
                         "the checkpoint path and in the table filename")
    ap.add_argument("--per-subject", action="store_true",
                    help="print the per-participant figures instead of the "
                         "pooled ones, which is what the per-participant tables need; "
                         "ignores --csv")
    args = ap.parse_args()

    if args.per_subject:
        for arch in ARCHITECTURES:
            for task, column in (("classification", "balanced_acc_mean"),
                                 ("recognition", "balanced_wer_mean")):
                run = os.path.join(args.root,
                                   f"{arch}_stft_ctc_{task}_beam_sweep")
                if not os.path.isdir(run):
                    continue
                d = per_subject(run, column, args.window)
                if d.empty:
                    continue
                wide = d.pivot_table(index=["protocol", "condition"],
                                     columns="subject", values="value")
                print(f"\n{arch} {task} ({column}, %)")
                print(wide.round(1).to_string())
        return

    records = []
    for arch in ARCHITECTURES:
        for task, spec in TASKS.items():
            run = os.path.join(args.root,
                               f"{arch}_stft_ctc_{task}_beam_sweep")
            if not os.path.isdir(run):
                print(f"missing: {run}")
                continue
            g = greedy_figures(run, spec["greedy"], args.window)
            b = beam_figures(run, spec["beam"], args.window)
            keys = sorted({k[0] for k in g} | {k[0] for k in b})
            for protocol, condition in keys:
                rec: dict[str, object] = dict(architecture=arch, task=task,
                                              protocol=protocol,
                                              condition=condition)
                for m in spec["greedy"]:
                    rec[f"greedy_{m}"] = round(
                        g.get(((protocol, condition), m), float("nan")), 2)
                for c in spec["beam"]:
                    rec[f"beam_{c}"] = round(
                        b.get(((protocol, condition), c), float("nan")), 2)
                rec["beam_config"] = b.get(((protocol, condition),
                                            "decode_config"), "")
                records.append(rec)

    table = pd.DataFrame(records)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwritten to {args.csv}")


if __name__ == "__main__":
    main()
