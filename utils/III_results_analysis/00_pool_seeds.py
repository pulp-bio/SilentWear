# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
00_pool_seeds.py

Pool experiments run with several seeds (reproduce_paper_scripts/40_run_seed_job.sh)
into one artifacts tree with the same layout as a single-seed run, so that the
analysis scripts I-IV can be applied to it unchanged.

Modes
  pool    (default) write <out_dir>/models/... where every per-fold / per-session value
          is the mean over seeds of the same fold / session:
            - metrics (balanced accuracy, accuracy, F1, ...): mean over seeds
            - confusion matrices (row-normalised): element-wise mean over seeds; since a
              fold's test set is identical across seeds, this equals normalising the
              summed counts
            - columns that must identify the same fold/session (fold_num, test_session,
              test_batch, test_idx, target, bar, ...) are checked to be identical
            - per-window predictions and training indices (y_pred, train_idx, val_idx)
              differ across seeds and are dropped
          JSON configs are copied from the first seed and annotated with the pooled seeds.
  spread  after running the analysis scripts on every seed directory, write
          <out_dir>/seed_spread/<table>.csv with the mean and std *across seeds* of every
          numeric (or array) value of each table in <seed_dir>/tables/.

Examples:
  python utils/III_results_analysis/00_pool_seeds.py \
    --seed_dirs artifacts/seed_42 artifacts/seed_52 artifacts/seed_62 \
    --out_dir artifacts/seeds_pooled
  python utils/III_results_analysis/00_pool_seeds.py --mode spread \
    --seed_dirs artifacts/seed_42 artifacts/seed_52 artifacts/seed_62 \
    --out_dir artifacts/seeds_pooled
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Columns that identify a fold / session / setting: must be identical across seeds
ID_COLS = {
    "cv_mode",
    "fold_num",
    "test_batch",
    "test_session",
    "test_idx",
    "y_true",
    "subject",
    "condition",
    "base_model",
    "zero_shot_test_batch",
    "num_prev_ft_rounds",
    "target",
    "n_pretrain_subjects",
    "pretrain_subjects",
    "bar",
    "unit",
    "train_sessions",
    "window_ms",
}
# Scalar identifiers used to align rows across seeds (in this order)
KEY_COLS = [
    "subject",
    "condition",
    "target",
    "n_pretrain_subjects",
    "pretrain_subjects",
    "bar",
    "unit",
    "train_sessions",
    "fold_num",
    "base_model",
    "test_session",
    "test_batch",
    "zero_shot_test_batch",
    "num_prev_ft_rounds",
]
# Seed-dependent columns without a meaningful average
DROP_COLS = {"y_pred", "train_idx", "val_idx", "checkpoint"}


# ------------------------- helpers -------------------------
def _parse_matrix(v) -> Optional[np.ndarray]:
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            v = ast.literal_eval(v)
    return np.asarray(v, dtype=float)


def _is_matrix_col(name: str) -> bool:
    return name == "confusion_matrix" or name.endswith("_cm") or name.startswith("cm_")


def _parse_array(v) -> Optional[np.ndarray]:
    """Parse numpy-printed arrays such as '[76.4 86.8 85.5]' or JSON lists."""
    if not isinstance(v, str) or not v.strip().startswith("["):
        return None
    txt = v.strip()
    try:
        return np.asarray(json.loads(txt), dtype=float)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        return np.asarray(txt.strip("[]").replace(",", " ").split(), dtype=float)
    except ValueError:
        return None


def _same(values: List) -> bool:
    first = values[0]
    for v in values[1:]:
        if isinstance(first, float) and isinstance(v, float) and np.isnan(first) and np.isnan(v):
            continue
        if v != first:
            return False
    return True


# ------------------------- pool -------------------------
def _align(dfs: List[pd.DataFrame], paths: List[Path]) -> List[pd.DataFrame]:
    """Sort every seed's table by its identifier columns so that rows correspond."""
    ref = dfs[0]
    for p, d in zip(paths[1:], dfs[1:]):
        if list(d.columns) != list(ref.columns):
            raise ValueError(f"{p}: columns differ from {paths[0]}")
        if len(d) != len(ref):
            raise ValueError(f"{p}: {len(d)} rows vs {len(ref)} in {paths[0]}")
    keys = [c for c in KEY_COLS if c in ref.columns]
    if not keys:
        return [d.reset_index(drop=True) for d in dfs]
    out = []
    for p, d in zip(paths, dfs):
        if d.duplicated(subset=keys).any():
            raise ValueError(f"{p}: duplicate rows for identifier columns {keys}")
        out.append(d.sort_values(keys, kind="stable").reset_index(drop=True))
    return out


def pool_csv(paths: List[Path], seeds: List[int]) -> pd.DataFrame:
    dfs = _align([pd.read_csv(p, dtype={"pretrain_subjects": str, "train_sessions": str}) for p in paths], paths)
    ref = dfs[0]

    out = {}
    for col in ref.columns:
        if col in DROP_COLS:
            continue
        cols = [d[col].tolist() for d in dfs]
        if all(_same([c[i] for c in cols]) for i in range(len(ref))):
            out[col] = cols[0]
        elif col in ID_COLS:
            raise ValueError(f"{paths[0].name}: identifier column '{col}' differs across seeds")
        elif "seed" in col:
            out[col] = ["|".join(str(c[i]) for c in cols) for i in range(len(ref))]
        elif _is_matrix_col(col):
            pooled = []
            for i in range(len(ref)):
                mats = [_parse_matrix(c[i]) for c in cols]
                pooled.append(
                    None if any(m is None for m in mats) else json.dumps(np.mean(mats, axis=0).tolist())
                )
            out[col] = pooled
        elif all(pd.api.types.is_numeric_dtype(d[col]) for d in dfs):
            out[col] = np.mean(np.vstack([d[col].to_numpy(dtype=float) for d in dfs]), axis=0)
        else:
            out[col] = ["|".join(str(c[i]) for c in cols) for i in range(len(ref))]
    return pd.DataFrame(out)


def pool_json(paths: List[Path], seeds: List[int]):
    obj = json.loads(paths[0].read_text())
    if isinstance(obj, dict):
        obj["pooled_from_seeds"] = seeds
        if "seeds" in obj:
            obj["seeds"] = {"pooled_run_seeds": seeds}
    return obj


def seed_of(d: Path) -> int:
    name = d.name
    if not name.startswith("seed_"):
        raise ValueError(f"seed directory must be named seed_<s>: {d}")
    return int(name.split("_", 1)[1])


def check_seed_dirs(seed_dirs: List[Path], sub: str) -> None:
    """Fail loudly if a seed directory (or its <sub> folder) is missing or empty."""
    for d in seed_dirs:
        if not (d / sub).is_dir() or not any((d / sub).iterdir()):
            raise SystemExit(
                f"[ERROR] no '{sub}' results in {d} (resolved: {d.resolve()}).\n"
                f"        Relative paths are resolved from the current directory ({Path.cwd()});"
                f" run from the repository root or pass absolute paths."
            )


def run_pool(seed_dirs: List[Path], out_dir: Path, allow_incomplete: bool) -> None:
    check_seed_dirs(seed_dirs, "models")
    seeds = [seed_of(d) for d in seed_dirs]
    rels = sorted(
        p.relative_to(seed_dirs[0])
        for p in (seed_dirs[0] / "models").rglob("*")
        if p.is_file() and p.suffix in (".csv", ".json")
    )
    missing = {r: [s for s, d in zip(seeds, seed_dirs) if not (d / r).exists()] for r in rels}
    missing = {r: m for r, m in missing.items() if m}
    if missing:
        msg = "\n".join(f"  {r} (missing for seeds {m})" for r, m in list(missing.items())[:20])
        if not allow_incomplete:
            raise SystemExit(f"[POOL] {len(missing)} files not present for every seed:\n{msg}")
        print(f"[POOL] skipping {len(missing)} incomplete files (--allow_incomplete):\n{msg}")

    n = 0
    for r in rels:
        if r in missing:
            continue
        srcs = [d / r for d in seed_dirs]
        dst = out_dir / r
        if r.suffix == ".csv":
            try:
                pooled = pool_csv(srcs, seeds)
            except ValueError as e:
                if not allow_incomplete:
                    raise
                print(f"[POOL] skipping {r}: {e}")
                missing[r] = ["mismatch"]
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            pooled.to_csv(dst, index=False)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(pool_json(srcs, seeds), indent=4, sort_keys=True))
        n += 1

    manifest = seed_dirs[0].parent / "runs_manifest.csv"
    runs = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()
    if not runs.empty:
        runs = runs[runs["seed"].isin(seeds)]
    info = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "seed_dirs": [str(d) for d in seed_dirs],
        "files_pooled": n,
        "files_skipped_incomplete": [str(r) for r in missing],
        "pooling": "per fold/session: mean over seeds of metrics and confusion matrices",
        "runs": runs.to_dict(orient="records"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pooled_manifest.json").write_text(json.dumps(info, indent=4))
    print(f"[POOL] {n} files pooled over seeds {seeds} -> {out_dir}/models")


# ------------------------- spread -------------------------
# Row identifiers of the analysis tables (checked, not sorted: tables are written in a fixed order)
TABLE_ID_COLS = [
    "subject",
    "subject_id",
    "target",
    "condition",
    "bar",
    "n_pretrain_subjects",
    "model_name",
    "model_name_id",
    "win_size_ms",
]


def spread_table(paths: List[Path], seeds: List[int]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in paths]
    ref = dfs[0]
    ids = [c for c in TABLE_ID_COLS if c in ref.columns]
    for p, d in zip(paths[1:], dfs[1:]):
        if list(d.columns) != list(ref.columns) or len(d) != len(ref):
            raise ValueError(f"{p}: columns/rows differ from {paths[0]}")
        for c in ids:
            if not d[c].astype(str).equals(ref[c].astype(str)):
                raise ValueError(f"{p}: row identifier '{c}' differs from {paths[0]}")

    out = {}
    for col in ref.columns:
        cols = [d[col].tolist() for d in dfs]
        if all(pd.api.types.is_numeric_dtype(d[col]) for d in dfs):
            vals = np.vstack([d[col].to_numpy(dtype=float) for d in dfs])
            if np.allclose(vals, vals[0], equal_nan=True):
                out[col] = vals[0]
            else:
                out[f"{col}_seed_mean"] = vals.mean(axis=0)
                out[f"{col}_seed_std"] = vals.std(axis=0)
            continue
        arrays = [[_parse_array(v) for v in c] for c in cols]
        if all(a is not None for c in arrays for a in c):
            stacked = [np.stack([c[i] for c in arrays]) for i in range(len(ref))]
            out[f"{col}_seed_mean"] = [json.dumps(np.round(s.mean(axis=0), 6).tolist()) for s in stacked]
            out[f"{col}_seed_std"] = [json.dumps(np.round(s.std(axis=0), 6).tolist()) for s in stacked]
        elif all(_same([c[i] for c in cols]) for i in range(len(ref))):
            out[col] = cols[0]
        else:
            out[f"{col}_per_seed"] = [" | ".join(str(c[i]) for c in cols) for i in range(len(ref))]
    df = pd.DataFrame(out)
    df.insert(0, "seeds", "|".join(map(str, seeds)))
    return df


def run_spread(seed_dirs: List[Path], out_dir: Path) -> None:
    check_seed_dirs(seed_dirs, "tables")
    seeds = [seed_of(d) for d in seed_dirs]
    names = sorted(p.name for p in (seed_dirs[0] / "tables").glob("*.csv"))
    dst_dir = out_dir / "seed_spread"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in names:
        paths = [d / "tables" / name for d in seed_dirs]
        if not all(p.exists() for p in paths):
            print(f"[SPREAD] skip {name}: not present for every seed")
            continue
        spread_table(paths, seeds).to_csv(dst_dir / name, index=False)
        n += 1
    print(f"[SPREAD] {n} tables -> {dst_dir}")


# ------------------------- CLI main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Pool multi-seed runs / compute seed spread")
    parser.add_argument("--mode", choices=["pool", "spread"], default="pool")
    parser.add_argument("--seed_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--allow_incomplete",
        action="store_true",
        help="pool only files present for every seed (default: fail if any is missing)",
    )
    args = parser.parse_args()

    if len(args.seed_dirs) < 2:
        sys.exit("need at least two seed directories")
    if args.mode == "pool":
        run_pool(args.seed_dirs, args.out_dir, args.allow_incomplete)
    else:
        run_spread(args.seed_dirs, args.out_dir)


if __name__ == "__main__":
    main()
