# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Dump the learned embeddings of an already trained fold, and project them.

The explainability hooks in ``offline_experiments/explainability.py`` run inside
the training loop, while this script reconstructs a fold from what the
run already wrote to disk:

* ``run_cfg.json`` gives the base and model configuration of the run;
* ``cv_summary.csv`` gives the exact row indices of the train, validation and
  test splits of every fold;
* ``<cv_mode>_fold_<k>.pt`` gives the weights.

The windows are reloaded from the dataset the run used, the splits are rebuilt
by index, the checkpoint is loaded into a new model, and the
activations of the requested layers are collected over train+val and test.

Two files are written under ``--out_dir``:

* ``embeddings_<layer>.npz`` with the full-dimensional activations, the class
  ids, the class names and the domain (trainval / test) of every sample. The
  thesis figures are drawn from these, so restyling a figure never re-runs a
  model;
* the PNG projections and the centroid-margin summary of
  ``run_explainability_for_fold``.

Example (model with Transformer encoder, global CV, vocalized condition):

    python3 offline_experiments/run_embedding_tsne.py \
        --run_dir artifacts_thesis/30_axis3_sequence_stage/transformer_classification/models/global/S01/vocalized/speechnet_transformer/w2000ms/model_1 \
        --fold 1 --layers pre_transformer pre_fc --methods tsne \
        --out_dir artifacts_thesis/embeddings/transformer_global_S01_vocalized
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from offline_experiments.Model_Master import Model_Master  # noqa: E402
from offline_experiments.explainability import (  # noqa: E402
    flatten_acts,
    grab_layer_over_dataloader,
    run_explainability_for_fold,
    stratified_downsample,
)
from offline_experiments.general_utils import (  # noqa: E402
    apply_datasets_normalization,
    check_data_directories,
    reset_all_seeds,
)
from utils.general_utils import load_all_h5files_from_folder  # noqa: E402


def _parse_idx(value) -> List[int]:
    """The split indices are stored in cv_summary.csv as a Python list literal."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    return [int(v) for v in ast.literal_eval(str(value))]


def load_run_dataframe(base_cfg: dict) -> pd.DataFrame:
    """Reload the windows of the run, in the order the training run saw them."""
    sub_id = base_cfg["data"]["subject_id"]
    window_ms = int(float(base_cfg["window"]["window_size_s"]) * 1000)
    dirs = check_data_directories(
        main_data_directory=Path(base_cfg["data"]["data_directory"]),
        all_subjects_models=isinstance(sub_id, list),
        sub_id=sub_id,
        condition=base_cfg["condition"],
        window_size_ms=window_ms,
        base_config=base_cfg,
    )
    df = pd.DataFrame()
    for d in dirs:
        df = pd.concat(
            (df, load_all_h5files_from_folder(d, key="wins_feats", print_statistics=False)),
            ignore_index=True,
        )
    return df.reset_index(drop=True)


def rebuild_fold(run_dir: Path, fold: int):
    """Rebuild the model and the splits of one fold of a finished run."""
    run_cfg = json.loads((run_dir / "run_cfg.json").read_text())
    base_cfg = deepcopy(run_cfg["base_cfg"])
    model_cfg = deepcopy(run_cfg["model_cfg"])

    summary = pd.read_csv(run_dir / "cv_summary.csv")
    rows = summary[summary["fold_num"] == fold]
    if rows.empty:
        raise SystemExit(f"fold {fold} not in {run_dir / 'cv_summary.csv'}")
    row = rows.iloc[0]
    cv_mode = str(row["cv_mode"])

    ckpt_path = run_dir / f"{cv_mode}_fold_{fold}.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint {ckpt_path}")

    df = load_run_dataframe(base_cfg)
    reset_all_seeds()

    master = Model_Master(base_cfg, model_cfg)
    master.df_train = df.loc[_parse_idx(row["train_idx"])].copy()
    master.df_val = df.loc[_parse_idx(row["val_idx"])].copy()
    master.df_test = df.loc[_parse_idx(row["test_idx"])].copy()

    apply_datasets_normalization(master, base_cfg, None)
    master.generate_training_labels()
    master.remap_all_datasets()
    master.register_model()

    model = master.model
    if not isinstance(model, torch.nn.Module):
        raise SystemExit(f"{run_dir} is not a torch run; there are no activations to project.")

    state = torch.load(ckpt_path, map_location=master.device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    print(f"[TSNE] loaded {ckpt_path} (epoch {state.get('epoch', '?')})")
    return master, base_cfg, cv_mode


def dump_embeddings(
    master,
    out_dir: Path,
    layers: Sequence[str],
    max_trainval: int,
    max_test: int,
    tag: str,
) -> None:
    """Write the full-dimensional activations of each layer to a .npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_name = {int(k): str(v) for k, v in (master.train_label_map or {}).items()}
    data_cols = master.data_col_to_consider + ["Label_train"]

    dtr = stratified_downsample(pd.concat([master.df_train, master.df_val]), max_trainval).copy()
    dtr["domain"] = "trainval"
    dte = stratified_downsample(master.df_test, max_test).copy()
    dte["domain"] = "test"
    combined = pd.concat([dtr, dte], ignore_index=True)
    domains = combined["domain"].to_numpy()

    loader = master.trainer_manager.create_dataloader_from_df(combined[data_cols], shuffle=False)

    for layer in layers:
        try:
            acts, labels = grab_layer_over_dataloader(master.model, layer, loader)
        except KeyError:
            print(f"[TSNE] layer '{layer}' not in this model; skipping.")
            continue
        X = flatten_acts(acts)
        y = labels.numpy().reshape(-1).astype(int)
        path = out_dir / f"embeddings_{layer}.npz"
        np.savez_compressed(
            path,
            X=X,
            y=y,
            domain=domains,
            class_names=np.array([id_to_name.get(int(c), str(c)) for c in sorted(set(y))]),
            class_ids=np.array(sorted(set(y))),
            layer=layer,
            tag=tag,
        )
        print(f"[TSNE] wrote {path}  X={X.shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True, help="model_<k> directory of a finished run")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--layers", nargs="+", default=["pre_transformer", "pre_bilstm", "pre_fc"])
    ap.add_argument("--methods", nargs="+", default=["tsne"], help="projections to plot (tsne/umap/pca)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_trainval", type=int, default=6000)
    ap.add_argument("--max_test", type=int, default=2500)
    ap.add_argument("--no_plots", action="store_true", help="only dump the .npz files")
    args = ap.parse_args()

    master, base_cfg, cv_mode = rebuild_fold(args.run_dir, args.fold)
    tag = "|".join(
        [
            str(base_cfg["data"]["subject_id"]),
            str(base_cfg["condition"]),
            cv_mode,
            f"fold{args.fold}",
            str(master.model_config["model"]["name"]),
        ]
    )

    dump_embeddings(master, args.out_dir, args.layers, args.max_trainval, args.max_test, tag)

    if not args.no_plots:
        run_explainability_for_fold(
            master,
            pd.concat([master.df_train, master.df_val]),
            master.df_test,
            args.out_dir,
            layers=args.layers,
            methods=args.methods,
            max_trainval=args.max_trainval,
            max_test=args.max_test,
        )
    print(f"[TSNE] done: {args.out_dir}")


if __name__ == "__main__":
    main()
