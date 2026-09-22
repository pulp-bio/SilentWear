# Copyright Carola Bonamico 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Model explainability on learned embeddings.

For a trained model, this module:
  1. grabs the activations of a named layer via forward hooks over a
     dataloader,
  2. reduces them to 2D with UMAP / t-SNE / PCA,
  3. quantifies separability and train->test shift with a centroid-margin analysis
     computed on the full embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

LAYER_ALIASES: Dict[str, Tuple[str, bool]] = {
    "pre_bilstm": ("rnn", True),                # input to the BiLSTM
    "pre_transformer": ("transformer", True),   # input to the Transformer encoder
    "pre_fc": ("fc", True),                     # input to the final fully-connected layer
}


# ---------------------------------------------------------------------------
# Activation extraction (forward hooks)
# ---------------------------------------------------------------------------


def grab_layer_over_dataloader(
    model: torch.nn.Module, layer: str, loader, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect (activations, labels) for a layer over a dataloader."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if layer in LAYER_ALIASES:
        name, hook_input = LAYER_ALIASES[layer]
    else:
        name, hook_input = layer, False

    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Layer '{name}' not found in model.")
    module = modules[name]

    acts: Dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        acts["out"] = out.detach()

    def pre_hook(_m, inp):
        x = inp[0] if isinstance(inp, (tuple, list)) else inp
        acts["out"] = x.detach()

    handle = (
        module.register_forward_pre_hook(pre_hook)
        if hook_input
        else module.register_forward_hook(fwd_hook)
    )

    all_acts, all_labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            acts.pop("out", None)
            _ = model(x.to(device))
            if "out" not in acts:
                handle.remove()
                raise RuntimeError(f"No activations captured for layer '{name}'.")
            all_acts.append(acts["out"].cpu())
            all_labels.append(y.detach().cpu())

    handle.remove()
    if not all_acts:
        raise RuntimeError("Loader produced no batches.")
    return torch.cat(all_acts, dim=0), torch.cat(all_labels, dim=0)


def flatten_acts(acts: torch.Tensor) -> np.ndarray:
    """Flatten activations to (N, D)."""
    return acts.flatten(start_dim=1).numpy().astype(np.float32)


def project(X: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    """Reduce X to 2D with the requested method (umap / tsne / pca)."""
    if not np.isfinite(X).all():
        raise ValueError("Projection input contains NaN/Inf.")

    if method == "umap":
        import umap
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            reducer = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.1, metric="euclidean",
                random_state=random_state, n_jobs=1
            )
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = float(min(30.0, max(5.0, (len(X) - 1) / 3.0)))
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown projection method '{method}'.")

    return np.asarray(reducer.fit_transform(X))


# ---------------------------------------------------------------------------
# Quantitative separability
# ---------------------------------------------------------------------------


def compute_centroid_margin_analysis(X: np.ndarray, y: np.ndarray, domains: np.ndarray) -> dict:
    """Centroids are built from trainval only; distances are measured for both
    domains. Returns per-sample margins and a summary dict.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1).astype(int)
    domains = np.asarray(domains).reshape(-1)

    if len(X) != len(y) or len(X) != len(domains):
        raise ValueError("X, y, and domains must have the same length.")

    train_mask = domains == "trainval"
    test_mask = domains == "test"
    if not np.any(train_mask):
        raise ValueError("No trainval samples found.")

    classes = sorted(np.unique(y[train_mask]))
    centroids = {c: X[train_mask & (y == c)].mean(axis=0) for c in classes if np.any(train_mask & (y == c))}

    d_true = np.full(len(y), np.nan)
    margin = np.full(len(y), np.nan)
    nearest_centroid_class = np.full(len(y), -1, dtype=int)

    for i in range(len(y)):
        if y[i] not in centroids:
            continue
        dists = {c: float(np.linalg.norm(X[i] - mu)) for c, mu in centroids.items()}
        d_true[i] = dists[y[i]]
        
        nearest_centroid_class[i] = min(dists, key=dists.__getitem__)
        
        wrong = {c: d for c, d in dists.items() if c != y[i]}
        if wrong:
            margin[i] = min(wrong.values()) - dists[y[i]]

    summary = {
        "train_margin_mean": float(np.nanmean(margin[train_mask])) if np.any(train_mask) else np.nan,
        "test_margin_mean": float(np.nanmean(margin[test_mask])) if np.any(test_mask) else np.nan,
        "train_margin_negative_frac": float(np.mean(margin[train_mask] < 0)) if np.any(train_mask) else np.nan,
        "test_margin_negative_frac": float(np.mean(margin[test_mask] < 0)) if np.any(test_mask) else np.nan,
        "train_nearest_centroid_acc": float(np.mean(nearest_centroid_class[train_mask] == y[train_mask])) if np.any(train_mask) else np.nan,
        "test_nearest_centroid_acc": float(np.mean(nearest_centroid_class[test_mask] == y[test_mask])) if np.any(test_mask) else np.nan,
    }
    return {"margin": margin, "nearest_centroid_class": nearest_centroid_class, "summary": summary}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _class_cmap(present_ids: List[int]):
    base = plt.get_cmap("tab10") if len(present_ids) <= 10 else plt.get_cmap("tab20")
    colors = [base(i % base.N) for i in range(len(present_ids))]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(present_ids) + 0.5, 1.0), cmap.N)
    return colors, cmap, norm


def plot_trainval_test_embeddings(
    Z: np.ndarray, y: np.ndarray, domains: np.ndarray, id_to_name: Dict[int, str],
    method: str, title: str, fig_save_path: Path,
) -> None:
    """Two panels (trainval, test) of the 2D embedding, coloured by class."""
    present_ids = sorted(set(int(v) for v in y))
    id_to_idx = {c: i for i, c in enumerate(present_ids)}
    colors, cmap, norm = _class_cmap(present_ids)
    y_idx = np.array([id_to_idx[int(v)] for v in y])

    fig, axs = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (dom, marker, size) in zip(axs, [("trainval", "o", 15), ("test", "o", 15)]):
        mask = domains == dom
        ax.scatter(Z[mask, 0], Z[mask, 1], c=y_idx[mask], cmap=cmap, norm=norm,
                   s=size, alpha=0.8, marker=marker, linewidths=0.0)
        ax.set_title(dom)
        ax.set_xlabel(f"{method.upper()}-1")
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel(f"{method.upper()}-2")

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=7,
                      markerfacecolor=colors[i], markeredgecolor="none",
                      label=id_to_name.get(c, str(c))) for i, c in enumerate(present_ids)]
    
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.11), ncol=min(len(handles), 8),
               frameon=False, title="Classes")
    fig.suptitle(title)
    
    fig.tight_layout(rect=(0.0, 0.22, 1.0, 0.95))
    
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_test_to_train_centroid_heatmap(
    X: np.ndarray, y: np.ndarray, domains: np.ndarray, id_to_name: Dict[int, str],
    title: str, fig_save_path: Path, normalize_features: bool = True,
) -> None:
    """Heatmap of mean distance from each true test class to every trainval centroid."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(int)
    domains = np.asarray(domains).reshape(-1)
    if normalize_features:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    train_mask = domains == "trainval"
    test_mask = domains == "test"
    classes = sorted(set(np.unique(y[train_mask])).intersection(set(np.unique(y[test_mask]))))
    if not classes:
        return

    centroids = {c: X[train_mask & (y == c)].mean(axis=0) for c in classes}
    M = np.full((len(classes), len(classes)), np.nan, dtype=np.float32)
    for i, true_cls in enumerate(classes):
        X_test_cls = X[test_mask & (y == true_cls)]
        if len(X_test_cls) == 0:
            continue
        for j, cen_cls in enumerate(classes):
            M[i, j] = float(np.linalg.norm(X_test_cls - centroids[cen_cls], axis=1).mean())

    labels = [id_to_name.get(c, str(c)) for c in classes]
    fig, ax = plt.subplots(figsize=(1.0 + len(classes), 1.0 + len(classes)))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Trainval class centroid")
    ax.set_ylabel("True test class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean distance")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def stratified_downsample(
    df: pd.DataFrame, max_samples: int, label_col: str = "Label_train",
    session_col: str = "session_id", random_state: int = 42,
) -> pd.DataFrame:
    """Downsample preserving (session x label) proportions when possible."""
    if len(df) <= max_samples:
        return df.copy()
    frac = max_samples / len(df)
    strat_cols = [c for c in (session_col, label_col) if c in df.columns]
    if not strat_cols:
        return df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
    sampled = (
        df.groupby(strat_cols, group_keys=False)
        .apply(lambda x: x.sample(n=max(1, int(round(len(x) * frac))), random_state=random_state))
    )
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=random_state)
    return sampled.reset_index(drop=True)


def run_explainability_for_fold(
    model_master, df_trainval: pd.DataFrame, df_test: pd.DataFrame, out_dir: Path,
    layers: Sequence[str], methods: Sequence[str], max_trainval: int = 8000, max_test: int = 2000,
) -> None:
    """Extract layer embeddings, project them (UMAP/t-SNE/PCA), and quantify
    separability via centroid-margin analysis.
    """
    if getattr(model_master, "kind", None) != "dl":
        return
    if df_trainval is None or df_trainval.empty or df_test is None or df_test.empty:
        print("[EXPLAIN] Empty trainval/test set; skipping.")
        return

    out_dir = Path(out_dir)
    id_to_name = {int(k): str(v) for k, v in (model_master.train_label_map or {}).items()}
    data_cols = model_master.data_col_to_consider + ["Label_train"]

    dtr = stratified_downsample(df_trainval, max_trainval).copy()
    dtr["domain"] = "trainval"
    dte = stratified_downsample(df_test, max_test).copy()
    dte["domain"] = "test"
    combined = pd.concat([dtr, dte], ignore_index=True)
    domains = combined["domain"].to_numpy()

    loader = model_master.trainer_manager.create_dataloader_from_df(
        combined[data_cols], shuffle=False
    )

    margin_rows = []
    for layer in layers:
        try:
            acts, labels = grab_layer_over_dataloader(model_master.model, layer, loader)
        except KeyError:
            print(f"[EXPLAIN] layer '{layer}' not in model; skipping.")
            continue

        X = flatten_acts(acts)
        y = labels.numpy().reshape(-1).astype(int)
        print(f"[EXPLAIN] layer '{layer}' embeddings: {X.shape}")

        # Quantitative separability on the full embeddings.
        analysis = compute_centroid_margin_analysis(X, y, domains)
        margin_rows.append({"layer": layer, "n_features": int(X.shape[1]), **analysis["summary"]})
        plot_test_to_train_centroid_heatmap(
            X, y, domains, id_to_name, title=f"{layer} - test-to-train centroid distance",
            fig_save_path=out_dir / layer / "centroid_distance_heatmap.png",
        )

        # 2D visualization per method.
        for method in methods:
            Z = project(X, method)
            plot_trainval_test_embeddings(
                Z, y, domains, id_to_name, method, title=f"{layer} | {method.upper()}",
                fig_save_path=out_dir / layer / f"{method}_trainval_test.png",
            )

    if margin_rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(margin_rows).to_csv(out_dir / "centroid_margin_summary.csv", index=False)
        print(f"[EXPLAIN] Saved centroid-margin summary: {out_dir / 'centroid_margin_summary.csv'}")


def maybe_run_explainability(
    model_master, base_config: dict, df_trainval: pd.DataFrame, df_test: pd.DataFrame, out_dir: Path,
) -> None:
    """Run explainability when enabled via experiment.explainability."""
    cfg = base_config.get("experiment", {}).get("explainability", {}) or {}
    if not cfg.get("enabled"):
        return
    methods = cfg.get("methods", ["umap", "tsne", "pca"])
    layers = cfg.get("layers", ["pre_bilstm", "pre_fc"])
    run_explainability_for_fold(model_master, df_trainval, df_test, out_dir, layers, methods)