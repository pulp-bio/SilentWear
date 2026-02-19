# Copyright 2026 Giusy Spacone
# Copyright 2026 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Script Responsible to Extract UMAP Projection
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

import umap
from sklearn.preprocessing import StandardScaler


@dataclass
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    scale_features: bool = True
    max_points: Optional[int] = None


class UMAP_Projection_Extractor:
    def __init__(
        self,
        config: Optional[UMAPConfig] = None,
        out_dir: Optional[Union[str, Path]] = None,
        subject_id: Optional[str] = None,
        dpi: int = 160,
    ) -> None:
        self.config = config or UMAPConfig()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.subject_id = subject_id
        self.dpi = dpi

    # ----------------------------
    # Public API
    # ----------------------------
    def plot_per_batch(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        condition: Optional[str] = None,
        color_by: str = "Label_str",
        title_prefix: str = "",
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> None:
        df_use = self._filter_condition(df, condition)
        batches = self._sorted_unique(df_use, "batch_id")

        if len(batches) == 0:
            print("[UMAP] No batches found for the requested condition.")
            return

        # Stable colors across all batches (within this call)
        global_label_order = self._global_label_order(df_use, color_by)

        for b in batches:
            dfi = df_use[df_use["batch_id"].astype(str) == str(b)].copy()
            emb, dfi = self._compute_embedding(dfi, feature_cols)

            title = self._mk_title(title_prefix, condition, f"Batch {b}")
            fig, _, _ = self._scatter_2d(
                dfi,
                emb,
                color_by=color_by,
                title=title,
                label_order=global_label_order,
                add_legend=True,
            )

            if save_name or self.out_dir:
                fname = save_name or self._default_filename(
                    scope="per_batch", condition=condition, extra=f"batch_{b}"
                )
                self._save_fig(fig, fname)

            if show:
                plt.show()
            else:
                plt.close(fig)

    def plot_per_session(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        session_id: Optional[str] = None,
        condition: Optional[str] = None,
        color_by: str = "Label_str",
        title_prefix: str = "",
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> None:
        print("\n\nPER SESSION PLOTTING")

        base_df = self._filter_condition(df, condition)

        # If a specific session is requested, keep base_df narrowed so label colors are stable
        # between with-rest and no-rest for exactly that session.
        if session_id is not None:
            base_df = base_df[base_df["session_id"].astype(str) == str(session_id)].copy()

        sessions = self._sorted_unique(base_df, "session_id")
        print("GOT sessions:", sessions)

        if len(sessions) == 0:
            print("[UMAP] No sessions found for the requested selection.")
            return

        # NOTE: We loop inc_rest and reuse the SAME global label order per session,
        # so colors stay identical between "with-rest" and "no-rest".
        for sess in sessions:
            # Build label order from WITH-REST data for this session (superset)
            dfs_all = base_df[base_df["session_id"].astype(str) == str(sess)].copy()
            if len(dfs_all) == 0:
                continue
            session_label_order = self._global_label_order(dfs_all, color_by)

            for inc_rest in [True, False]:
                rest_tag = "with-rest" if inc_rest else "no-rest"

                dfs = dfs_all.copy()
                if not inc_rest:
                    print("Analysis without rest")
                    dfs = dfs[dfs[color_by].astype(str) != "rest"].copy()

                print("plotting session", sess, "|", rest_tag)
                print("Total data point in current session", len(dfs))

                batches = self._sorted_unique(dfs, "batch_id")
                print("Batches:", batches)

                if len(batches) == 0:
                    continue

                n_panels = len(batches) + 1
                ncols = min(3, n_panels)
                nrows = int(np.ceil(n_panels / ncols))

                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(5.5 * ncols, 5.0 * nrows),
                    dpi=self.dpi,
                )
                axes = np.array(axes).reshape(-1)

                # Per-batch panels
                for i, b in enumerate(batches):
                    dfi = dfs[dfs["batch_id"].astype(str) == str(b)].copy()
                    print("Processing batch:", b, "Total data point:", len(dfi))

                    emb, dfi = self._compute_embedding(dfi, feature_cols)
                    ax = axes[i]
                    self._scatter_2d(
                        dfi,
                        emb,
                        color_by=color_by,
                        title=f"Batch {b}",
                        ax=ax,
                        add_legend=False,
                        label_order=session_label_order,  # << stable colors
                    )

                # Aggregate panel (use its handles/labels for legend)
                dfa = dfs.copy()
                emb_a, dfa = self._compute_embedding(dfa, feature_cols)
                ax = axes[len(batches)]
                _, handles_a, labels_a = self._scatter_2d(
                    dfa,
                    emb_a,
                    color_by=color_by,
                    title="ALL batches",
                    ax=ax,
                    add_legend=False,
                    label_order=session_label_order,  # << stable colors
                )

                # Disable unused axes
                for j in range(n_panels, len(axes)):
                    axes[j].axis("off")

                cond_label = condition if condition is not None else "all"
                suptitle = self._mk_title(
                    title_prefix,
                    f"{cond_label} | {rest_tag}",
                    f"Session {sess}",
                )

                # One global legend
                if handles_a and labels_a:
                    ncol = min(6, len(labels_a))
                    fig.legend(
                        handles=handles_a,
                        labels=labels_a,
                        loc="upper center",
                        ncol=ncol,
                        bbox_to_anchor=(0.5, 0.92),
                        frameon=True,
                        title=color_by,
                    )

                fig.suptitle(suptitle, fontsize=14, y=0.98)
                fig.subplots_adjust(top=0.84, wspace=0.30, hspace=0.35)

                if save_name or self.out_dir:
                    fname = save_name or self._default_filename(
                        scope="per_session",
                        condition=f"{cond_label}_{rest_tag}",
                        extra=f"session_{sess}",
                    )
                    self._save_fig(fig, fname)

                if show:
                    plt.show()
                else:
                    plt.close(fig)

    def plot_across_sessions(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        condition: Optional[str] = None,
        color_by: str = "Label_str",
        title_prefix: str = "",
        save_name: Optional[str] = None,
        show: bool = True,
    ) -> None:
        print("\n\nACROSS SESSIONS PLOTTING")

        base_df = self._filter_condition(df, condition)
        sessions = self._sorted_unique(base_df, "session_id")
        if len(sessions) == 0:
            print("[UMAP] No sessions found for the requested selection.")
            return

        # Global label order from WITH-REST superset, reused for both modes
        global_label_order = self._global_label_order(base_df, color_by)

        for inc_rest in [True, False]:
            df_use = base_df.copy()
            rest_tag = "with-rest" if inc_rest else "no-rest"

            if not inc_rest:
                print("Analysis without rest")
                df_use = df_use[df_use[color_by].astype(str) != "rest"].copy()

            # sessions might shrink if all points removed in some sessions
            sessions_use = self._sorted_unique(df_use, "session_id")
            if len(sessions_use) == 0:
                print("[UMAP] No sessions found after filtering.")
                continue

            n_panels = len(sessions_use) + 1
            ncols = min(3, n_panels)
            nrows = int(np.ceil(n_panels / ncols))

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5.5 * ncols, 5.0 * nrows),
                dpi=self.dpi,
            )
            axes = np.array(axes).reshape(-1)

            # Per-session panels
            for i, sess in enumerate(sessions_use):
                dfi = df_use[df_use["session_id"].astype(str) == str(sess)].copy()
                emb, dfi = self._compute_embedding(dfi, feature_cols)
                ax = axes[i]

                self._scatter_2d(
                    dfi,
                    emb,
                    color_by=color_by,
                    title=f"Session {sess}",
                    ax=ax,
                    add_legend=False,
                    label_order=global_label_order,  # << stable colors across everything
                )

            # Aggregate panel for legend
            dfa = df_use.copy()
            emb_a, dfa = self._compute_embedding(dfa, feature_cols)
            ax = axes[len(sessions_use)]
            _, handles_a, labels_a = self._scatter_2d(
                dfa,
                emb_a,
                color_by=color_by,
                title="ALL sessions",
                ax=ax,
                add_legend=False,
                label_order=global_label_order,  # << stable colors across everything
            )

            # Disable unused axes
            for j in range(n_panels, len(axes)):
                axes[j].axis("off")

            cond_label = condition if condition is not None else "all"
            suptitle = self._mk_title(
                title_prefix,
                f"{cond_label} | {rest_tag}",
                "Across sessions (per-session + aggregate)",
            )

            # One global legend (top center)
            if handles_a and labels_a:
                ncol = min(6, len(labels_a))
                fig.legend(
                    handles=handles_a,
                    labels=labels_a,
                    loc="upper center",
                    ncol=ncol,
                    bbox_to_anchor=(0.5, 0.92),
                    frameon=True,
                    title=color_by,
                )

            fig.suptitle(suptitle, fontsize=14, y=0.98)
            fig.subplots_adjust(top=0.84, wspace=0.30, hspace=0.35)

            if save_name or self.out_dir:
                fname = save_name or self._default_filename(
                    scope="across_sessions",
                    condition=f"{cond_label}_{rest_tag}",
                )
                self._save_fig(fig, fname)

            if show:
                plt.show()
            else:
                plt.close(fig)

    # ----------------------------
    # Internals
    # ----------------------------
    def _compute_embedding(
        self, df: pd.DataFrame, feature_cols: Sequence[str]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        if len(feature_cols) == 0:
            raise ValueError("feature_cols is empty")

        X = df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=True)
        print("Computing embeddings, X has size", X.shape)

        good = np.isfinite(X).all(axis=1)
        df = df.loc[good].copy()
        X = X[good]

        if self.config.max_points is not None and len(df) > self.config.max_points:
            df = df.sample(n=self.config.max_points, random_state=self.config.random_state)
            X = df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=True)

        if self.config.scale_features:
            X = StandardScaler().fit_transform(X)

        reducer = umap.UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
        )
        emb = reducer.fit_transform(X)
        print("Embeddings okay:", emb.shape)
        return emb, df

    def _global_label_order(self, df: pd.DataFrame, color_by: str) -> List[str]:
        """Sorted global label order used to keep label->color mapping stable."""
        if color_by not in df.columns:
            return []
        labels = df[color_by].astype(str).dropna().unique().tolist()
        labels.sort()
        return labels

    def _scatter_2d(
        self,
        df: pd.DataFrame,
        emb: np.ndarray,
        color_by: str,
        title: str,
        ax: Optional[plt.Axes] = None,
        add_legend: bool = False,
        label_order: Optional[List[str]] = None,
    ) -> Tuple[plt.Figure, List[Line2D], List[str]]:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.0, 5.5), dpi=self.dpi)
        else:
            fig = ax.figure

        if color_by not in df.columns:
            raise KeyError(f"color_by column '{color_by}' not found in df")

        labels = df[color_by].astype(str).to_numpy()

        # Use global label order if provided; otherwise local unique labels
        if label_order is None or len(label_order) == 0:
            uniq = pd.unique(labels).tolist()
        else:
            uniq = list(label_order)

        # Stable label -> index
        lut = {u: i for i, u in enumerate(uniq)}

        # Colormap uses total number of labels for stable color mapping
        cmap = cm.get_cmap("tab20", max(len(uniq), 1))

        # Convert labels -> indices (unknown labels handled safely)
        c_idx = np.array([lut.get(x, 0) for x in labels], dtype=int)
        colors = cmap(c_idx)

        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=10, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(True, alpha=0.2)

        present = set(pd.unique(labels).tolist())
        handles: List[Line2D] = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=cmap(lut[u]),
                markeredgecolor="none",
                label=u,
            )
            for u in uniq
            if u in present
        ]
        handle_labels = [h.get_label() for h in handles]

        if add_legend and len(handles) <= 25:
            ax.legend(
                handles=handles,
                labels=handle_labels,
                title=color_by,
                ncol=1,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                frameon=True,
            )

        return fig, handles, handle_labels

    def _filter_condition(self, df: pd.DataFrame, condition: Optional[str]) -> pd.DataFrame:
        if condition is None:
            return df
        if "condition" not in df.columns:
            return df
        return df[df["condition"] == condition].copy()

    @staticmethod
    def _sorted_unique(df: pd.DataFrame, col: str) -> List[str]:
        """
        Robust sorting for IDs that are often numeric but might be strings.
        Returns list of strings, sorted numerically when possible.
        """
        if col not in df.columns:
            return []

        s = df[col].dropna().astype(str)
        uniq = s.unique().tolist()

        def _key(x: str):
            try:
                return (0, int(float(x)))
            except Exception:
                return (1, x)

        uniq.sort(key=_key)
        return uniq

    def _mk_title(self, prefix: str, condition: Optional[str], scope: str) -> str:
        parts: List[str] = []
        if self.subject_id:
            parts.append(f"Subject: {self.subject_id}")
        if prefix:
            parts.append(prefix)
        parts.append(scope)
        if condition is not None:
            parts.append(f"Condition: {condition}")
        return " | ".join(parts)

    def _default_filename(self, scope: str, condition: Optional[str], extra: str = "") -> str:
        sid = self.subject_id or "unknown_subject"
        cond = condition or "all"
        tail = f"_{extra}" if extra else ""
        return f"umap_{sid}_{scope}_{cond}{tail}.png"

    def _save_fig(self, fig: plt.Figure, filename: str) -> None:
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / filename
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[UMAP] Saved: {out_path}")
