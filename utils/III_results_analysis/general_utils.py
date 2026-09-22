# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import json
import hashlib
import ast
import re
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator, FixedLocator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS_DIR = Path(
    os.environ.get("SILENTWEAR_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
)
import copy
from utils.I_data_preparation.experimental_config import get_active_labels

# keys you might want to ignore when comparing "exact same run_cfg"
VOLATILE_KEYS = {
    "timestamp",
    "time",
    "datetime",
    "seed",
    "run_id",
    "output_dir",
    "log_dir",
    "wandb",
    "git_commit",
}


def generate_training_labels(
    include_rest: bool = False,
    original_label_map: dict = {},
    label_mode: str = "word",
):
    """
    Generate:
        - train_label_map: {train_id: word | sentence}
        - train_to_orig:  {train_id: orig_id}
        - orig_to_train:  {orig_id: train_id}
        - num_classes
    """
    original_map = original_label_map if original_label_map else get_active_labels(label_mode)

    if include_rest:
        # identity mapping
        train_label_map = original_map.copy()
        train_to_orig = {k: k for k in original_map.keys()}
        orig_to_train = {k: k for k in original_map.keys()}
    else:
        # remove rest (assumes rest is orig label 0)
        filtered_items = [(k, v) for k, v in original_map.items() if k != 0]
        train_label_map = {new_k: text for new_k, (_, text) in enumerate(filtered_items)}
        train_to_orig = {new_k: orig_k for new_k, (orig_k, _) in enumerate(filtered_items)}
        orig_to_train = {orig_k: new_k for new_k, (orig_k, _) in enumerate(filtered_items)}

    return train_label_map, train_to_orig, orig_to_train


def drop_path(d, path):
    """
    Delete a nested key given a path like:
    ["base_cfg","data","subject_id"].
    If any part is missing, do nothing.
    """
    cur = d
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return
        cur = cur[k]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)


def normalized_run_cfg(cfg, ignore_keys):
    cfg = copy.deepcopy(cfg)
    for p in ignore_keys:
        drop_path(cfg, p)
    return cfg


def canonicalize(obj):
    """
    Convert a dict to a canonical JSON string.
    Needed to use .unique() on pandas df.
    """
    
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalize_and_canonicalize(cfg, ignore_keys):
    normalized = normalized_run_cfg(cfg, ignore_keys=ignore_keys)
    canonical = canonicalize(normalized)
    return canonical


def dict_to_canonical(d):
    return json.dumps(d, sort_keys=True)


def drop_keys_recursive(obj, drop_keys=set()):
    """Remove volatile keys recursively from dict/list structures."""
    if isinstance(obj, dict):
        return {k: drop_keys_recursive(v, drop_keys) for k, v in obj.items() if k not in drop_keys}
    if isinstance(obj, list):
        return [drop_keys_recursive(x, drop_keys) for x in obj]
    return obj


def cfg_signature(run_cfg: dict, drop_keys=None) -> str:
    """Stable hash for comparing run_cfg equality."""
    drop_keys = drop_keys or set()
    cleaned = drop_keys_recursive(run_cfg, drop_keys)
    s = json.dumps(cleaned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_cm_cell(x):
    """Robustly parse a confusion-matrix cell from CSV (string -> list -> array)."""
    if pd.isna(x):
        return None
    cm = ast.literal_eval(x)
    return np.array(cm, dtype=float)


def mean_std_confusion_matrices(cm_series: pd.Series):
    """Calculate mean and std of confusion matrices from a pandas Series of string representations."""
    cms_list = [parse_cm_cell(v) for v in cm_series]
    cms_list = [cm for cm in cms_list if cm is not None]
    if len(cms_list) == 0:
        return None, None
    
    cms_arr = np.stack(cms_list, axis=0)  # (n_folds, C, C)
    return np.mean(cms_arr, axis=0), np.std(cms_arr, axis=0)


def _to_array(x):
    """Accept list-of-lists or stringified list-of-lists."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return np.array(x, dtype=float)


def _recall_from_cm(cm: np.ndarray):
    row_sum = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            np.diag(cm),
            row_sum,
            out=np.full(row_sum.shape, np.nan, dtype=float),
            where=row_sum != 0,
        )


def _get_text_labels_from_train_label_map(cell):
    if isinstance(cell, str):
        cell = ast.literal_eval(cell)
    if not isinstance(cell, dict):
        return None
    items = sorted(((int(k), v) for k, v in cell.items()), key=lambda kv: kv[0])
    return [v for _, v in items]


def fmt_sci(x: float) -> str:
    """Format 0.001 -> 1e-3, 1.0 -> 1, etc."""
    if x is None:
        return "NA"
    if x == 0:
        return "0"
    ax = abs(x)
    if (ax < 1e-2) or (ax >= 1e3):
        return f"{x:.0e}"
    return f"{x:g}"


def plot_subject_text_accuracy_grid_from_summary(
    summary_df: pd.DataFrame,
    vocalized_condition: str = "vocalized",
    silent_condition: str = "silent",
    condition_col: str = "condition",
    subject_col: str = "subject",
    title_extras: str | None = None,
    save_path: Path | None = None,
):
    subjects = sorted(pd.unique(summary_df[subject_col]))

    fig, axes = plt.subplots(
        nrows=len(subjects),
        ncols=2,
        figsize=(16, max(3.2, 3.2 * len(subjects))),
        squeeze=False,
        sharey=True,
    )

    conds = [(vocalized_condition, 0, "Vocalized"), (silent_condition, 1, "Silent")]

    for r, subj in enumerate(subjects):
        subj_df = summary_df[summary_df[subject_col] == subj]

        for cond_value, c, cond_title in conds:
            ax = axes[r, c]
            df_sc = subj_df[subj_df[condition_col] == cond_value]
            
            if df_sc.empty:
                ax.set_axis_off()
                continue
            
            # If multiple runs exist for same subject/condition, aggregate across runs:
            # - overall acc mean/std across runs
            overall_mean = df_sc["balanced_acc_mean"].values[0]
            overall_std = df_sc["balanced_acc_std"].values[0]

            # collect per-run per-text recalls using mean_cm (one per run)
            recalls = []
            labels = None

            for _, row in df_sc.iterrows():
                cm = _to_array(row["mean_cm"])
                if cm is None:
                    continue
                recalls.append(_recall_from_cm(cm))

                if (
                    labels is None
                    and "train_label_map" in row
                    and row["train_label_map"] is not None
                ):
                    labels = _get_text_labels_from_train_label_map(row["train_label_map"])

            if not recalls:
                ax.text(0.5, 0.5, "No mean_cm found", ha="center", va="center")
                ax.set_axis_off()
                continue

            recalls_arr = np.vstack(recalls)  # (n_runs, C)
            mean_text = np.nanmean(recalls_arr, axis=0)
            std_text = np.nanstd(recalls_arr, axis=0)

            C = len(mean_text)
            if labels is None or len(labels) != C:
                labels = [f"text_{i}" for i in range(C)]

            x = np.arange(C)
            ax.bar(x, mean_text, yerr=std_text, capsize=3)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")

            ax.set_title(
                f"SUBJECT: {subj} | {cond_title} | ACC: {overall_mean:.3f} ± {overall_std:.3f}"
            )

            if c == 0:
                ax.set_ylabel("Per-text accuracy (recall)")

    suptitle = "Comparisons"
    if title_extras is not None:
        print(title_extras)
        suptitle = f"{suptitle} | {title_extras}"
    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")


def load_all_results(
    models_dire,
    subjects_to_consider=["S01", "S02", "S03", "S04"],
    conditions_to_consider=["silent", "vocalized"],
):
    """
    Load and summarize saved experiment results from a directory tree of completed runs.

    This function scans a results directory containing per-subject and per-condition subfolders,
    loads run metadata (`run_cfg.json`) and CV summaries (`cv_summary.csv`), and returns a single
    pandas DataFrame where each row corresponds to one trained run (one model variant).

    Expected directory structure
    ----------------------------
    models_dire/
        <SUBJECT>/
            <CONDITION>/
                <MODEL_TYPE_FOLDER>/         # e.g., random_forest/, speechnet/
                    <RUN_ID>/                # e.g., model_1/, seed_0/, timestamp_*/
                        run_cfg.json
                        cv_summary.csv

    Notes on required files
    -----------------------
    - `run_cfg.json` must contain at least:
        - run_cfg["model_cfg"]["model"]["name"]               (model architecture name)
        - run_cfg["experimental_settings"]["window_size_ms"]  (window size used for the run)
        - run_cfg["experimental_settings"]["include_rest"]    (whether 'rest' label was used)
      Optionally:
        - run_cfg["condition"]                               (overrides folder-level condition)
        - run_cfg["seeds"]                                   (used to compute a seed signature)

    - `cv_summary.csv` must contain at least:
        - "balanced_accuracy"                                (one value per CV fold)
        - "confusion_matrix"                                 (serialized confusion matrix per fold)

    Behavior
    --------
    - Missing subject/condition folders are skipped silently.
    - Runs missing `run_cfg.json` or `cv_summary.csv` are skipped silently.
    - Confusion matrices are aggregated across folds using `mean_std_confusion_matrices(...)`.
    - Training label maps are derived from `include_rest` using `generate_training_labels(...)`
      to ensure consistent label ordering (useful for confusion matrix display).

    Parameters
    ----------
    models_dire : pathlib.Path
        Root directory containing the saved runs for an experiment (e.g., .../models/inter_session).
    subjects_to_consider : list[str]
        Subject identifiers to include (e.g., ["S01", "S02"]).
    conditions_to_consider : list[str]
        Conditions to include (e.g., ["silent", "vocalized"]).

    Returns
    -------
    pd.DataFrame
        A table where each row corresponds to one run folder. Key columns include:

        Identification / metadata:
        - subject (str)
        - condition (str)
        - model_type_folder (str): name of the folder grouping runs (e.g., "random_forest")
        - model_name (str): architecture name from run_cfg["model_cfg"]["model"]["name"]
        - model_id (str): run folder name (curr_model_folder.name)
        - run_path (str): absolute/relative path to the run folder
        - include_rest (bool)
        - win_size_ms (int)
        
        Performance summary:
        - balanced_acc_vals (np.ndarray): balanced accuracy per CV fold in [0,1]
        - balanced_acc_mean (float): mean of balanced_acc_vals
        - balanced_acc_std (float): std of balanced_acc_vals
        - mean_cm (list[list[float]] or None): mean confusion matrix across folds
        - std_cm (list[list[float]] or None): std confusion matrix across folds

        Config tracking:
        - run_cfg (dict): full loaded JSON config for inspection
        - run_cfg_signature_exact (str): hash/signature of run_cfg["model_cfg"]
        - run_cfg_signature_seeds (str): signature of run_cfg["seeds"] or "default"
        - train_label_map (dict): mapping used during training (possibly without 'rest')

    """
    
    all_rows = []

    for subject in subjects_to_consider:
        for condition in conditions_to_consider:
            subject_folder_path = models_dire / subject / condition
            if not subject_folder_path.exists():
                continue

            for model_folder in subject_folder_path.iterdir():
                if not model_folder.is_dir():
                    continue

                model_runs = [p for p in model_folder.iterdir() if p.is_dir()]
                # If the same model was trained multiple times (e.g., different seeds), we have multiple runs
                # print("Model type:", model_folder.name, "contains", len(model_runs), "variants") 
                for curr_model_folder in model_runs:
                    # Read the config
                    run_cfg_file = curr_model_folder / "run_cfg.json"
                    if not run_cfg_file.exists():
                        continue

                    with open(run_cfg_file, "r", encoding="utf-8") as f:
                        run_cfg = json.load(f)

                    include_rest = run_cfg["experimental_settings"]["include_rest"]
                    # This is to map training labels back to original labels (if rest was removed during training)
                    label_mode = run_cfg.get("experimental_settings", {}).get("label_mode", "word")
                    original_label_map = get_active_labels(label_mode)
                    # training labels (keep if you need label order)
                    train_label_map, train_to_orig, orig_to_train = generate_training_labels(
                        include_rest=include_rest,
                        original_label_map=original_label_map,
                        label_mode=label_mode,
                    )

                    cv_path = curr_model_folder / "cv_summary.csv"
                    if not cv_path.exists():
                        continue
                    model_summary_file = pd.read_csv(cv_path)

                    mean_cm, std_cm = mean_std_confusion_matrices(
                        model_summary_file["confusion_matrix"]
                    )
                    # config signature for "exactly same run cfg" comparison
                    sig_full = cfg_signature(run_cfg["model_cfg"], drop_keys=set())  # truly exact
                    # If models were trained with different seeds, we need to keep track
                    if run_cfg.get("seeds") is not None:
                        sig_seeds = cfg_signature(run_cfg["seeds"], drop_keys=set())
                    else:
                        sig_seeds = "default"

                    # sig_stable = cfg_signature(run_cfg, drop_keys=VOLATILE_KEYS)       # ignore volatile
                    balanc_acc_vals_array = np.array(model_summary_file["balanced_accuracy"], dtype=float)
                    
                    row = {
                        "subject": subject,
                        "condition": run_cfg.get("condition", condition),
                        "model_type_folder": model_folder.name,
                        "model_name": run_cfg["model_cfg"]["model"]["name"],
                        "include_rest": include_rest,
                        "balanced_acc_mean": float(np.mean(balanc_acc_vals_array)),
                        "balanced_acc_vals": balanc_acc_vals_array,
                        "balanced_acc_std": float(np.std(balanc_acc_vals_array)),
                        "mean_cm": None if mean_cm is None else mean_cm.tolist(),
                        "std_cm": None if std_cm is None else std_cm.tolist(),
                        "run_cfg_signature_exact": sig_full,
                        "run_cfg_signature_seeds": sig_seeds,
                        "run_path": str(curr_model_folder),
                        "model_id": str(curr_model_folder.name),
                        "run_cfg": run_cfg,  # keep full config for inspection
                        "train_label_map": train_label_map,
                        "win_size_ms": run_cfg["experimental_settings"]["window_size_ms"],
                    }
                    all_rows.append(row)
                    
    summary_df = pd.DataFrame(all_rows)
    return summary_df


def save_per_condition_seed_report_csv(
    accs_seeds_raw,  # (n_seeds, n_subjects) in 0..1 OR %
    accs_vals_seeds_raw,  # (n_seeds, n_subjects) each entry is array-like of fold vals (0..1 OR %)
    subjects,  # list[str] len n_subjects
    condition_name,  # "silent"/"vocalized"
    out_csv_path,  # Path
    values_are_fraction=True,  # True if values are in 0..1
):
    """
    Save per-condition CSV with:
      - per-seed per-subject mean (balanced_acc_mean)
      - per-seed per-subject fold values (balanced_acc_vals) as list-string
      - per-subject mean/std across seeds (based on balanced_acc_mean)
      - TOTAL row: mean/std across subjects of per-subject mean (same logic as your prints)
    """

    accs_mean = np.asarray(accs_seeds_raw)  # (n_seeds, n_subjects)
    accs_vals = np.asarray(accs_vals_seeds_raw, dtype=object)

    if accs_mean.ndim != 2:
        raise ValueError(f"accs_seeds_raw must be 2D, got {accs_mean.shape}")

    n_seeds, n_subjects = accs_mean.shape
    if accs_vals.shape != (n_seeds, n_subjects):
        raise ValueError(
            f"accs_vals_seeds_raw shape mismatch: expected {(n_seeds, n_subjects)}, got {accs_vals.shape}"
        )

    if len(subjects) != n_subjects:
        raise ValueError("subjects length mismatch")

    # containers for fold stats
    fold_mean = np.zeros((n_seeds, n_subjects))
    fold_std = np.zeros((n_seeds, n_subjects))

    # compute fold-level mean/std
    for seed in range(n_seeds):
        for sub in range(n_subjects):
            vals = np.asarray(accs_vals[seed, sub], dtype=float)
            if values_are_fraction:
                vals = vals * 100.0
            fold_mean[seed, sub] = float(np.mean(vals))
            fold_std[seed, sub] = float(np.std(vals))

    # stringify fold values
    def _vals_to_str(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        arr = np.asarray(v, dtype=float)
        if values_are_fraction:
            arr = arr * 100.0
        return "[" + ", ".join(f"{x:.2f}" for x in arr.tolist()) + "]"

    # build dataframe
    df = pd.DataFrame({"subject_id": subjects})

    for s in range(n_seeds):
        df[f"seed_{s}_fold_mean"] = np.round(fold_mean[s], 3)
        df[f"seed_{s}_fold_std"] = np.round(fold_std[s], 3)
        df[f"seed_{s}_fold_vals"] = [_vals_to_str(accs_vals[s, j]) for j in range(n_subjects)]

    total_row = {c: "" for c in df.columns}
    total_row["subject_id"] = "AVERAGE"
    for s in range(n_seeds):
        mean_val = np.mean(np.asarray(df[f"seed_{s}_fold_mean"].values, dtype=float))
        total_row[f"seed_{s}_fold_vals"] = f"{mean_val:.3f}"

    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    print(f"[CSV saved] {condition_name}: {out_csv_path}")


def return_data_directories(
    main_data_dire_proc, sub_ids, base_cfg, all_subject_models, condition, win_size_ms
):
    data_dire_proc = []
    if all_subject_models == False:
        # check if we want to train with silent - vocalized or both
        if condition != "voc_and_silent":
            data_dire_proc.append(
                main_data_dire_proc
                / Path(
                    f"{base_cfg['paths']['win_and_feats']}/{sub_ids}/{condition}/WIN_{win_size_ms}"
                )
            )
        else:
            data_dire_proc.append(
                main_data_dire_proc
                / Path(f"{base_cfg['paths']['win_and_feats']}/{sub_ids}/silent/WIN_{win_size_ms}")
            )
            data_dire_proc.append(
                main_data_dire_proc
                / Path(
                    f"{base_cfg['paths']['win_and_feats']}/{sub_ids}/vocalized/WIN_{win_size_ms}"
                )
            )

    elif all_subject_models == True:
        for curr_sub_id in sub_ids:
            if condition != "voc_and_silent":
                data_dire_proc.append(
                    main_data_dire_proc
                    / Path(
                        f"{base_cfg['paths']['win_and_feats']}/{curr_sub_id}/{condition}/WIN_{win_size_ms}"
                    )
                )
            else:
                data_dire_proc.append(
                    main_data_dire_proc
                    / Path(
                        f"{base_cfg['paths']['win_and_feats']}/{curr_sub_id}/silent/WIN_{win_size_ms}"
                    )
                )
                data_dire_proc.append(
                    main_data_dire_proc
                    / Path(
                        f"{base_cfg['paths']['win_and_feats']}/{curr_sub_id}/vocalized/WIN_{win_size_ms}"
                    )
                )
    for curr_data_dire_proc in data_dire_proc:
        if curr_data_dire_proc.exists() == False:
            print("Data directory:", curr_data_dire_proc, "does not exist, exist")
            sys.exit()
    return data_dire_proc


def resolve_csv_path(condition_dir: Path, model_run: Optional[str]) -> Optional[Path]:
    candidates: list[tuple[int, Path]] = []
    for net_dir in condition_dir.iterdir():
        if not net_dir.is_dir():
            continue
        for win_dir in net_dir.iterdir():
            if not win_dir.is_dir():
                continue
            if model_run is not None:
                p = win_dir / model_run / "cv_summary.csv"
                if p.exists():
                    return p
            else:
                for mr_dir in win_dir.iterdir():
                    if not mr_dir.is_dir():
                        continue
                    m = re.match(r"model_(\d+)$", mr_dir.name)
                    if m:
                        p = mr_dir / "cv_summary.csv"
                        if p.exists():
                            candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1][1]


def model_run_tag(model_run: Optional[str], csv_path: Path) -> str:
    if model_run is not None:
        return model_run
    return csv_path.parent.name


def window_from_path(csv_path: Path) -> float:
    for part in csv_path.parts:
        m = re.match(r"w(\d+)ms$", part)
        if m:
            return float(m.group(1)) / 1000.0
    return float("nan")


def build_save_path(base_save_path: Path, model_run_tag: str, suffix: str) -> Path:
    return base_save_path.with_name(f"{base_save_path.stem}_{model_run_tag}{suffix}")


# ---------------------------------------------------------------------------
# Multi-subject block layout
# ---------------------------------------------------------------------------


def build_multi_subject_blocks(
    ax: Axes,
    subjects: Sequence[str],
    x_values: np.ndarray,
    series: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    *,
    with_average: bool = True,
    series_styles: Optional[Dict[str, Dict[str, Any]]] = None,
    x_label: str = "x",
    y_label: str = "Value",
    y_lim: Optional[tuple] = None,
    y_major_step: Optional[float] = None,
    x_tick_labels: Optional[Sequence[str]] = None,
    gap: float = 1,
    block_margin: float = 0.1,
    label_y_offset: float = 5.0,
    label_fontsize: int = 10,
    x_tick_rotation: int = 0,
) -> List[float]:
    """
    Draw a multi-block line plot on *ax* where each subject (plus optionally an
    "Average" block) occupies its own horizontal segment separated by a gap.

    Parameters
    ----------
    ax : matplotlib Axes
    subjects : ordered list of subject names
    x_values : 1-D integer or float array of the shared x-axis positions
        (e.g. session counts, window sizes).  The *order* defines the column
        positions within each block.
    series : dict  { series_name -> { subject_name -> {"mean": array, "std": array} } }
        Each inner dict maps a subject (or "Average") to arrays of length
        len(x_values).
    with_average : if True (and len(subjects) > 1), append an "Average" block
        whose values are computed here as mean ± std-across-subjects.
    series_styles : dict  { series_name -> {"color": ..., "marker": ..., ...} }
        Passed directly to ax.errorbar.  Falls back to tab10 colours.
    x_label, y_label : axis labels
    y_lim : (lo, hi) for the y-axis
    y_major_step : step for MultipleLocator on y-axis (None = auto)
    x_tick_labels : labels for x_values ticks (default: str(x_values[i]))
    gap : blank columns between blocks
    label_y_offset : distance (in data units) below the top y-limit for block
        name labels
    label_fontsize : font size of block name labels

    Returns
    -------
    centers : list of float – x-axis centre of each block (for legend placement)
    """
    from matplotlib.ticker import MultipleLocator, FixedLocator

    nX = len(x_values)
    block_width = nX + gap

    blocks: List[str] = list(subjects)
    if with_average and len(subjects) > 1:
        blocks.append("Average")

    n_blocks = len(blocks)

    palette = plt.colormaps["tab10"]
    if series_styles is None:
        series_styles = {}
    for i, sname in enumerate(series):
        if sname not in series_styles:
            series_styles[sname] = {"color": palette(i % 10), "marker": "o"}

    # alternating background
    for bi, name in enumerate(blocks):
        start = bi * block_width
        face = "#f4f4f4" if (bi % 2 == 1) else "#ffffff"
        ax.axvspan(start - block_margin, start + nX - 1 + block_margin, color=face, zorder=0)

    ax.grid(True, which="major", linewidth=0.35, alpha=0.25)

    centers: List[float] = []
    pos_map = {float(v): j for j, v in enumerate(x_values)}

    # collect all means for Average computation
    # shape: {sname: list of arrays per subject}
    all_means: Dict[str, List[np.ndarray]] = {s: [] for s in series}
    all_stds: Dict[str, List[np.ndarray]] = {s: [] for s in series}

    already_labelled: set = set()

    for bi, name in enumerate(blocks):
        start = bi * block_width

        for sname, subj_data in series.items():
            if name == "Average":
                means_stack = np.array(all_means[sname])   # (n_subj, nX)
                if means_stack.size == 0:
                    continue
                y_mean = np.nanmean(means_stack, axis=0)
                y_std  = np.nanstd(means_stack, axis=0)
            else:
                entry = subj_data.get(name)
                if entry is None:
                    continue
                y_mean = np.asarray(entry["mean"], dtype=float)
                y_std  = np.asarray(entry["std"],  dtype=float)
                all_means[sname].append(y_mean)
                all_stds[sname].append(y_std)

            x_plot = np.array(
                [start + pos_map[float(v)] for v in x_values], dtype=float
            )

            style = series_styles.get(sname, {"color": "steelblue", "marker": "o"})
            do_label = sname not in already_labelled
            ax.errorbar(
                x_plot, y_mean, yerr=y_std,
                fmt=f"-{style.get('marker', 'o')}",
                color=style.get("color", "steelblue"),
                markersize=style.get("markersize", 4),
                linewidth=style.get("linewidth", 1.0),
                capsize=style.get("capsize", 2.5),
                elinewidth=style.get("elinewidth", 0.8),
                alpha=style.get("alpha", 0.9),
                label=sname if do_label else None,
            )
            if do_label:
                already_labelled.add(sname)

        centers.append(start + (nX - 1) / 2.0)

    # x-axis ticks: one major tick per x-value per block
    major_xticks, major_xlabels, minor_xticks = [], [], []
    tick_labels = x_tick_labels if x_tick_labels is not None else [str(v) for v in x_values]
    for bi in range(n_blocks):
        start = bi * block_width
        for j, (v, lbl) in enumerate(zip(x_values, tick_labels)):
            minor_xticks.append(start + j)
            major_xticks.append(start + j)
            major_xlabels.append(lbl)

    ax.xaxis.set_major_locator(FixedLocator(major_xticks))
    ha = "center" if x_tick_rotation == 0 else ("right" if x_tick_rotation > 0 else "left")
    ax.set_xticklabels(major_xlabels, rotation=x_tick_rotation, ha=ha, fontsize=8)
    ax.tick_params(axis="x", which="major", length=3, width=0.7)
    ax.set_xlim(-block_margin, (n_blocks - 1) * block_width + nX - 1 + block_margin)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))

    # block name labels near top
    y_top = ax.get_ylim()[1] if y_lim is None else y_lim[1]
    for c, name in zip(centers, blocks):
        ax.text(
            c, y_top - label_y_offset, str(name),
            ha="center", va="bottom", fontsize=label_fontsize,
        )

    return centers


def build_twin_axis_blocks(
    ax1: Axes,
    ax2: Axes,
    subjects: Sequence[str],
    x_values: np.ndarray,
    series1: Dict[str, Dict[str, Dict[str, np.ndarray]]], 
    series2: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    series1_styles: Optional[Dict[str, Dict[str, Any]]] = None,
    series2_styles: Optional[Dict[str, Dict[str, Any]]] = None,
    with_average: bool = True,
    x_label: str = "x",
    y1_label: str = "Metric 1",
    y2_label: str = "Metric 2",
    y1_lim: Optional[tuple] = None,
    y2_lim: Optional[tuple] = None,
    y1_major_step: Optional[float] = None,
    y2_major_step: Optional[float] = None,
    x_tick_labels: Optional[Sequence[str]] = None,
    gap: float = 1.0,
    block_margin: float = 0.5,
    label_y_offset: float = 5.0,
    label_fontsize: int = 10,
    label_position: str = "top",  # "top" or "bottom"
    x_tick_rotation: int = 0,
) -> List[float]:
    """
    Draw a multi-block line plot utilizing twin Y-axes (ax1 and ax2) where each 
    subject (plus an optional "Average" block) occupies its own horizontal segment.
    """

    nX = len(x_values)
    block_width = nX + gap

    blocks: List[str] = list(subjects)
    if with_average and len(subjects) > 1:
        blocks.append("Average")

    n_blocks = len(blocks)

    if series1_styles is None:
        series1_styles = {list(series1.keys())[0]: {"color": "blue", "marker": "o"}}
    if series2_styles is None:
        series2_styles = {list(series2.keys())[0]: {"color": "red", "marker": "o"}}

    # Grid setup (only on primary axis)
    ax1.grid(True, which="major", linewidth=0.35, alpha=0.20)
    ax2.grid(False)

    centers: List[float] = []
    pos_map = {float(v): j for j, v in enumerate(x_values)}

    all_means_1: Dict[str, List[np.ndarray]] = {s: [] for s in series1}
    all_stds_1: Dict[str, List[np.ndarray]] = {s: [] for s in series1}
    all_means_2: Dict[str, List[np.ndarray]] = {s: [] for s in series2}
    all_stds_2: Dict[str, List[np.ndarray]] = {s: [] for s in series2}

    already_labelled: set = set()

    for bi, name in enumerate(blocks):
        start = bi * block_width
        
        # Draw alternating background shades
        face = "#f4f4f4" if (bi % 2 == 1) else "#ffffff"
        ax1.axvspan(start - block_margin, start + nX - 1 + block_margin, color=face, zorder=0)

        # Plot Series 1 (Left Y-Axis)
        for sname, subj_data in series1.items():
            if name == "Average":
                means_stack = np.array(all_means_1[sname])
                if means_stack.size == 0:
                    continue
                y_mean = np.nanmean(means_stack, axis=0)
                y_std  = np.nanstd(means_stack, axis=0)
            else:
                entry = subj_data.get(name)
                if entry is None:
                    continue
                y_mean = np.asarray(entry["mean"], dtype=float)
                y_std  = np.asarray(entry["std"],  dtype=float)
                all_means_1[sname].append(y_mean)
                all_stds_1[sname].append(y_std)

            x_plot = np.array([start + pos_map[float(v)] for v in x_values], dtype=float)
            style = series1_styles.get(sname, {"color": "blue", "marker": "o"})
            do_label = sname not in already_labelled
            
            ax1.errorbar(
                x_plot, y_mean, yerr=y_std,
                fmt=f"-{style.get('marker', 'o')}",
                color=style.get("color", "blue"),
                markersize=style.get("markersize", 3),
                linewidth=style.get("linewidth", 0.5),
                capsize=style.get("capsize", 2),
                alpha=style.get("alpha", 0.95),
                label=sname if do_label else None,
            )
            if do_label:
                already_labelled.add(sname)

        # Plot Series 2 (Right Y-Axis)
        for sname, subj_data in series2.items():
            if name == "Average":
                means_stack = np.array(all_means_2[sname])
                if means_stack.size == 0:
                    continue
                y_mean = np.nanmean(means_stack, axis=0)
                y_std  = np.nanstd(means_stack, axis=0)
            else:
                entry = subj_data.get(name)
                if entry is None:
                    continue
                y_mean = np.asarray(entry["mean"], dtype=float)
                y_std  = np.asarray(entry["std"],  dtype=float)
                all_means_2[sname].append(y_mean)
                all_stds_2[sname].append(y_std)

            x_plot = np.array([start + pos_map[float(v)] for v in x_values], dtype=float)
            style = series2_styles.get(sname, {"color": "red", "marker": "o"})
            do_label = sname not in already_labelled
            
            ax2.errorbar(
                x_plot, y_mean, yerr=y_std,
                fmt=f"-{style.get('marker', 'o')}",
                color=style.get("color", "red"),
                markersize=style.get("markersize", 3),
                linewidth=style.get("linewidth", 0.5),
                capsize=style.get("capsize", 2),
                alpha=style.get("alpha", 0.85),
                label=sname if do_label else None,
            )
            if do_label:
                already_labelled.add(sname)

        centers.append(start + (nX - 1) / 2.0)

    # Setup ticks and labels
    major_xticks, major_xlabels, minor_xticks = [], [], []
    tick_labels = x_tick_labels if x_tick_labels is not None else [str(v) for v in x_values]
    
    for bi in range(n_blocks):
        start = bi * block_width
        for j, (v, lbl) in enumerate(zip(x_values, tick_labels)):
            minor_xticks.append(start + j)
            major_xticks.append(start + j)
            major_xlabels.append(lbl)

    ax1.xaxis.set_major_locator(FixedLocator(major_xticks))
    ax1.set_xticklabels(major_xlabels, rotation=x_tick_rotation, ha="center", fontsize=8)
    ax1.tick_params(axis="x", which="major", length=3, width=0.7)
    
    # Global Layout
    ax1.set_xlim(-block_margin, (n_blocks - 1) * block_width + nX - 1 + block_margin)
    ax1.set_xlabel(x_label)
    
    ax1.set_ylabel(y1_label, color="blue")
    ax1.tick_params(axis="y", colors="blue")
    ax1.spines["left"].set_color("blue")
    if y1_lim is not None:
        ax1.set_ylim(*y1_lim)
    if y1_major_step is not None:
        ax1.yaxis.set_major_locator(MultipleLocator(y1_major_step))

    ax2.set_ylabel(y2_label, color="red")
    ax2.tick_params(axis="y", colors="red")
    ax2.spines["right"].set_color("red")
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    if y2_lim is not None:
        ax2.set_ylim(*y2_lim)
    if y2_major_step is not None:
        ax2.yaxis.set_major_locator(MultipleLocator(y2_major_step))

    # Add block name labels at the top
    y_top = ax1.get_ylim()[1] if y1_lim is None else y1_lim[1]
    y_bottom = ax1.get_ylim()[0] if y1_lim is None else y1_lim[0]
    for c, name in zip(centers, blocks):
        if label_position.lower() in ["up", "top"]:
            # Upper position (default)
            ax1.text(
                c, y_top - label_y_offset, str(name),
                ha="center", va="bottom", fontsize=label_fontsize,
            )
        else:
            # Down position
            ax1.text(
                c, y_bottom + label_y_offset, str(name),
                ha="center", va="bottom", fontsize=label_fontsize,
            )

    return centers