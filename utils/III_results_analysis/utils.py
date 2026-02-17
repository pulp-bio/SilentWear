import os
import json, hashlib, ast
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
from pathlib import Path
ARTIFACTS_DIR = Path(os.environ.get('SILENTWEAR_ARTIFACTS_DIR', Path(__file__).resolve().parents[2]/'artifacts'))
import copy
from utils.I_data_preparation.experimental_config import ORIGINAL_LABELS
from sklearn.metrics import ConfusionMatrixDisplay

# keys you might want to ignore when comparing "exact same run_cfg"
VOLATILE_KEYS = {
    "timestamp", "time", "datetime", "seed", "run_id",
    "output_dir", "log_dir", "wandb", "git_commit"
}




def generate_training_labels(include_rest: bool= False, 
                             original_label_map : dict = {}):
    """
    Generate:
        - train_label_map: {train_id: word}
        - train_to_orig:  {train_id: orig_id}
        - orig_to_train:  {orig_id: train_id}
        - num_classes
    """
    
    original_map = original_label_map

    if include_rest:
        # identity mapping
        train_label_map = original_map.copy()
        train_to_orig = {k: k for k in original_map.keys()}
        orig_to_train = {k: k for k in original_map.keys()}
    else:
        # remove rest (assumes rest is orig label 0)
        filtered_items = [(k, v) for k, v in original_map.items() if k != 0]

        # train labels become 0..7
        train_label_map = {
            new_k: word for new_k, (_, word) in enumerate(filtered_items)
        }
        train_to_orig = {
            new_k: orig_k for new_k, (orig_k, _) in enumerate(filtered_items)
        }
        orig_to_train = {
            orig_k: new_k for new_k, (orig_k, _) in enumerate(filtered_items)
        }

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
    Needed to use .uniqu() on pandas df. 
    """
    

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def normalize_and_canonicalize(cfg, ignore_keys):
    normalized = normalized_run_cfg(cfg, ignore_keys=ignore_keys)
    canonical  = canonicalize(normalized)
    return canonical


def dict_to_canonical(d):
    return json.dumps(d, sort_keys=True)

def drop_keys_recursive(obj, drop_keys=set()):
    """Remove volatile keys recursively from dict/list structures."""
    if isinstance(obj, dict):
        return {k: drop_keys_recursive(v, drop_keys)
                for k, v in obj.items() if k not in drop_keys}
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
    cms = [parse_cm_cell(v) for v in cm_series]
    cms = [cm for cm in cms if cm is not None]
    if len(cms) == 0:
        return None, None
    cms = np.stack(cms, axis=0)  # (n_folds, C, C)
    return cms.mean(axis=0), cms.std(axis=0)


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
        return np.divide(np.diag(cm), row_sum,
                         out=np.full(row_sum.shape, np.nan, dtype=float),
                         where=row_sum != 0)

def _get_word_labels_from_train_label_map(cell):
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


def plot_subject_word_accuracy_grid_from_summary(
    summary_df: pd.DataFrame,
    vocalized_condition: str = "vocalized",
    silent_condition: str = "silent",
    condition_col: str = "condition",
    subject_col: str = "subject",
    title_extras : str = None,
    save_path : Path = None,
):
    subjects = sorted(pd.unique(summary_df[subject_col]))

    fig, axes = plt.subplots(
        nrows=len(subjects), ncols=2,
        figsize=(16, max(3.2, 3.2 * len(subjects))),
        squeeze=False, sharey=True
    )

    conds = [(vocalized_condition, 0, "Vocalized"),
             (silent_condition, 1, "Silent")]

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
            overall_std  = df_sc["balanced_acc_std"].values[0]

            # collect per-run per-word recalls using mean_cm (one per run)
            recalls = []
            labels = None

            for _, row in df_sc.iterrows():
                cm = _to_array(row["mean_cm"])
                if cm is None:
                    continue
                recalls.append(_recall_from_cm(cm))

                if labels is None and "train_label_map" in row and row["train_label_map"] is not None:
                    labels = _get_word_labels_from_train_label_map(row["train_label_map"])

            if not recalls:
                ax.text(0.5, 0.5, "No mean_cm found", ha="center", va="center")
                ax.set_axis_off()
                continue

            recalls = np.vstack(recalls)  # (n_runs, C)
            mean_word = np.nanmean(recalls, axis=0)
            std_word  = np.nanstd(recalls, axis=0)

            C = len(mean_word)
            if labels is None or len(labels) != C:
                labels = [f"word_{i}" for i in range(C)]

            x = np.arange(C)
            ax.bar(x, mean_word, yerr=std_word, capsize=3)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")

            ax.set_title(f"SUBJECT: {subj} | {cond_title} | ACC: {overall_mean:.3f} ± {overall_std:.3f}")
            
            if c == 0:
                ax.set_ylabel("Per-word accuracy (recall)")

    suptitle = "Comparisons"
    if title_extras is not None:   
        print(title_extras)
        suptitle = f"{suptitle} | {title_extras}"
    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()
    plt.show()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    



def load_all_results(models_dire, subjects_to_consider=["S01", "S02", "S03", "S04"], conditions_to_consider=["silent", "vocalized"]):
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
                #print("Model type:", model_folder.name, "contains", len(model_runs), "variants")

                for curr_model_folder in model_runs:
                    # Read the config
                    run_cfg_file = curr_model_folder / "run_cfg.json"
                    if not run_cfg_file.exists():
                        continue

                    with open(run_cfg_file, "r", encoding="utf-8") as f:
                        run_cfg = json.load(f)

                    include_rest = run_cfg["experimental_settings"]["include_rest"]
                    # This is to map training labels back to original labels (if rest was removed during training)
                    original_label_map = ORIGINAL_LABELS.copy() 

                    # training labels (keep if you need label order)
                    train_label_map, train_to_orig, orig_to_train = generate_training_labels(
                        include_rest=include_rest,
                        original_label_map=original_label_map
                    )

                    cv_path = curr_model_folder / "cv_summary.csv"
                    if not cv_path.exists():
                        continue
                    model_summary_file = pd.read_csv(cv_path)

                    mean_cm, std_cm = mean_std_confusion_matrices(model_summary_file["confusion_matrix"])
                    # config signature for "exactly same run cfg" comparison
                    sig_full = cfg_signature(run_cfg['model_cfg'], drop_keys=set())                 # truly exact
                    # If models where trained with differnt seeds, we need to keep track 
                    if run_cfg.get("seeds") is not None:
                        sig_seeds = cfg_signature(run_cfg["seeds"], drop_keys=set())
                    else:
                        sig_seeds = "default"

                    #sig_stable = cfg_signature(run_cfg, drop_keys=VOLATILE_KEYS)       # ignore volatile
                    balanc_acc_vals_array = np.array(model_summary_file["balanced_accuracy"])
                    row = {
                        "subject": subject,
                        "condition": run_cfg.get("condition", condition),
                        "model_type_folder": model_folder.name,
                        "model_name": run_cfg["model_cfg"]["model"]["name"],
                        "include_rest": include_rest,

                        "balanced_acc_mean": np.mean(balanc_acc_vals_array),
                        
                        "balanced_acc_vals" : balanc_acc_vals_array, 
                        "balanced_acc_std": np.std(balanc_acc_vals_array),

                        "mean_cm": None if mean_cm is None else mean_cm.tolist(),
                        "std_cm": None if std_cm is None else std_cm.tolist(),

                        "run_cfg_signature_exact": sig_full,
                        "run_cfg_signature_seeds": sig_seeds,
                        "run_path": str(curr_model_folder),
                        "model_id": str(curr_model_folder.name), 
                        "run_cfg": run_cfg,  # keep full config for inspection
                        "train_label_map" : train_label_map,

                        "win_size_ms": run_cfg['experimental_settings']['window_size_ms']
                    }

                    all_rows.append(row)

    summary_df = pd.DataFrame(all_rows)

    return summary_df




################## For seed experiments ##################################
def save_per_condition_seed_report_csv(
    accs_seeds_raw,             # (n_seeds, n_subjects) in 0..1 OR %
    accs_vals_seeds_raw,        # (n_seeds, n_subjects) each entry is array-like of fold vals (0..1 OR %)
    subjects,                   # list[str] len n_subjects
    condition_name,             # "silent"/"vocalized"
    out_csv_path,               # Path
    values_are_fraction=True,   # True if values are in 0..1
    ):
    """
    Save per-condition CSV with:
      - per-seed per-subject mean (balanced_acc_mean)
      - per-seed per-subject fold values (balanced_acc_vals) as list-string
      - per-subject mean/std across seeds (based on balanced_acc_mean)
      - TOTAL row: mean/std across subjects of per-subject mean (same logic as your prints)
    """

    accs_mean = np.asarray(accs_seeds_raw)          # (n_seeds, n_subjects)
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
    fold_std  = np.zeros((n_seeds, n_subjects))

    # compute fold-level mean/std
    for seed in range(n_seeds):
        for sub in range(n_subjects):
            vals = np.asarray(accs_vals[seed, sub], dtype=float)
            if values_are_fraction:
                vals = vals * 100.0
            fold_mean[seed, sub] = np.mean(vals)
            fold_std[seed, sub]  = np.std(vals)

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
        df[f"seed_{s}_fold_std"]  = np.round(fold_std[s], 3)
        df[f"seed_{s}_fold_vals"] = [_vals_to_str(accs_vals[s, j]) for j in range(n_subjects)]


    total_row = {c: "" for c in df.columns}
    total_row["subject_id"] = "AVERAGE"
    for s in range(n_seeds):
        total_row[f"seed_{s}_fold_vals"] = np.mean(df[f"seed_{s}_fold_mean"].values)

    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    print(f"[CSV saved] {condition_name}: {out_csv_path}")


def return_data_directories(main_data_dire_proc, sub_ids, base_cfg, all_subject_models, condition, win_size_ms):
    data_dire_proc = []
    if all_subject_models == False: 
        # check if we want to train with silent - vocalized or both 
        if condition != "voc_and_silent":
            data_dire_proc.append(main_data_dire_proc / Path(f"{base_cfg['paths']['win_and_feats']}/{sub_ids}/{condition}/WIN_{win_size_ms}"))
        else:
            data_dire_proc.append(main_data_dire_proc / Path(f"{base_cfg['paths']['win_and_feats']}/{sub_ids}/silent/WIN_{win_size_ms}"))
            data_dire_proc.append(main_data_dire_proc / Path(f"{base_cfg['paths']['win_and_feats']}/{sub_ids}/vocalized/WIN_{win_size_ms}"))

    elif all_subject_models == True:
        for curr_sub_id in sub_ids:
            if condition != "voc_and_silent":
                data_dire_proc.append(main_data_dire_proc / Path(f"{base_cfg['paths']['win_and_feats']}/{curr_sub_id}/{condition}/WIN_{win_size_ms}"))
            else:
                data_dire_proc.append(main_data_dire_proc / Path(f"{base_cfg['paths']['win_and_feats']}/{curr_sub_id}/silent/WIN_{win_size_ms}"))
                data_dire_proc.append(main_data_dire_proc / Path(f"{base_cfg['paths']['win_and_feats']}/{curr_sub_id}/vocalized/WIN_{win_size_ms}"))
    for curr_data_dire_proc in data_dire_proc:        
        if curr_data_dire_proc.exists() == False:
            print("Data directory:", curr_data_dire_proc, "does not exist, exist")
            sys.exit()
    return data_dire_proc