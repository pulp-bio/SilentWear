"""
Utils for Ablations
"""

import pandas as pd
from pathlib import Path
import numpy as np
import yaml
import re
import json


# -----------------------
# Helpers
# -----------------------
def dump_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def safe_tag(s: str) -> str:
    """Make a string safe for folder names."""
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    return s


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

def json_str(x) -> str:
    return json.dumps(x)

############################################################

def apply_blocks(model_cfg: dict, out_channels_list, kernel_list, pool_list):
    """
    Overrides model config
    
    :param model_cfg: Description
    :type model_cfg: dict
    :param out_channels_list: Description
    :param kernel_list: Description
    :param pool_list: Description
    """
    blocks = model_cfg["model"]["kwargs"]["blocks_config"]
    if len(blocks) != len(out_channels_list) or len(blocks) != len(kernel_list) or len(blocks) != len(pool_list):
        raise ValueError("blocks_config length mismatch")

    for i, b in enumerate(blocks):
        b["out_channels"] = int(out_channels_list[i])
        b["kernel"] = kernel_list[i]
        b["pool"] = pool_list[i]



def summarize_subject_run(run_dir: Path):
    """
    Read per-run cv_summary.csv and return mean/std of the chosen metric across folds.
    """
    csv_path = run_dir / "cv_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    metric_col = "balanced_accuracy"
    vals = pd.to_numeric(df[metric_col]).values
    if len(vals) == 0:
        raise ValueError(f"Metric column '{metric_col}' has no numeric values in {csv_path}")

    return {
        "metric_col": metric_col,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n_folds": int(len(vals)),
    }


def extract_hparams(model_cfg: dict) -> dict:
    kwargs = model_cfg["model"].get("kwargs", {})
    train_cfg = kwargs.get("train_cfg", {})

    optimizer_cfg = train_cfg.get("optimizer_cfg", {})
    opt_name = optimizer_cfg.get("name", train_cfg.get("optimizer", None))

    lr = optimizer_cfg.get("lr", train_cfg.get("lr", None))
    wd = train_cfg.get("weight_decay", None)
    epochs = train_cfg.get("num_epochs", None)

    return {
        "model_name": model_cfg["model"].get("name", ""),
        "p_dropout": kwargs.get("p_dropout", None),
        "global_pool": kwargs.get("global_pool", None),
        "optimizer": opt_name,
        "lr": lr,
        "weight_decay": wd,
        "num_epochs": epochs,
    }


def update_master_csv(master_csv_path: Path, row: dict, key_cols=("variant",)):
    """
    Append a row to master CSV, or overwrite an existing row matching key_cols.
    """
    master_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if master_csv_path.exists():
        df = pd.read_csv(master_csv_path)
        # find matching rows
        mask = pd.Series([True] * len(df))
        for k in key_cols:
            mask &= (df[k].astype(str) == str(row[k]))
        if mask.any():
            # overwrite first match
            idx = df[mask].index[0]
            for k, v in row.items():
                df.at[idx, k] = v
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    # stable ordering: keep key first
    key_first = [c for c in key_cols if c in df.columns]
    rest = [c for c in df.columns if c not in key_first]
    df = df[key_first + rest]

    df.to_csv(master_csv_path, index=False)



def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable