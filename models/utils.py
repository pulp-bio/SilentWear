# # SPDX-FileCopyrightText: 2026 ETH Zurich
# # SPDX-License-Identifier: Apache-2.0

"""
Utils function for models
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from models.models_factory import ModelSpec, build_model_from_spec
import torch
from utils.I_data_preparation.experimental_config import FS


def compute_metrics(y_true, y_pred):
    """
    Docstring for compute_metrics

    :param y_true: true labels
    :param y_pred: predicted labels

    Returns metrics,y_true, y_pred
    """
    # ===== Metrics summary =====
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Add also weighted metrics to take into account class imbalace
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }

    print("=== Test Metrics ===")
    print(f"Accuracy        : UNBALANCED {acc:.2f}        - BALANCED {balanced_acc:.2f}")
    print(
        f"Precision       : MACRO      {precision_macro:.2f}  - WEIGHTED {precision_weighted:.2f}"
    )
    print(f"Recall          : MACRO      {recall_macro:.2f}     - WEIGHTED {recall_weighted:.2f}")
    print(f"F1-score        : MACRO      {f1_macro:.2f}         - WEIGHTED {f1_weighted:.2f}")

    # print(cm)

    return metrics, y_true, y_pred


import torch.nn as nn


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def check_weights_updated(before_state_dict, model_after):
    """
    Returns True if at least one parameter tensor differs after loading.
    """
    after_sd = model_after.state_dict()
    changed = False

    for k, v_before in before_state_dict.items():
        if k not in after_sd:
            continue
        v_after = after_sd[k]

        # only compare tensors
        if torch.is_tensor(v_before) and torch.is_tensor(v_after):
            if not torch.equal(v_before, v_after):
                changed = True
                break

    return changed


def load_pretrained_model(base_cfg, model_cfg, pretrained_model_path):
    num_classes = 9 if base_cfg["experiment"]["include_rest"] else 8

    spec = ModelSpec(
        kind=model_cfg["model"]["kind"],
        name=model_cfg["model"]["name"],
        kwargs=model_cfg["model"]["kwargs"],
    )

    ctx = {
        "num_channels": 14,
        "num_samples": int(base_cfg["window"]["window_size_s"] * FS),
        "num_classes": num_classes,
    }

    model = build_model_from_spec(spec, ctx)

    # snapshot BEFORE
    before_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # load checkpoint on CPU, no CUDA
    cpt = torch.load(pretrained_model_path, map_location="cpu")
    state_dict = cpt.get("model_state_dict", cpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    if check_weights_updated(before_sd, model):
        return model
    else:
        print("No weights changed after load — check checkpoint keys / strictness.")
        return None
