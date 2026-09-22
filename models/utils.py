# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
Utils function for models
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from models.models_factory import ModelSpec, build_model_from_spec
import numpy as np
import torch
import torch.nn as nn
from utils.I_data_preparation.experimental_config import FS, get_active_labels, build_label_maps


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


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

    print("\n=== Test Metrics ===")
    print(f"{ 'Accuracy':<15}: UNBALANCED {acc:6.2f}  - BALANCED {balanced_acc:6.2f}")
    print(f"{ 'Precision':<15}: MACRO      {precision_macro:6.2f}  - WEIGHTED {precision_weighted:6.2f}")
    print(f"{ 'Recall':<15}: MACRO      {recall_macro:6.2f}  - WEIGHTED {recall_weighted:6.2f}")
    print(f"{ 'F1-score':<15}: MACRO      {f1_macro:6.2f}  - WEIGHTED {f1_weighted:6.2f}")

    # print(cm)

    return metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# Recognition metrics
# ---------------------------------------------------------------------------


def build_word_vocabulary(texts) -> Tuple[str, ...]:
    """Vocabulary of every word appearing across *all* the given texts."""
    return tuple(sorted({w for t in texts for w in str(t).split()}))


def snap_texts_to_vocabulary(texts, vocabulary):
    """Rewrite every word of `texts` as its nearest vocabulary word."""
    import editdistance

    vocab = tuple(vocabulary)
    if not vocab:
        return [str(t) for t in texts]

    cache = {w: w for w in vocab}

    def nearest(word: str) -> str:
        hit = cache.get(word)
        if hit is None:
            hit = min(vocab, key=lambda cand: (editdistance.eval(word, cand), cand))
            cache[word] = hit
        return hit

    return [" ".join(nearest(w) for w in str(t).split()) for t in texts]


def compute_wer_metrics(y_true, y_pred, verbose: bool = True, vocabulary=None):
    """Compute recognition metrics (WER, CER) for the free-character CTC decoder.

    Used in place of accuracy/precision/recall/F1 when the model is decoded as a
    free-character CTC recogniser. This is closed-set recognition scored by edit
    distance.
    """
    import jiwer

    wer = float(jiwer.wer(y_true, y_pred))
    cer = float(jiwer.cer(y_true, y_pred))

    # Balanced variants: mean of per-class WER/CER, where each unique y_true
    # text is a class, so over-represented classes do not dominate the score
    # (mirrors balanced vs unbalanced accuracy in compute_metrics).
    groups = {}
    for ref, hyp in zip(y_true, y_pred):
        groups.setdefault(ref, ([], []))
        groups[ref][0].append(ref)
        groups[ref][1].append(hyp)
    balanced_wer = float(np.mean([jiwer.wer(refs, hyps) for refs, hyps in groups.values()]))
    balanced_cer = float(np.mean([jiwer.cer(refs, hyps) for refs, hyps in groups.values()]))

    vocab = build_word_vocabulary(y_true) if vocabulary is None else tuple(vocabulary)
    y_pred_snapped = snap_texts_to_vocabulary(y_pred, vocab)
    vocab_wer = float(jiwer.wer(y_true, y_pred_snapped))
    snapped_groups = {}
    for ref, hyp in zip(y_true, y_pred_snapped):
        snapped_groups.setdefault(ref, ([], []))
        snapped_groups[ref][0].append(ref)
        snapped_groups[ref][1].append(hyp)
    balanced_vocab_wer = float(
        np.mean([jiwer.wer(refs, hyps) for refs, hyps in snapped_groups.values()])
    )

    metrics = {
        "wer": wer,
        "balanced_wer": balanced_wer,
        "cer": cer,
        "balanced_cer": balanced_cer,
        "vocab_wer": vocab_wer,
        "balanced_vocab_wer": balanced_vocab_wer,
        "vocab_size": int(len(vocab)),
        "n_samples": int(len(y_true)),
    }

    if verbose:
        print("\n=== Test Metrics (recognition) ===")
        print(f"{'WER':<15}: UNBALANCED {wer:6.4f}  - BALANCED {balanced_wer:6.4f}")
        print(f"{'CER':<15}: UNBALANCED {cer:6.4f}  - BALANCED {balanced_cer:6.4f}")
        print(
            f"{'WER (vocab)':<15}: UNBALANCED {vocab_wer:6.4f}  - BALANCED {balanced_vocab_wer:6.4f}"
            f"   [vocab={len(vocab)} words]"
        )

    return metrics, y_true, y_pred


# ---------------------------------------------------------------------------
# Model size and cost
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model: nn.Module, example_input: torch.Tensor) -> Optional[int]:
    """Total FLOPs of one forward pass on ``example_input``.

    Uses PyTorch's native ``torch.utils.flop_counter.FlopCounterMode`` (no extra
    dependency). A multiply-add is counted as 2 FLOPs (~2x the MAC count) for every 
    op that has a flop formula: conv, linear/matmul and scaled-dot-product attention.

    Returns the total FLOPs (int) or ``None`` if counting is unavailable/fails, so
    it can never break a training run.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None

    was_training = model.training
    model.eval()
    # Disable the multihead-attention fast path so the Transformer encoder's
    # matmuls are visible to the counter; restored in `finally`.
    fastpath = None
    try:
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    except Exception:
        fastpath = None

    try:
        flop_counter = FlopCounterMode(display=False)
        with flop_counter, torch.no_grad():
            model(example_input)
        return int(flop_counter.get_total_flops())
    except Exception as exc:  # pragma: no cover - never break a run over profiling
        print(f"[FLOPs] counting skipped ({type(exc).__name__}: {exc}).")
        return None
    finally:
        if fastpath is not None:
            try:
                torch.backends.mha.set_fastpath_enabled(fastpath)
            except Exception:
                pass
        if was_training:
            model.train()


def conv_input_shape(model: nn.Module, example_input: torch.Tensor) -> Optional[Tuple[int, ...]]:
    """Shape of the tensor that actually enters the first convolution.

    Returns the shape tuple, or ``None`` if there is no conv or the pass fails
    (profiling must never break a run).
    """
    conv = next(
        (m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))),
        None,
    )
    if conv is None:
        return None

    captured: dict = {}

    def _hook(_module, inputs):
        if inputs and hasattr(inputs[0], "shape"):
            captured["shape"] = tuple(inputs[0].shape)

    handle = conv.register_forward_pre_hook(_hook)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example_input)
    except Exception as exc:  # pragma: no cover - never break a run over profiling
        print(f"[Conv input shape] skipped ({type(exc).__name__}: {exc}).")
        return None
    finally:
        handle.remove()
        if was_training:
            model.train()

    return captured.get("shape")


# ---------------------------------------------------------------------------
# Checkpoints and architecture info
# ---------------------------------------------------------------------------


def check_weights_updated(before_state_dict: dict, model_after: nn.Module) -> bool:
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


def resolve_num_classes_from_cfg(
    base_cfg: dict, model_cfg: dict, train_label_map: dict | None = None
) -> int:
    """Resolve output classes from config."""
    model_section = model_cfg.get("model", {}) or {}
    kwargs = model_section.get("kwargs", {}) or {}
    train_cfg = kwargs.get("train_cfg", {}) or {}
    loss_name = str(train_cfg.get("loss_name", "")).lower().strip()

    if model_section.get("kind") == "dl" and loss_name == "ctc":
        if train_label_map is None:
            raise ValueError(
                "train_label_map must be provided when loss_name='ctc'."
            )

        from utils.I_data_preparation.ctc_text_mapper import CTCTextMapper, DEFAULT_BLANK_ID

        ctc_cfg = train_cfg.get("ctc")
        if not isinstance(ctc_cfg, dict):
            raise KeyError("For loss_name='ctc', model.kwargs.train_cfg.ctc must be provided.")

        lexicon_path = ctc_cfg.get("lexicon_path")
        if not lexicon_path:
            raise ValueError(
                "For loss_name='ctc', provide model.kwargs.train_cfg.ctc.lexicon_path."
            )

        use_full_alphabet = str(ctc_cfg.get("decoding", "lexicon")).lower() == "recognition"
        mapper = CTCTextMapper(
            lexicon_path=lexicon_path,
            train_label_map=train_label_map,
            blank_id=ctc_cfg.get("blank_id", DEFAULT_BLANK_ID),
            use_full_alphabet=use_full_alphabet,
        )
        print("CTC token vocab size (without blank):", len(mapper.char_to_int))
        return len(mapper.char_to_int)

    include_rest = bool(base_cfg["experiment"].get("include_rest", False))
    label_mode = base_cfg.get("experiment", {}).get("label_mode", "word")
    original_label_map = get_active_labels(label_mode)

    if not original_label_map:
        raise ValueError(f"get_active_labels('{label_mode}') returned an empty map.")

    if include_rest:
        return len(original_label_map)
    else:
        return len([k for k in original_label_map if k != 0])


def load_pretrained_model(
    base_cfg: dict,
    model_cfg: dict,
    pretrained_model_path: Union[str, Path],
    train_label_map: Optional[dict] = None,
) -> Optional[nn.Module]:
    # CTC needs the training label map to size the output vocabulary. Rebuild it
    # from the config when the caller did not supply one.
    if train_label_map is None:
        include_rest = bool(base_cfg.get("experiment", {}).get("include_rest", False))
        label_mode = base_cfg.get("experiment", {}).get("label_mode", "word")
        train_label_map, _, _ = build_label_maps(label_mode, include_rest)

    num_classes = resolve_num_classes_from_cfg(base_cfg, model_cfg, train_label_map=train_label_map)

    spec = ModelSpec(
        kind=model_cfg["model"]["kind"],
        name=model_cfg["model"]["name"],
        kwargs=model_cfg["model"]["kwargs"],
    )

    channel_order = base_cfg.get("channel_order")
    num_channels = len(channel_order) if channel_order else 14

    ctx = {
        "num_channels": num_channels,
        "num_samples": int(base_cfg["window"]["window_size_s"] * FS),
        "num_classes": num_classes,
    }

    model = build_model_from_spec(spec, ctx)

    # snapshot BEFORE
    before_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if str(pretrained_model_path).endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(pretrained_model_path)
    else:
        cpt = torch.load(pretrained_model_path, map_location="cpu")
        state_dict = cpt.get("model_state_dict", cpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. missing={len(missing)} unexpected={len(unexpected)}")

    if check_weights_updated(before_sd, model):
        return model
    else:
        print("No weights changed after load — check checkpoint keys / strictness.")
        return None
    

def save_model_architecture_to_csv(
    model: nn.Module, model_name: str, example_input: Optional[torch.Tensor] = None
) -> Optional[Path]:
    """
    Extract layer-wise parameter information and save it to a CSV file.
    """
    output_dir = PROJECT_ROOT / "models" / "cnn_architectures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / "architectures_info" / f"{model_name}_architecture_info.csv"
    
    if not isinstance(model, nn.Module):
        print(f"[INFO] Model {model_name} is not a torch.nn.Module. Skipping architecture extraction.")
        return None

    total_params, trainable_params =  count_params(model)
    
    rows = []
    rows.append({
        "Layer_Name": "GLOBAL_SUMMARY",
        "Layer_Type": "Total_Parameters",
        "Parameters_Count": total_params
    })
    rows.append({
        "Layer_Name": "GLOBAL_SUMMARY",
        "Layer_Type": "Trainable_Parameters",
        "Parameters_Count": trainable_params
    })

    if example_input is not None:
        input_shape = tuple(example_input.shape)
        rows.append({
            "Layer_Name": "GLOBAL_SUMMARY",
            "Layer_Type": "Input_Shape",
            "Parameters_Count": str(input_shape)
        })
        conv_in_shape = conv_input_shape(model, example_input)
        if conv_in_shape is not None:
            domain = getattr(model, "domain", "time")
            print(f"[Conv input] {model_name}: {conv_in_shape} entering the conv stack "
                  f"(domain='{domain}', from model input {input_shape})")
            rows.append({
                "Layer_Name": "GLOBAL_SUMMARY",
                "Layer_Type": "Conv_Input_Shape",
                "Parameters_Count": str(conv_in_shape)
            })
        flops = count_flops(model, example_input)
        if flops is not None:
            print(f"[FLOPs] {model_name}: {flops:,} FLOPs/sample ({flops / 1e6:.1f} MFLOPs) "
                  f"for input {input_shape}")
            rows.append({
                "Layer_Name": "GLOBAL_SUMMARY",
                "Layer_Type": "Forward_FLOPs_per_sample",
                "Parameters_Count": flops
            })

    # Inspecting layers and their parameters
    for name, module in model.named_modules():
        layer_params = sum(p.numel() for p in module.parameters(recurse=False))
        if len(list(module.children())) == 0 or layer_params > 0:
            display_name = name if name != "" else "root"
            rows.append({
                "Layer_Name": display_name,
                "Layer_Type": type(module).__name__,
                "Parameters_Count": layer_params
            })
            
    # Saving to CSV
    df_architecture = pd.DataFrame(rows)
    df_architecture.to_csv(file_path, index=False)
    
    print(f"[INFO] Architecure saved: {file_path}")
    return file_path