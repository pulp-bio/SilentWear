# Copyright ETH Zurich 2026
# Modified by: Carola Bonamico; Date: 10/09/2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#

"""
models_factory.py

Centralized model factory + registry for experiments.

Supports:
- Deep Learning models: return torch.nn.Module
- Classical ML models: return sklearn-style estimators (fit/predict), or compatible wrappers

Usage
-----
spec = ModelSpec(kind="dl", name="speech_net", kwargs={"dropout_rate": 0.1})
model = build_model_from_spec(spec, ctx)

spec = ModelSpec(kind="ml", name="logreg", kwargs={"C": 1.0})
estimator = build_model_from_spec(spec, ctx)

# Or with a custom factory bypassing the registry:
custom_model = build_model(kind="ml", factory=my_custom_ml_factory, model_kwargs={"alpha": 0.5})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union, Literal
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


ModelKind = Literal["dl", "ml"]

# For ML, we keep it very permissive to avoid hard-depending on sklearn typing
MLEstimator = Any

DLFactory = Callable[..., nn.Module]
MLFactory = Callable[..., MLEstimator]

DL_MODEL_REGISTRY: Dict[str, DLFactory] = {}
ML_MODEL_REGISTRY: Dict[str, MLFactory] = {}


@dataclass(frozen=True)
class ModelSpec:
    """
    Declarative model spec used by experiment configs.

    kind:
        "dl" -> registry returns torch.nn.Module
        "ml" -> registry returns estimator (sklearn-like)

    name:
        key in the respective registry

    kwargs:
        model-specific keyword arguments
    """

    kind: ModelKind
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registration decorators
# ---------------------------------------------------------------------------


def register_dl_model(name: str) -> Callable[[DLFactory], DLFactory]:
    """Decorator to register a deep learning model factory."""

    def deco(fn: DLFactory) -> DLFactory:
        if name in DL_MODEL_REGISTRY:
            raise ValueError(f"DL model '{name}' already registered.")
        DL_MODEL_REGISTRY[name] = fn
        return fn

    return deco


def register_ml_model(name: str) -> Callable[[MLFactory], MLFactory]:
    """Decorator to register a classical ML model factory."""

    def deco(fn: MLFactory) -> MLFactory:
        if name in ML_MODEL_REGISTRY:
            raise ValueError(f"ML model '{name}' already registered.")
        ML_MODEL_REGISTRY[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------------------
# Unified builders
# ---------------------------------------------------------------------------


def build_model(
    *,
    kind: ModelKind,
    name: Optional[str] = None,
    factory: Optional[Callable[..., Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> Union[nn.Module, MLEstimator]:
    """
    Build and return a model instance (DL or ML).

    Parameters
    ----------
    kind:
        "dl" or "ml"
    name:
        Registry key. Optional if `factory` is provided.
    factory:
        Explicit factory callable. Takes precedence over `name` lookup.
    model_kwargs:
        Optional model-specific kwargs (hyperparams, architecture params, etc.)
    ctx:
        Context dict produced by the pipeline (e.g. num_channels, num_samples,
        num_classes). DL factories receive ctx; ML factories do not.

    Returns
    -------
    torch.nn.Module or estimator
    """
    ctx = ctx or {}
    model_kwargs = model_kwargs or {}

    if factory is None and name is None:
        raise ValueError("Must provide either 'name' (registry key) or 'factory' (callable).")

    if kind == "dl":
        # 1. Choose the factory to use: explicit or from registry
        actual_factory = factory
        if actual_factory is None:
            if name not in DL_MODEL_REGISTRY:
                raise KeyError(f"Unknown DL model '{name}'. Available: {sorted(DL_MODEL_REGISTRY)}")
            actual_factory = DL_MODEL_REGISTRY[name]

        # 2. Instantiate the model, passing both ctx and model_kwargs
        obj = actual_factory(**ctx, **model_kwargs)
        if not isinstance(obj, nn.Module):
            raise TypeError("DL factory did not return nn.Module.")
        return obj

    elif kind == "ml":
        # 1. Determine the factory to use: explicit or from registry
        actual_factory = factory
        if actual_factory is None:
            if name not in ML_MODEL_REGISTRY:
                raise KeyError(f"Unknown ML model '{name}'. Available: {sorted(ML_MODEL_REGISTRY)}")
            actual_factory = ML_MODEL_REGISTRY[name]

        # 2. Instantiate the estimator, passing only model_kwargs (no ctx)
        return actual_factory(**model_kwargs)

    else:
        raise ValueError(f"Unknown kind '{kind}'. Must be 'dl' or 'ml'.")


def build_model_from_spec(spec: ModelSpec, ctx: Dict[str, Any]) -> Union[nn.Module, MLEstimator]:
    """Convenience wrapper to build from a ModelSpec."""
    return build_model(kind=spec.kind, name=spec.name, model_kwargs=spec.kwargs, ctx=ctx)


# ---------------------------------------------------------------------------
# Classical ML registrations
# ---------------------------------------------------------------------------


@register_ml_model("random_forest")
def random_forest_factory(
    random_state: int = 0,
    **kwargs: Any,
) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    default: Dict[str, Any] = dict(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    default.update(kwargs)

    return RandomForestClassifier(**default)


# ---------------------------------------------------------------------------
# DL Registrations
# ---------------------------------------------------------------------------


@register_dl_model("speechnet")
def speechnet(
    *,
    num_channels: int,
    num_samples: int,
    num_classes: int,
    **model_kwargs,
) -> nn.Module:
    """
    Factory for SpeechNet.

    Required ctx keys:
      - num_channels
      - num_samples
      - num_classes

    Optional kwargs:
      - p_dropout
      - anything else (future-proof), passed through if supported
    """
    from models.cnn_architectures.SpeechNet import SpeechNet

    train_cfg = model_kwargs.get("train_cfg", {})
    loss_name = str(train_cfg.get("loss_name", "cross_entropy")).lower().strip()
    if loss_name not in {"ctc", "cross_entropy"}:
        raise ValueError(f"Unsupported loss_name='{loss_name}'.")

    use_ctc = loss_name == "ctc"
    output_classes = num_classes + (1 if use_ctc else 0)

    return SpeechNet(
        C=num_channels,
        T=num_samples,
        output_classes=output_classes,
        **model_kwargs,
    )


@register_dl_model("emg_transformer")
def emg_transformer(
    *, 
    num_channels: int, 
    num_samples: int,
    num_classes: int, 
    **model_kwargs
    ) -> nn.Module:

    """
    Factory for EMGTransformer.

    Required ctx keys:
      - num_channels
      - num_samples
      - num_classes
    """
    _ = num_samples
    try:
        from models.cnn_architectures.EMGTransformer import EMGTransformer
    except ImportError as exc:
        raise ImportError(
            "EMGTransformer not found. Make sure the architecture is implemented and importable."
        ) from exc

    # train_cfg is consumed by the trainer, not by the model constructor.
    model_kwargs = dict(model_kwargs)
    train_cfg = model_kwargs.get("train_cfg", {})
    loss_name = str(train_cfg.get("loss_name", "cross_entropy")).lower().strip()
    if loss_name not in {"ctc", "cross_entropy"}:
        raise ValueError(f"Unsupported loss_name='{loss_name}'.")

    use_ctc = loss_name == "ctc"
    model_kwargs.pop("train_cfg", None)

    return EMGTransformer(
        num_features=num_channels,
        num_outs=num_classes + (1 if use_ctc else 0),
        in_chans=num_channels,
        **model_kwargs,
    )
    
    
@register_dl_model("speechnet_transformer")
def speechnet_transformer(
    *,
    num_channels: int,
    num_samples: int,
    num_classes: int,
    **model_kwargs,
) -> nn.Module:
    """
    Factory for SpeechNetTransformer.

    Transformer variant of SpeechNet (BiLSTM replaced by a Transformer encoder,
    matched at equal parameter budget). Handles CE/CTC.

    Required ctx keys:
      - num_channels
      - num_samples
      - num_classes

    Optional kwargs:
      - d_model, nhead, num_layers, dim_feedforward (Transformer sizing)
      - any SpeechNet architecture argument (domain, blocks_config, mfcc_cfg, ...)
    """
    from models.cnn_architectures.SpeechNetTransformer import SpeechNetTransformer

    train_cfg = model_kwargs.get("train_cfg", {})
    loss_name = str(train_cfg.get("loss_name", "cross_entropy")).lower().strip()
    if loss_name not in {"ctc", "cross_entropy"}:
        raise ValueError(f"Unsupported loss_name='{loss_name}'.")

    use_ctc = loss_name == "ctc"
    output_classes = num_classes + (1 if use_ctc else 0)

    return SpeechNetTransformer(
        C=num_channels,
        T=num_samples,
        output_classes=output_classes,
        **model_kwargs,
    )