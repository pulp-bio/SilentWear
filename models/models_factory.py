# Copyright 2026 Giusy Spacone
# Copyright 2026 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union, Literal
import torch.nn as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------------------------------------
# Types
# -------------------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------------------
# Registration decorators
# -------------------------------------------------------------------------------------------------


def register_dl_model(name: str):
    """Decorator to register a deep learning model factory."""

    def deco(fn: DLFactory) -> DLFactory:
        if name in DL_MODEL_REGISTRY:
            raise ValueError(f"DL model '{name}' already registered.")
        DL_MODEL_REGISTRY[name] = fn
        return fn

    return deco


def register_ml_model(name: str):
    """Decorator to register a classical ML model factory."""

    def deco(fn: MLFactory) -> MLFactory:
        if name in ML_MODEL_REGISTRY:
            raise ValueError(f"ML model '{name}' already registered.")
        ML_MODEL_REGISTRY[name] = fn
        return fn

    return deco


# -------------------------------------------------------------------------------------------------
# Unified builders
# -------------------------------------------------------------------------------------------------


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
        Explicit factory callable. If provided, it takes precedence over registry lookup.
    model_kwargs:
        Optional model-specific kwargs (hyperparams, architecture params, etc.)
    ctx:
        Context dict produced by the pipeline. Factories may ignore keys they don't need.

        Suggested ctx keys (examples):
          - num_classes (int)
          - include rest class

    Returns
    -------
    torch.nn.Module or estimator
    """
    ctx = ctx or {}
    model_kwargs = model_kwargs or {}

    # 1) choose factory
    if factory is None:
        if name is None:
            raise ValueError("Provide either `name` (registry key) or `factory`.")
        if kind == "dl":
            if name not in DL_MODEL_REGISTRY:
                raise KeyError(f"Unknown DL model '{name}'. Available: {sorted(DL_MODEL_REGISTRY)}")
            factory = DL_MODEL_REGISTRY[name]
            # ctx contains num of classes, needed for pytorch models
            print("ctx")
            print(ctx)
            print("kwargs")
            print(model_kwargs)
            obj = factory(**ctx, **model_kwargs)
            if not isinstance(obj, nn.Module):
                raise TypeError("DL factory did not return nn.Module.")

        elif kind == "ml":
            if name not in ML_MODEL_REGISTRY:
                raise KeyError(f"Unknown ML model '{name}'. Available: {sorted(ML_MODEL_REGISTRY)}")
            factory = ML_MODEL_REGISTRY[name]
            # don't pass ctx
            obj = factory(**model_kwargs)
        else:
            raise ValueError(f"Unknown kind '{kind}'. Must be 'dl' or 'ml'.")

    if not callable(factory):
        raise TypeError("Factory must be callable.")

    return obj


def build_model_from_spec(spec: ModelSpec, ctx: Dict[str, Any]) -> Union[nn.Module, MLEstimator]:
    """Convenience wrapper to build from a ModelSpec."""
    return build_model(kind=spec.kind, name=spec.name, model_kwargs=spec.kwargs, ctx=ctx)


# -------------------------------------------------------------------------------------------------
# Classical ML registrations
# -------------------------------------------------------------------------------------------------


@register_ml_model("random_forest")
def random_forest_factory(
    random_state: int = 0,
    **kwargs,
):
    from sklearn.ensemble import RandomForestClassifier

    default = dict(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    default.update(kwargs)

    print("Registered Random Forest!")
    return RandomForestClassifier(**default)


# -------------------------------------------------------------------------------------------------
# DL Registrations
# -------------------------------------------------------------------------------------------------


@register_dl_model("speechnet")
def speechnet(
    *,
    num_channels: int,
    num_samples: int,
    num_classes: int,
    **model_kwargs,
) -> nn.Module:
    """
    Factory for EpiDeNet.

    Required ctx keys:
      - num_channels
      - num_samples
      - num_classes

    Optional kwargs:
      - p_dropout
      - anything else (future-proof), passed through if supported
    """
    from models.cnn_architectures.SpeechNet import SpeechNet

    # print("speech net base with kwars", model_kwargs)
    return SpeechNet(
        C=num_channels,
        T=num_samples,
        output_classes=num_classes,
        **model_kwargs,  # <-- passes blocks_config, dropout, etc.
    )
