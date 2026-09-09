"""Lightweight model class registry.

Models register themselves with :func:`register_model`; :class:`ModelRunner`
reads :data:`MODEL_REGISTRY` when no ``model_class`` is supplied. The fork's
text-path default (``qwen3``) is registered from ``qwen3.py``.

ponytail: dict-based; no decorator-state registry needed until a third
model class joins. Until then, ``model_class=...`` kwargs suffice.
"""

from __future__ import annotations

from typing import Type

MODEL_REGISTRY: dict[str, Type] = {}


def register_model(name: str):
    """Decorator: register ``cls`` under ``name`` in :data:`MODEL_REGISTRY`."""

    def decorator(cls: Type) -> Type:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type:
    """Look up a registered model class. Raises :class:`KeyError` if unknown."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model name {name!r}; registered: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


__all__ = ["MODEL_REGISTRY", "get_model_class", "register_model"]