"""Auxiliary supervision registry and helpers."""

from .losses import compute_auxiliary_loss
from .specs import AuxHeadSpec, apply_aux_weight, get_aux_specs, list_aux_presets

__all__ = [
    "AuxHeadSpec",
    "apply_aux_weight",
    "compute_auxiliary_loss",
    "get_aux_specs",
    "list_aux_presets",
]
