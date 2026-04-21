"""Masked auxiliary loss helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .specs import AuxHeadSpec


def compute_auxiliary_loss(
    *,
    aux_predictions: dict[str, torch.Tensor] | None,
    aux_targets: dict[str, torch.Tensor] | None,
    aux_masks: dict[str, torch.Tensor] | None,
    aux_specs: tuple[AuxHeadSpec, ...],
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not aux_specs:
        zero = torch.zeros((), device=device or torch.device("cpu"))
        return zero, {"aux_loss": 0.0, "aux_active": 0.0}

    if aux_predictions is None or aux_targets is None or aux_masks is None:
        raise ValueError("auxiliary predictions, targets, and masks must all be provided when aux_specs are enabled")

    if device is None:
        first_prediction = next(iter(aux_predictions.values()), None)
        device = first_prediction.device if first_prediction is not None else torch.device("cpu")

    total_loss = torch.zeros((), device=device)
    total_active = 0
    metrics: dict[str, float] = {"aux_loss": 0.0, "aux_active": 0.0}

    for spec in aux_specs:
        prediction = aux_predictions[spec.name]
        target = aux_targets[spec.name]
        mask = aux_masks[spec.name].reshape(-1).to(dtype=torch.bool)
        active_count = int(mask.sum().item())
        metrics[f"{spec.name}_active"] = float(active_count)
        if active_count == 0:
            metrics[f"{spec.name}_loss"] = 0.0
            metrics[f"{spec.name}_acc"] = 0.0
            continue

        if spec.head_type == "binary":
            flat_prediction = prediction.reshape(-1)
            flat_target = target.reshape(-1).float()
            per_item_loss = F.binary_cross_entropy_with_logits(
                flat_prediction,
                flat_target,
                reduction="none",
            )
            head_accuracy = ((flat_prediction > 0).float()[mask] == flat_target[mask]).float().mean()
        elif spec.head_type == "multiclass":
            per_item_loss = F.cross_entropy(
                prediction,
                target.long(),
                reduction="none",
            )
            head_accuracy = (prediction.argmax(dim=-1)[mask] == target.long()[mask]).float().mean()
        else:
            raise ValueError(f"Unsupported auxiliary head type: {spec.head_type}")

        head_loss = per_item_loss[mask].mean()
        total_loss = total_loss + spec.loss_weight * head_loss
        total_active += active_count
        metrics[f"{spec.name}_loss"] = float(head_loss.item())
        metrics[f"{spec.name}_acc"] = float(head_accuracy.item())

    metrics["aux_loss"] = float(total_loss.item())
    metrics["aux_active"] = float(total_active)
    return total_loss, metrics
