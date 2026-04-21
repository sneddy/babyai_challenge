"""Auxiliary head preset definitions."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace


@dataclass(frozen=True)
class AuxHeadSpec:
    name: str
    head_type: str
    loss_weight: float = 0.1
    num_classes: int = 1

    def __post_init__(self) -> None:
        if self.head_type not in {"binary", "multiclass"}:
            raise ValueError(f"Unsupported auxiliary head type: {self.head_type}")
        if self.head_type == "binary" and self.num_classes != 1:
            raise ValueError("binary heads must use num_classes=1")
        if self.head_type == "multiclass" and self.num_classes <= 1:
            raise ValueError("multiclass heads must use num_classes > 1")

    @property
    def output_dim(self) -> int:
        return 1 if self.head_type == "binary" else self.num_classes

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "name": self.name,
            "head_type": self.head_type,
            "loss_weight": self.loss_weight,
            "num_classes": self.num_classes,
        }


IN_FRONT_OF_CLASSES = (
    "empty",
    "wall",
    "door_open",
    "door_closed",
    "door_locked",
    "key",
    "ball",
    "box",
    "other",
)


_AUX_PRESETS: dict[str, tuple[AuxHeadSpec, ...]] = {
    "aux_v1": (
        AuxHeadSpec(
            name="in_front_of_what",
            head_type="multiclass",
            num_classes=len(IN_FRONT_OF_CLASSES),
            loss_weight=0.2,
        ),
        AuxHeadSpec(name="obj_in_instr_visible", head_type="binary", loss_weight=0.1),
        AuxHeadSpec(name="holding_target_object", head_type="binary", loss_weight=0.1),
        AuxHeadSpec(name="adjacent_to_target_object", head_type="binary", loss_weight=0.1),
        AuxHeadSpec(name="valid_drop_position", head_type="binary", loss_weight=0.1),
        AuxHeadSpec(name="valid_pickup_action", head_type="binary", loss_weight=0.1),
        AuxHeadSpec(name="fixed_target_visible", head_type="binary", loss_weight=0.1),
    ),
}


def list_aux_presets() -> list[str]:
    return sorted(_AUX_PRESETS)


def get_aux_specs(preset_name: str | None) -> tuple[AuxHeadSpec, ...]:
    if not preset_name:
        return ()
    try:
        return _AUX_PRESETS[preset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown auxiliary preset: {preset_name}") from exc


def apply_aux_weight(
    aux_specs: tuple[AuxHeadSpec, ...],
    aux_weight: float | None,
) -> tuple[AuxHeadSpec, ...]:
    if aux_weight is None:
        return tuple(aux_specs)
    return tuple(replace(spec, loss_weight=float(aux_weight)) for spec in aux_specs)
