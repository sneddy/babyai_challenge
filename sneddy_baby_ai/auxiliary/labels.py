"""Auxiliary label extraction from live BabyAI environments."""

from __future__ import annotations

from typing import Any

import numpy as np

from .specs import AuxHeadSpec, IN_FRONT_OF_CLASSES


IN_FRONT_OF_TO_ID = {name: index for index, name in enumerate(IN_FRONT_OF_CLASSES)}


def build_aux_targets(env: Any, aux_specs: tuple[AuxHeadSpec, ...]) -> tuple[dict[str, int], dict[str, int]]:
    labels: dict[str, int] = {}
    masks: dict[str, int] = {}

    instruction = getattr(env, "instrs", None)
    putnext_instruction = _find_putnext_instruction(instruction)
    carry_desc = _find_carry_target_desc(instruction)
    fixed_desc = getattr(putnext_instruction, "desc_fixed", None) if putnext_instruction is not None else None
    referenced_descs = _collect_obj_descs(instruction)

    for spec in aux_specs:
        label, mask = _compute_head(
            spec.name,
            env,
            referenced_descs,
            carry_desc,
            fixed_desc,
            putnext_instruction,
        )
        labels[spec.name] = int(label)
        masks[spec.name] = int(mask)

    return labels, masks


def _compute_head(name: str, env: Any, referenced_descs, carry_desc, fixed_desc, putnext_instruction) -> tuple[int, int]:
    if name == "in_front_of_what":
        return _encode_front_object(env), 1
    if name == "obj_in_instr_visible":
        if not referenced_descs:
            return 0, 0
        return int(any(_desc_is_visible(env, desc) for desc in referenced_descs)), 1
    if name == "holding_target_object":
        if carry_desc is None:
            return 0, 0
        return int(_carrying_matches_desc(env, carry_desc)), 1
    if name == "adjacent_to_target_object":
        if putnext_instruction is None:
            return 0, 0
        return int(_putnext_alignment(env, putnext_instruction, require_empty_front=False)), 1
    if name == "valid_drop_position":
        if putnext_instruction is None:
            return 0, 0
        return int(_putnext_alignment(env, putnext_instruction, require_empty_front=True)), 1
    if name == "valid_pickup_action":
        if carry_desc is None:
            return 0, 0
        return int(_valid_pickup_action(env, carry_desc)), 1
    if name == "fixed_target_visible":
        if fixed_desc is None:
            return 0, 0
        return int(_desc_is_visible(env, fixed_desc)), 1
    if name == "need_pickup_phase":
        if putnext_instruction is None or carry_desc is None:
            return 0, 0
        return int(not _carrying_matches_desc(env, carry_desc)), 1
    if name == "need_drop_phase":
        if putnext_instruction is None or carry_desc is None:
            return 0, 0
        return int(_carrying_matches_desc(env, carry_desc)), 1
    raise ValueError(f"Unsupported auxiliary head: {name}")


def _encode_front_object(env: Any) -> int:
    front_cell = _get_front_cell(env)
    if front_cell is None:
        return IN_FRONT_OF_TO_ID["empty"]

    cell_type = str(getattr(front_cell, "type", "other"))
    if cell_type == "wall":
        return IN_FRONT_OF_TO_ID["wall"]
    if cell_type == "door":
        if bool(getattr(front_cell, "is_locked", False)):
            return IN_FRONT_OF_TO_ID["door_locked"]
        if bool(getattr(front_cell, "is_open", False)):
            return IN_FRONT_OF_TO_ID["door_open"]
        return IN_FRONT_OF_TO_ID["door_closed"]
    if cell_type in {"key", "ball", "box"}:
        return IN_FRONT_OF_TO_ID[cell_type]
    return IN_FRONT_OF_TO_ID["other"]


def _putnext_alignment(env: Any, instruction: Any, *, require_empty_front: bool) -> bool:
    move_desc = getattr(instruction, "desc_move", None)
    fixed_desc = getattr(instruction, "desc_fixed", None)
    if move_desc is None or fixed_desc is None:
        return False
    if not _carrying_matches_desc(env, move_desc):
        return False

    front_pos = _normalize_pos(getattr(env, "front_pos", None))
    if front_pos is None:
        return False
    if require_empty_front and _get_front_cell(env) is not None:
        return False

    for fixed_pos in _iter_desc_positions(env, fixed_desc):
        if _pos_next_to(front_pos, fixed_pos):
            return True
    return False


def _desc_is_visible(env: Any, desc: Any) -> bool:
    for obj, pos in _iter_desc_objects_and_positions(env, desc):
        if obj is getattr(env, "carrying", None):
            continue
        if pos is None:
            continue
        if _env_in_view(env, pos):
            return True
    return False


def _valid_pickup_action(env: Any, desc: Any) -> bool:
    if getattr(env, "carrying", None) is not None:
        return False
    front_cell = _get_front_cell(env)
    if front_cell is None:
        return False
    for obj in getattr(desc, "obj_set", []):
        if obj is front_cell:
            return True
    return False


def _carrying_matches_desc(env: Any, desc: Any) -> bool:
    carrying = getattr(env, "carrying", None)
    if carrying is None:
        return False
    for obj in getattr(desc, "obj_set", []):
        if obj is carrying:
            return True
    return False


def _collect_obj_descs(instruction: Any) -> list[Any]:
    if instruction is None:
        return []

    descs: list[Any] = []
    for attr in ("desc", "desc_move", "desc_fixed"):
        desc = getattr(instruction, attr, None)
        if desc is not None:
            descs.append(desc)
    for attr in ("instr_a", "instr_b"):
        nested = getattr(instruction, attr, None)
        if nested is not None:
            descs.extend(_collect_obj_descs(nested))
    return descs


def _find_putnext_instruction(instruction: Any) -> Any | None:
    if instruction is None:
        return None
    if hasattr(instruction, "desc_move") and hasattr(instruction, "desc_fixed"):
        return instruction
    for attr in ("instr_a", "instr_b"):
        nested = getattr(instruction, attr, None)
        found = _find_putnext_instruction(nested)
        if found is not None:
            return found
    return None


def _find_carry_target_desc(instruction: Any) -> Any | None:
    if instruction is None:
        return None
    if hasattr(instruction, "desc_move"):
        return getattr(instruction, "desc_move")
    if hasattr(instruction, "desc"):
        class_name = type(instruction).__name__.lower()
        if "pickup" in class_name:
            return getattr(instruction, "desc")
    for attr in ("instr_a", "instr_b"):
        nested = getattr(instruction, attr, None)
        found = _find_carry_target_desc(nested)
        if found is not None:
            return found
    return None


def _iter_desc_positions(env: Any, desc: Any):
    for _obj, pos in _iter_desc_objects_and_positions(env, desc):
        if pos is not None:
            yield pos


def _iter_desc_objects_and_positions(env: Any, desc: Any):
    objects = list(getattr(desc, "obj_set", []))
    positions = list(getattr(desc, "obj_poss", []))
    carrying = getattr(env, "carrying", None)

    for index, obj in enumerate(objects):
        pos = _object_current_pos(obj)
        if pos is None and index < len(positions):
            pos = _normalize_pos(positions[index])
        if obj is carrying:
            yield obj, None
            continue
        yield obj, pos


def _object_current_pos(obj: Any) -> tuple[int, int] | None:
    return _normalize_pos(getattr(obj, "cur_pos", None))


def _get_front_cell(env: Any) -> Any | None:
    front_pos = _normalize_pos(getattr(env, "front_pos", None))
    grid = getattr(env, "grid", None)
    if front_pos is None or grid is None or not hasattr(grid, "get"):
        return None
    return grid.get(*front_pos)


def _env_in_view(env: Any, pos: tuple[int, int]) -> bool:
    in_view = getattr(env, "in_view", None)
    if callable(in_view):
        return bool(in_view(*pos))
    return False


def _normalize_pos(pos: Any) -> tuple[int, int] | None:
    if pos is None:
        return None
    array = np.asarray(pos)
    if array.size != 2:
        return None
    return int(array[0]), int(array[1])


def _pos_next_to(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) == 1
