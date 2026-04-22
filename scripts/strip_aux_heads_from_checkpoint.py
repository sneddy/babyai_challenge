#!/usr/bin/env python3
"""Remove selected auxiliary heads from an exported BabyAI checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to exported .pt checkpoint.")
    parser.add_argument("--output", required=True, help="Path to write the filtered .pt checkpoint.")
    parser.add_argument(
        "--drop-head",
        action="append",
        dest="drop_heads",
        required=True,
        help="Auxiliary head name to remove. Repeat for multiple heads.",
    )
    return parser


def _filter_aux_state(aux_state: dict[str, torch.Tensor] | None, drop_heads: set[str]) -> dict[str, torch.Tensor]:
    if not aux_state:
        return {}
    filtered: dict[str, torch.Tensor] = {}
    for key, value in aux_state.items():
        if not any(key.startswith(f"heads.{head}.") for head in drop_heads):
            filtered[key] = value
    return filtered


def _filter_aux_config(aux_config: dict | None, drop_heads: set[str]) -> dict | None:
    if not aux_config:
        return aux_config
    filtered = dict(aux_config)
    heads = filtered.get("heads")
    if isinstance(heads, list):
        filtered["heads"] = [head for head in heads if head.get("name") not in drop_heads]
    return filtered


def main() -> None:
    args = build_parser().parse_args()
    drop_heads = set(args.drop_heads)
    payload = torch.load(args.input, map_location="cpu")
    payload["aux_state"] = _filter_aux_state(payload.get("aux_state"), drop_heads)
    payload["aux_config"] = _filter_aux_config(payload.get("aux_config"), drop_heads)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print(f"wrote {output_path}")
    print(f"dropped heads: {sorted(drop_heads)}")
    print(f"remaining aux_state keys: {sorted((payload.get('aux_state') or {}).keys())}")
    if payload.get("aux_config"):
        print(f"remaining aux_config heads: {[head['name'] for head in payload['aux_config'].get('heads', [])]}")


if __name__ == "__main__":
    main()
