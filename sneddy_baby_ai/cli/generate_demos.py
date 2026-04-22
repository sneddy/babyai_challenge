"""Thin CLI for BabyAI demonstration generation."""

from __future__ import annotations

import argparse

from ..auxiliary.specs import list_aux_presets
from ..data.generation import generate_demo_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BabyAI expert demonstrations.")
    parser.add_argument("--envs", required=True, help="Comma-separated envs or aliases: easy, moderate, hard, all.")
    parser.add_argument("--output-dir", help="Target directory for --envs mode.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux-preset", choices=list_aux_presets(), help="Optional auxiliary supervision preset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_demo_suite(
        envs=args.envs,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        aux_preset=args.aux_preset,
    )


if __name__ == "__main__":
    main()
