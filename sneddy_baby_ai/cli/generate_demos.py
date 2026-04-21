"""Thin CLI for BabyAI demonstration generation."""

from __future__ import annotations

import argparse

from ..config.loader import list_config_presets
from ..data.generation import generate_demo_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate BabyAI expert demonstrations.")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--env", help="Single env name, e.g. BabyAI-GoToObj-v0.")
    selector.add_argument("--envs", help="Comma-separated envs or aliases: easy, moderate, hard, all.")
    parser.add_argument("--output", help="Single output .npz path for --env mode.")
    parser.add_argument("--output-dir", help="Target directory for --envs mode.")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-path", help="Optional mission vocabulary path.")
    parser.add_argument("--config", default="default", choices=list_config_presets())
    parser.add_argument("--time-limit-sec", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_demo_suite(
        env=args.env,
        envs=args.envs,
        output=args.output,
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
        vocab_path=args.vocab_path,
        config_name=args.config,
        time_limit_sec=args.time_limit_sec,
    )


if __name__ == "__main__":
    main()
