"""Thin CLI for PPO training."""

from __future__ import annotations

import argparse

from ..config.loader import list_config_presets, list_model_presets
from ..envs.curriculum import CURRICULUM_STAGES
from ..training.rl.service import run_train_rl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BabyAI PPO policy.")
    parser.add_argument("--env", help="Single env name, e.g. BabyAI-GoToObj-v0.")
    parser.add_argument("--envs", help="Comma-separated envs or aliases: easy, moderate, hard, all.")
    parser.add_argument("--stage", choices=[stage.name for stage in CURRICULUM_STAGES])
    parser.add_argument("--all-envs", action="store_true")
    parser.add_argument("--easy", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--config", default="default", choices=list_config_presets())
    parser.add_argument("--model-preset", choices=list_model_presets())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, help="Override total PPO timesteps.")
    parser.add_argument("--resume", help="Resume from an SB3 .zip checkpoint.")
    parser.add_argument("--warm-start", help="Initialize from a PPO .zip or exported .pt checkpoint.")
    parser.add_argument("--bc-checkpoint", help="Warm-start from an exported BC checkpoint.")
    parser.add_argument("--bc-demos", nargs="+", help="Train BC first from demos, then warm-start PPO from it.")
    parser.add_argument(
        "--bc-min-sampling-proba",
        type=float,
        default=None,
        help="Override adaptive demo sampling floor during BC warm-start generation.",
    )
    parser.add_argument("--rebuild-vocab", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_train_rl(
        env=args.env,
        envs=args.envs,
        stage=args.stage,
        all_envs=args.all_envs,
        easy=args.easy,
        moderate=args.moderate,
        hard=args.hard,
        run_name=args.run_name,
        config_name=args.config,
        model_preset=args.model_preset,
        seed=args.seed,
        total_timesteps=args.timesteps,
        resume_path=args.resume,
        warm_start_path=args.warm_start,
        bc_checkpoint_path=args.bc_checkpoint,
        bc_demo_paths=args.bc_demos,
        bc_min_sampling_proba=args.bc_min_sampling_proba,
        rebuild_vocab=args.rebuild_vocab,
    )


if __name__ == "__main__":
    main()
