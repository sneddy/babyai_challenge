"""Thin CLI for behavior-cloning training."""

from __future__ import annotations

import argparse

from ..auxiliary.specs import list_aux_presets
from ..config.loader import list_config_presets, list_model_presets
from ..training.bc.service import parse_eval_envs, train_bc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BabyAI policy with behavior cloning.")
    parser.add_argument("--demos", nargs="+", required=True, help="One or more .npz demo files.")
    parser.add_argument("--checkpoint", required=True, help="Output checkpoint path (.pt).")
    parser.add_argument("--vocab", required=True, help="Mission vocabulary JSON path.")
    parser.add_argument("--config", default="default", choices=list_config_presets())
    parser.add_argument("--model-preset", choices=list_model_presets())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", help="Optional PPO .zip or exported .pt checkpoint.")
    parser.add_argument("--eval-envs", help="Comma-separated env names for holdout evaluation.")
    parser.add_argument(
        "--min-sampling-proba",
        type=float,
        default=None,
        help="Override adaptive demo sampling floor when holdout env evaluation is enabled.",
    )
    parser.add_argument("--recurrent", action="store_true", help="Train recurrent BC instead of feedforward BC.")
    parser.add_argument("--aux-preset", choices=list_aux_presets(), help="Optional auxiliary supervision preset.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_bc(
        demo_path=args.demos,
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        config_name=args.config,
        model_preset=args.model_preset,
        seed=args.seed,
        warm_start_path=args.warm_start,
        eval_env_names=parse_eval_envs(args.eval_envs),
        min_sampling_proba=args.min_sampling_proba,
        recurrent=args.recurrent,
        aux_preset=args.aux_preset,
    )


if __name__ == "__main__":
    main()
