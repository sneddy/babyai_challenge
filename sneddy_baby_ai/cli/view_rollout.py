"""CLI for rendering BabyAI policy-vs-expert comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm.auto import tqdm

from ..analysis.rollout import compare_policy_vs_expert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render or save BabyAI policy-vs-expert comparisons.")
    parser.add_argument("--env", required=True, help="Env name, e.g. BabyAI-PutNextLocal-v0.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True, help="Exported .pt checkpoint for the policy side of the comparison.")
    parser.add_argument("--mode", choices=["compare"], default="compare")
    parser.add_argument(
        "--output-dir",
        default="sneddy_baby_ai/artifacts/rollouts",
        help="Directory where rendered GIFs will be saved.",
    )
    parser.add_argument("--count", type=int, default=10, help="Number of consecutive seeds to render.")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--duration-ms", type=int, default=180)
    return parser


def _env_stem(env_name: str) -> str:
    return env_name.replace("BabyAI-", "").replace("-v0", "").lower()


def main() -> None:
    args = build_parser().parse_args()

    if args.count <= 0:
        raise RuntimeError("--count must be positive.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_stem = _env_stem(args.env)
    saved_paths: list[Path] = []

    for offset in tqdm(range(args.count), desc=f"rollouts {env_stem}", unit="rollout"):
        seed = args.seed + offset
        output_path = output_dir / f"{env_stem}_{args.mode}_seed{seed}.{args.format}"
        policy_episode, expert_episode, gif_path = compare_policy_vs_expert(
            env_name=args.env,
            seed=seed,
            checkpoint_path=args.checkpoint,
            output_path=str(output_path),
            max_steps=args.max_steps,
            device=args.device,
            duration_ms=args.duration_ms,
            output_format=args.format,
        )
        print(
            f"{gif_path} | "
            f"policy status={policy_episode.status} success={policy_episode.success} reward={policy_episode.reward:.3f} steps={policy_episode.steps} | "
            f"expert status={expert_episode.status} success={expert_episode.success} reward={expert_episode.reward:.3f} steps={expert_episode.steps}"
        )
        saved_paths.append(gif_path)

    print(f"saved {len(saved_paths)} files to {output_dir}")


if __name__ == "__main__":
    main()
