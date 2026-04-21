"""Thin CLI for local submission evaluation."""

from __future__ import annotations

import argparse
import json

from ..submission.evaluator import evaluate_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an exported submission against a seed file.")
    parser.add_argument("--submission-dir", default="demo_submission")
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = evaluate_submission(
        submission_dir=args.submission_dir,
        seed_file=args.seed_file,
        max_steps=args.max_steps,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
