"""Thin CLI for exporting leaderboard submissions."""

from __future__ import annotations

import argparse

from ..submission.exporter import DEFAULT_SUBMISSION_DIR, export_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a trained policy into demo_submission/.")
    parser.add_argument("--checkpoint", required=True, help="Exported checkpoint (.pt).")
    parser.add_argument("--output-dir", default=str(DEFAULT_SUBMISSION_DIR))
    parser.add_argument("--zip-output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    zip_path = export_submission(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        zip_output=args.zip_output,
    )
    print(zip_path)


if __name__ == "__main__":
    main()
