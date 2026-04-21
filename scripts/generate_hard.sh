#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.generate_demos \
  --envs hard \
  --episodes 1000 \
  --output-dir sneddy_baby_ai/artifacts/demos/hard
