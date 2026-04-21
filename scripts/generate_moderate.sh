#!/usr/bin/env bash
set -euo pipefail

python3 -m sneddy_baby_ai.cli.generate_demos \
  --envs moderate \
  --episodes 5000 \
  --output-dir sneddy_baby_ai/artifacts/demos/moderate \
  --aux-preset aux_v1
