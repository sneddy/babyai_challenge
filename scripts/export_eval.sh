#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: sh scripts/export_eval.sh <checkpoint.pt>" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -f "${CHECKPOINT_PATH}" ]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

CHECKPOINT_STEM="$(basename "${CHECKPOINT_PATH}" .pt)"
SUBMISSION_DIR="submissions"
RESULTS_DIR="evaluation/results"
ZIP_OUTPUT="${SUBMISSION_DIR}/${CHECKPOINT_STEM}.zip"
RENAMED_RESULT="${RESULTS_DIR}/${CHECKPOINT_STEM}.json"

mkdir -p "${SUBMISSION_DIR}"
mkdir -p "${RESULTS_DIR}"

TMP_BEFORE="$(mktemp)"
find "${RESULTS_DIR}" -maxdepth 1 -type f -name '*.json' -print | sort > "${TMP_BEFORE}"

"${PYTHON_BIN}" -m sneddy_baby_ai.cli.export_submission \
  --checkpoint "${CHECKPOINT_PATH}" \
  --zip-output "${ZIP_OUTPUT}"

"${PYTHON_BIN}" - "${ZIP_OUTPUT}" "${RESULTS_DIR}" <<'PY'
import runpy
import sys

import sneddy_baby_ai.envs.runtime  # noqa: F401

submission = sys.argv[1]
results_dir = sys.argv[2]
sys.argv = [
    "evaluate_submission.py",
    "--submission",
    submission,
    "--seed-dir",
    "babyai-ml8103-leaderboard-2026/eval_seeds",
    "--output-dir",
    results_dir,
]
runpy.run_path("babyai-ml8103-leaderboard-2026/evaluation/evaluate_submission.py", run_name="__main__")
PY

TMP_AFTER="$(mktemp)"
find "${RESULTS_DIR}" -maxdepth 1 -type f -name '*.json' -print | sort > "${TMP_AFTER}"

TMP_NEW_RESULT="$(mktemp)"
"${PYTHON_BIN}" - "${TMP_BEFORE}" "${TMP_AFTER}" "${TMP_NEW_RESULT}" "${ZIP_OUTPUT}" <<'PY'
import sys
import json
from pathlib import Path

before_path = Path(sys.argv[1])
after_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
zip_output = sys.argv[4]
before = {line.strip() for line in before_path.read_text(encoding="utf-8").splitlines() if line.strip()}
after = [line.strip() for line in after_path.read_text(encoding="utf-8").splitlines() if line.strip()]
new_files = [path for path in after if path not in before]
matches = []
for path in new_files:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("submission_path") == zip_output:
        matches.append(path)
if len(matches) != 1:
    raise SystemExit(
        f"Expected exactly one matching result json for {zip_output}, found {len(matches)}: {matches}"
    )
output_path.write_text(matches[0], encoding="utf-8")
PY

NEW_RESULT="$(cat "${TMP_NEW_RESULT}")"

mv "${NEW_RESULT}" "${RENAMED_RESULT}"

rm -f "${TMP_BEFORE}" "${TMP_AFTER}" "${TMP_NEW_RESULT}"

echo "Submission zip: ${ZIP_OUTPUT}"
echo "Evaluation json: ${RENAMED_RESULT}"
