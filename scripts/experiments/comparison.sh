#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/comparison.sh [experiments.yaml]
#
# Env overrides:
#   PYTHON=python
#   CONTAINER=bdpt_twopoints_mitsuba-mitsuba-1
#   WORKDIR=/workspace/bdpt_twopoints_mitsuba
#   SETPATH_SCRIPT=/home/mitsuba/setpath.sh

PYTHON="${PYTHON:-python}"
CONTAINER="${CONTAINER:-bdpt_twopoints_mitsuba-mitsuba-1}"
WORKDIR="${WORKDIR:-/home/mitsuba}"
SETPATH_SCRIPT="${SETPATH_SCRIPT:-/home/mitsuba/setpath.sh}"
EXP_FILE="${1:-scripts/experiments.yaml}"

echo "[1/2] Running experiments (host Python, Mitsuba in Docker container: ${CONTAINER})"
"${PYTHON}" scripts/experiments/run_experiments.py "${EXP_FILE}" \
  --resume \
  --container "${CONTAINER}" \
  --workdir "${WORKDIR}" \
  --setpath-script "${SETPATH_SCRIPT}"

echo "[2/2] Analyzing results (host Python)"
"${PYTHON}" scripts/experiments/analyze_results.py "${EXP_FILE}"

# if --older-metrics or -o is set, also run the older analysis for comparison
if [[ "${2:-}" == "--older-metrics" || "${2:-}" == "-o" ]]; then
  echo "[3/3] Analyzing results (host Python, older version of analysis for comparison)"
  "${PYTHON}" scripts/experiments/analyze_results_older.py "${EXP_FILE}"
fi

echo "Done."

