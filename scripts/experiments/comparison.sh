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
EXP_FILE="${1:-scripts/experiments.yaml}"

EXTRA_ARGS=()
if [[ -n "${CONTAINER:-}" ]]; then EXTRA_ARGS+=(--container "${CONTAINER}"); fi
if [[ -n "${WORKDIR:-}" ]]; then EXTRA_ARGS+=(--workdir "${WORKDIR}"); fi
if [[ -n "${SETPATH_SCRIPT:-}" ]]; then EXTRA_ARGS+=(--setpath-script "${SETPATH_SCRIPT}"); fi

echo "[1/2] Running experiments (host Python, Mitsuba in Docker)"
"${PYTHON}" scripts/experiments/run_experiments.py "${EXP_FILE}" \
  --resume \
  "${EXTRA_ARGS[@]}"

echo "[2/2] Analyzing results (host Python)"
"${PYTHON}" scripts/experiments/analyze_results.py "${EXP_FILE}"

# if --older-metrics or -o is set, also run the older analysis for comparison
if [[ "${2:-}" == "--older-metrics" || "${2:-}" == "-o" ]]; then
  echo "[3/3] Analyzing results (host Python, older version of analysis for comparison)"
  "${PYTHON}" scripts/experiments/analyze_results_older.py "${EXP_FILE}"
fi

echo "Done."

