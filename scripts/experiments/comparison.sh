#!/usr/bin/env bash
set -euo pipefail

# Usage (from the repository root):
#   bash scripts/experiments/comparison.sh <experiment.yaml>
#
# Inside the container Mitsuba is called directly. On the host it is called
# through `docker exec` in the running container; env overrides (otherwise
# taken from the `docker:` section of the YAML):
#   PYTHON=python3
#   CONTAINER=onemorevertexneehierarchicalgeometrysampling-mitsuba-1
#   WORKDIR=/home/mitsuba
#   SETPATH_SCRIPT=/home/mitsuba/setpath.sh

PYTHON="${PYTHON:-python3}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/experiments/comparison.sh <experiment.yaml>" >&2
  exit 1
fi
EXP_FILE="$1"

EXTRA_ARGS=()
if [[ -n "${CONTAINER:-}" ]]; then EXTRA_ARGS+=(--container "${CONTAINER}"); fi
if [[ -n "${WORKDIR:-}" ]]; then EXTRA_ARGS+=(--workdir "${WORKDIR}"); fi
if [[ -n "${SETPATH_SCRIPT:-}" ]]; then EXTRA_ARGS+=(--setpath-script "${SETPATH_SCRIPT}"); fi

echo "[1/2] Running experiments"
"${PYTHON}" scripts/experiments/run_experiments.py "${EXP_FILE}" \
  --resume \
  "${EXTRA_ARGS[@]}"

echo "[2/2] Analyzing results"
"${PYTHON}" scripts/experiments/analyze_results.py "${EXP_FILE}"

echo "Done."

