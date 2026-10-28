#!/usr/bin/env bash
# Compile Mitsuba (this repository) inside the container:
#     docker compose run --rm mitsuba bash scripts/build_mitsuba.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f config.py ]; then
    cp build/config-linux-gcc.py config.py
fi
python2.7 "$(which scons)" -j"$(nproc)" "$@"
echo "Mitsuba built in dist/ (source setpath.sh to use it)."
