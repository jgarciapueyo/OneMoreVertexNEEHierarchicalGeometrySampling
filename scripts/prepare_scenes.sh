#!/usr/bin/env bash
# Generate the scene geometry that is not stored in the repository.
# Run from anywhere; files that already exist are kept.
set -euo pipefail
cd "$(dirname "$0")/.."

# Veach scene (Fig. 5): 30k quads with VMF(kappa=5) orientations. The
# placement used in the paper was not recorded; a fixed seed makes this
# file reproducible, but not identical to the paper's.
if [ ! -f scenes/scene_geometries/quads-k5.xml ]; then
    python3 scenes/scene_geometries/create_geometry.py -n=30000 -f=scenes/scene_geometries/quad_2tri.obj \
        -d -1 0 0 -k 5 -a 0 -10 1 -b 10 10 3 -s 0 -o scenes/scene_geometries/quads-k5.xml
fi

# Lighthouse scenes (Figs. 1 and 11): 50k snowflakes (seed 42)
if [ ! -f scenes/7_glints_lighthouse/50k_snowflakes.xml ]; then
    python3 scenes/7_glints_lighthouse/create_scene.py --snowflakes_only \
        --output scenes/7_glints_lighthouse/50k_snowflakes.xml
fi

# Sparsity experiment (Fig. 8): thinned hemispheres (deterministic)
if ! ls scenes/9_rebuttal_sphere/hemispheres_log/*.obj > /dev/null 2>&1; then
    python3 scenes/9_rebuttal_sphere/create_icospheres_densities.py \
        -o scenes/9_rebuttal_sphere/hemispheres_log > /dev/null
fi

echo "Scene geometry ready."
