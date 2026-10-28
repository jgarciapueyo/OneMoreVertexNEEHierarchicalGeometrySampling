#!/usr/bin/env bash
# Reproduce the results of a figure of the paper. Run inside the container
# from the repository root, after scripts/build_mitsuba.sh:
#
#     docker compose run --rm mitsuba bash scripts/reproduce.sh <figure> [--quick]
#
# figures: fig1 fig4 fig5 fig6 fig7 fig8 fig9 fig10 fig11 fig12
# --quick  low sample counts and a single repetition (minutes instead of
#          hours); the results only show the trends of the figures.
#
# Experiment results go to the output_root of each config in
# scripts/experiments/ (under scenes/*/results/), the others to results/.
set -eo pipefail
cd "$(dirname "$0")/.."

FIG="${1:-}"
QUICK=0
[ "${2:-}" = "--quick" ] && QUICK=1
if [ -z "$FIG" ]; then
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

source setpath.sh
bash scripts/prepare_scenes.sh

EXP=scripts/experiments
run_config() {
    local cfg="$1"
    if [ "$QUICK" = 1 ]; then
        cfg=$(python3 "$EXP/make_quick_config.py" "$cfg")
    fi
    echo "=== $cfg"
    python3 "$EXP/run_experiments.py" "$cfg" --resume --local
    python3 "$EXP/analyze_results.py" "$cfg"
}

render() {  # render <scene> <spp> <output.exr>
    mkdir -p "$(dirname "$3")"
    echo "=== $1 at $2 spp -> $3"
    mitsuba -z "$1" -Dspp="$2" -o "$3" > "${3%.exr}.log" 2>&1
    grep "Render time" "${3%.exr}.log" || true
}

case "$FIG" in
    fig1)   # Lighthouse, thin-lens camera: ours 2k spp vs PT 100k spp
        if [ "$QUICK" = 1 ]; then OURS=16; PT=128; else OURS=2048; PT=100000; fi
        render scenes/7_glints_lighthouse/lighthouse_thinlens_oneehgsptracermis.xml $OURS results/fig1/ours.exr
        render scenes/7_glints_lighthouse/lighthouse_thinlens_path.xml $PT results/fig1/pt.exr ;;
    fig11)  # Lighthouse, perspective camera, equal time: ours 2k spp vs PT 4500 spp
        if [ "$QUICK" = 1 ]; then OURS=16; PT=32; else OURS=2048; PT=4500; fi
        render scenes/7_glints_lighthouse/lighthouse_oneehgsptracermis.xml $OURS results/fig11/ours.exr
        render scenes/7_glints_lighthouse/lighthouse_path.xml $PT results/fig11/pt.exr ;;
    fig4)   # Importance visualization: per-node CSV and total variation distance
        mkdir -p results/fig4
        mitsuba -z scenes/0_importance_quads/importance_quads.xml -DoutputDebugCSV=true -Dspp=1 \
            -o results/fig4/render.exr > results/fig4/render.log 2>&1
        mv bvh_debug.csv results/fig4/
        python3 scripts/figures/fig4_importance_tvd.py results/fig4/bvh_debug.csv --plot results/fig4/tvd.pdf
        echo "Visualize with: python3 scripts/polyscope/view_bvh.py results/fig4/bvh_debug.csv (on the host)" ;;
    fig5)   run_config $EXP/ablationv2_asteroids.yaml
            run_config $EXP/ablation_asteroids6_smallerspotlight_trueablation.yaml
            run_config $EXP/ablation2_veach.yaml ;;
    fig6)   run_config $EXP/ablation_disney_spec.yaml ;;
    fig7)   run_config $EXP/ablation_asteroids3.yaml
            run_config $EXP/ablation_asteroids3_gaussianSA.yaml ;;
    fig8)   # Sparsity: hemispheres with varying area ratio and subdivisions
        if [ "$QUICK" = 1 ]; then ARGS="--spp 2 --ref-spp 64"; else ARGS=""; fi
        python3 scenes/9_rebuttal_sphere/run_hemispheres.py -i scenes/9_rebuttal_sphere/hemispheres_log \
            -r results/fig8/runs --resume $ARGS
        python3 scenes/9_rebuttal_sphere/analyze_hemisphere_runs.py -i results/fig8/runs ;;
    fig9)   run_config $EXP/ablation_asteroids3_ppg.yaml ;;
    fig10)  run_config $EXP/ablation_asteroids4_smallerspotlight_ppg.yaml ;;
    fig12)  run_config $EXP/config-v11-cbox-n100.yaml ;;
    *)      echo "Unknown figure '$FIG'"; exit 1 ;;
esac
