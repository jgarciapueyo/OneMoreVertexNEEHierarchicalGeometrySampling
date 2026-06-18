# Experiments Pipeline

This folder contains the experiment runner and analyzer used to sweep Mitsuba parameters, collect metrics, and produce comparison plots/tables.

## Files

- `run_experiments.py`: runs renders for all generated configs.
- `analyze_results.py`: computes metrics and generates comparison outputs.
- `comparison.sh`: convenience wrapper to run both steps.
- `config-v*.yaml`: immutable experiment configs.

## Quick Usage

### 1) Run + analyze (recommended)

```bash
bash scripts/experiments/comparison.sh scripts/experiments/config-v2.yaml
```

### 2) Run manually

```bash
python3 scripts/experiments/run_experiments.py scripts/experiments/config-v2.yaml --resume
python3 scripts/experiments/analyze_results.py scripts/experiments/config-v2.yaml
```

Use `--container`, `--workdir`, and `--setpath-script` on the runner to override YAML docker settings if needed.

## Config Format (YAML)

Top-level structure:

```yaml
docker:
  container: <docker-container-name>
  workdir: <path-inside-container>
  setpath_script: /home/mitsuba/setpath.sh

scene: <scene-path-visible-in-container>
output_root: <results-output-folder>

reference:
  inherit_config_params: false
  # Option A (fast): use pre-rendered EXR and skip per-config reference renders
  # pre_rendered_exr: <absolute-or-relative-exr-path>

  # Option B (render reference):
  scene: <reference-scene-path>
  spp: 8192
  # optional:
  # params:
  #   key: value
  # exclude_params: [param_a, param_b]

spp_schedule: [1, 2, 4, 8, 16, 32, 64]
repetitions_per_spp: 3

global_defaults:
  integrator: doublestep
  maxDepth: 3

named_configs:
  - name: baseline_path
    params:
      __scene: <optional-scene-override>
      integrator: path

sweep:
  leafSamplingMode: [Primitive, SphericalAABB]
  maxLeafSize: [1, 2, 4, 8]

analysis:
  baseline_config_name: baseline_path
  # optional: keep only baseline + N best techniques (by MSE at rank SPP)
  # n_top_techniques: 2
  top_k_configs: 6
  time_grid_points: 10
  inset_bbox: [0.7, 0.7, 0.75, 0.75]
  # optional global exposure for summary grid tonemapping:
  # exposure_scale: 1.0
  # optional explicit targets for rendered summary figures:
  # summary_equal_spp: 256
  # summary_equal_time_s: 0.40
```

LaTeX assets are auto-generated under:

- `output_root/latex/<test_name>_figures.tex`
- `output_root/latex/figs/<test_name>/summary_grid_spp.<ext>`
- `output_root/latex/figs/<test_name>/summary_grid_time.<ext>`
- `output_root/latex/figs/<test_name>/logscale_y/*.png|*.svg`

The generated LaTeX includes default quantitative plots for:

- `mean_flip` vs `spp`
- `mean_flip` vs `time`
- `mean_E` vs `spp`

## Key Config Fields

### `docker`

- `container`: target running Mitsuba container.
- `workdir`: working directory inside container.
- `setpath_script`: sourced before invoking `mitsuba`.

### `scene` and `output_root`

- `scene`: default scene used for tested runs.
- `output_root`: where all per-config folders and summary outputs are written.

### `reference`

Two modes:

1. **Pre-rendered reference (fast mode)**
   - Set `reference.pre_rendered_exr`.
   - Runner materializes `reference.exr` per config (symlink/hardlink/copy fallback).
   - `reference.scene` and `reference.spp` are not used for rendering.

2. **Rendered reference (legacy/default mode)**
   - Set `reference.scene` and `reference.spp`.
   - Optional `inherit_config_params`, `params`, and `exclude_params` control reference parameters.

### `spp_schedule` and `repetitions_per_spp`

- `spp_schedule`: sample counts evaluated per config.
- `repetitions_per_spp`: repeated runs for variance estimation.

### `global_defaults`, `named_configs`, `sweep`

- `global_defaults`: shared default parameters.
- `named_configs`: explicit config list with optional overrides.
- `sweep`: cartesian product grid.

Reserved runner key:

- `__scene` inside config params changes scene for that config only (not passed to Mitsuba as `-D`).

### `analysis`

- `baseline_config_name`: preferred baseline for comparisons.
- `top_k_configs`: number of top configs highlighted in summary plots.
- `time_grid_points`: number of synthetic equal-time points when baseline times are unavailable.
- `inset_bbox`: normalized inset box `[x0, y0, x1, y1]` used in rendered summary figures.
- `summary_equal_spp`: optional explicit SPP target for rendered equal-samples summary.
- `summary_equal_time_s`: optional explicit time target (seconds) for rendered equal-time summary.

## Naming Behavior

Config folders use short names derived from sweep/named config labels (for example: `Primitive_1`, `SphericalAABB_2`).

- If two configs resolve to the same short name, suffixes `_2`, `_3`, ... are appended deterministically.

## Output Highlights

Per config:

- `metrics.csv`, `metrics_summary.csv`
- `convergence_psnr.png`

Cross-config:

- `cross_config_equal_spp.csv`
- `cross_config_equal_time.csv`
- `cross_config_summary.csv`

Grouped summaries:

- `grouped_metrics_long.csv`
- `grouped_config_overview.csv`
- `grouped_best_by_spp.csv`
- `grouped_best_by_time.csv`
- `grouped_best_mse_by_spp.csv`
- `render_summary_equal_spp.png`
- `render_summary_equal_time.png`
- plots for PSNR/MSE/variance/pareto summaries

## Notes

- Prefer creating a new config file (`config-v3.yaml`, `config-v4.yaml`, ...) instead of mutating historical configs.
- Avoid `docker exec -it ...` in scripts; non-interactive execution is required for reliable capture.
