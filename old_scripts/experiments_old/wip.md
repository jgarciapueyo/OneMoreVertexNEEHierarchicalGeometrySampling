BDPT TwoPoints Mitsuba Experiment Pipeline - Iteration Notes
Date: 2026-03-19

Goal
- Run many Mitsuba configurations reproducibly.
- Save per-config renders/logs/timing.
- Compute convergence metrics against a reference.
- Produce comparison-ready tables/plots for ranking and tradeoff analysis.

Current assumptions
- Python runs on host.
- Mitsuba runs inside Docker container (container has Mitsuba, Python may be host-only).
- Scene and output paths are visible from container workdir mount.

Main files
- Experiment config examples:
  - scripts/experiments/config-v0.yaml
  - scripts/experiments/config-v1.yaml
  - scripts/experiments/config-v2.yaml
- Runner:
  - scripts/experiments/run_experiments.py
- Analyzer:
  - scripts/experiments/analyze_results.py
- Wrapper:
  - scripts/experiments/comparison.sh

Current config conventions
- scene: path inside container workdir (example: scenes/experiments/.../scene.xml)
- output_root: output folder for one experiment run
- global_defaults: default `-D key=value` params
- sweep: cartesian parameter grid
- named_configs: explicit additional configurations
- spp_schedule: tested spp values
- repetitions_per_spp: repeats per spp point
- reference:
  - inherit_config_params
  - scene / spp (rendered reference mode)
  - pre_rendered_exr (fast mode, skips reference renders)
- analysis:
  - baseline_config_name
  - top_k_configs
  - time_grid_points

Naming scheme (current)
- Config directory names are short and human-readable.
- Sweep configs use sweep-value-based labels, e.g.:
  - Primitive_1
  - SphericalAABB_2
- No `config_0001_` prefix and no hash suffix.
- If a short name collides, runner appends `_2`, `_3`, ... deterministically.

Docker execution model
- Python invokes Mitsuba with:
  - docker exec -w <workdir> <container> bash -lc "source <setpath.sh> && mitsuba ..."
- Do NOT use `-it` in scripted runs.
- Store both timing views:
  - wall_time_s (host measured)
  - mitsuba_time_s (parsed from Mitsuba output)

Expected output layout
- results/<experiment_name>/<config_name>/
  - config.json
  - reference.exr
  - reference.meta.json
  - stdout_reference.log / stderr_reference.log (rendered-reference mode)
  - timings.csv
  - metrics.csv
  - metrics_summary.csv
  - convergence_psnr.png
  - runs/spp_XXXX(.meta/.stdout/.stderr/.exr)

- results/<experiment_name>/core cross-config outputs
  - summary.csv
  - cross_config_equal_spp.csv
  - cross_config_equal_time.csv
  - cross_config_summary.csv
  - cross_config_equal_time_delta_psnr.png

- results/<experiment_name>/grouped outputs (for easier downstream comparison)
  - grouped_metrics_long.csv
  - grouped_best_by_spp.csv
  - grouped_best_by_time.csv
  - grouped_best_mse_by_spp.csv
  - grouped_config_overview.csv
  - grouped_top_psnr_vs_spp.png
  - grouped_top_psnr_vs_time.png
  - grouped_top_mse_vs_spp.png
  - grouped_best_config_wins.png
  - grouped_variance_summary.png
  - grouped_pareto_peak_psnr_time.png

Metrics now tracked in analysis tables
- Quality:
  - PSNR, MSE, RMSE
- Variance:
  - var_mse, var_psnr, var_effective_time_s (where available)
- Deltas vs baseline:
  - delta_psnr, delta_mse, delta_var_mse, delta_var_psnr
- Speed:
  - effective_time_s, speedup metrics, time-to-target metrics

How to run
1) Preferred wrapper:
   - bash scripts/experiments/comparison.sh scripts/experiments/config-v2.yaml

2) Or direct:
   - python3 scripts/experiments/run_experiments.py scripts/experiments/config-v2.yaml --resume --container <container_name> --workdir <container_workdir>
   - python3 scripts/experiments/analyze_results.py scripts/experiments/config-v2.yaml

Validation checklist after each run
- [ ] Each config folder exists and has config.json.
- [ ] reference.exr exists (or pre-rendered source was materialized).
- [ ] runs/spp_*.exr exists for scheduled spp values.
- [ ] *.stdout.log and *.stderr.log are saved for rendered runs.
- [ ] metrics.csv + metrics_summary.csv exist per config.
- [ ] Cross-config CSVs are generated.
- [ ] Grouped CSVs and grouped summary plots are generated.
- [ ] Baseline selected correctly (prefer named baseline_path).

Important comparison definitions
- Equal SPP: compare baseline and candidate at same spp.
- Equal Time: compare metrics at same time budget (baseline time points by default).
- Speedup-to-target: time_baseline_to_target_psnr / time_candidate_to_target_psnr.

Known risks / caveats
- EXR reading depends on Python packages/backend support.
- Mitsuba stdout format may change; timing regex may need updates.
- If baseline selection is wrong, comparisons can be misleading.
- Equal-time interpolation skips out-of-range regions.

Next iteration ideas
1) Retry-on-failure for unstable renders.
2) Parallel config execution with safe process pool.
3) Optional report export (Markdown/LaTeX).
4) Composite figure helper for paper-ready panels.
5) Deterministic seed policy per config/spp/repeat.
6) Compact experiment manifest for reproducibility.

Suggested host Python packages
- pyyaml
- numpy
- imageio
- matplotlib

Versioning note
- Keep each experiment config immutable (config-v0.yaml, config-v1.yaml, ...).
- Add a new config file for behavior changes; do not overwrite old configs.
