# Experiment Pipeline

## How it works

### 1. Geometry generation

`scenes/scene_geometries/create_geometry.py` (and `create_geometry_disney.py`) generates
Mitsuba XML files on demand, scattering many small quads or asteroids using vMF-distributed
orientations. The per-scene XML files (e.g. `wildcard_spotlight.xml`) then `<include>` those
generated geometry XMLs via a `geometry` parameter.

### 2. Scene XML pairs

Every scene directory holds two flavours:

- A *test* scene (e.g. `wildcard_spotlight.xml`) that loads the additional-vertex / doublestep integrator.
- A *path-tracer reference* scene (`*_pathref.xml`) used to render the ground truth.

### 3. Config YAML

Drives everything. Declares:

- Docker container + workdir
- Scene + reference scene (or a pre-rendered EXR path via `reference.pre_rendered_exr`)
- SPP schedule and repetitions per SPP
- `global_defaults`: shared integrator parameters for all tested configs
- `named_configs` and/or `sweep`: the actual technique variants to compare
- `analysis`: metric settings, aspect ratio for LaTeX figure export

### 4. Entry point

```bash
bash scripts/experiments/comparison.sh scripts/experiments/XXXX.yaml
```

Calls `run_experiments.py` (renders all configs inside Docker) then `analyze_results.py`
(FLIP/PSNR/MSE metrics → CSV + SVG/PNG plots + LaTeX `.tex` stub).

### 5. Sphere rebuttal (`scenes/9_rebuttal_sphere/`)

Completely separate pipeline. Has its own `run_hemispheres.py` and scene generators. No
corresponding YAML in `scripts/experiments/`. Different flow altogether.

---

## Config classification

### Strongest signals for "paper-final"

- `is_ablation: true` in the `analysis` block
- `enableFastLUT: 1` (late optimisation added near submission)
- `figure_format: svg` in `analysis`
- Uses `ablate*` parameter flags (`ablateGaussian`, `ablateVMF`, `ablateCosineXs`, …)
- Listed in `comparison_disney.sh`

### Paper-final configs

| File | Purpose |
|---|---|
| `ablation/ablation_asteroids5_trueablation.yaml` | True ablation — far asteroids scene (`2_wildcard_asteroids`) |
| `ablation/ablation_asteroids6_smallerspotlight_trueablation.yaml` | True ablation — close spotlight scene (`22_wildcard_asteroids_smallerspotlight`); most evolved, has `is_ablation: true` + `figure_format: svg` |
| `ablation/ablation2_veach.yaml` | True ablation — Veach scene; has `is_ablation: true`, full `ablate*` flags, `aspect_ratio` block for paper figure |
| `config-v9.A1` – `config-v9.A6` | Disney BRDF sweep (diffuse → specular → metal × two roughness levels); explicitly listed in `comparison_disney.sh` |
| `config-v9.B1`, `B2`, `B3` | Disney geometry distribution variants (random-homogeneous, grouped-clusters, heterogeneous); also in `comparison_disney.sh` |
| `config-v10-veach.yaml` | Main Veach scene comparison (path vs. doublestep variants) |
| `config-v12.A-glint.yaml`, `config-v12.B-glint.yaml` | Glints/lighthouse scenes; highest version number |

### Gray area (PPG comparisons)

Whether these matter depends on whether a PPG comparison appears in the paper:

- `ablation/ablation_asteroids3_ppg.yaml` — PPG vs. ours, far asteroids
- `ablation/ablation_asteroids4_smallerspotlight_ppg.yaml` — PPG vs. ours, close spotlight (note: wrong scene path `scenes/22...` without `experiments/`, may never have run successfully)
- `ablation/ablationv2_asteroids.yaml` — has `is_ablation: true` and `ablate*` flags, but `ablation_asteroids5_trueablation` appears to supersede it

### Deprecated / early-development

| File(s) | Why deprecated |
|---|---|
| `config-v0.yaml` | Scene path has no `experiments/` prefix (`scenes/cbox.xml`); very first smoke test |
| `config-v1.yaml` | No `experiments/` prefix; first wildcard spotlight test |
| `config-v2.yaml` | First leaf-sampling-mode comparison; sweeps `maxLeafSize` grid — parameter exploration |
| `config-v3-less.yaml`, `v5-wide.yaml`, `v6-k5-30k.yaml`, `v7-k5-30k.yaml` | All on `1_wildcard_spotlight`; various parameter searches |
| `config-v8-asteroids.yaml` | First asteroids run (`8_asteroids_v6`); versioned dev sweep |
| `config-v9.1` – `config-v9.10` (numeric, no letter suffix) | Pre-A/B-series Disney sweeps with coarser specular steps; superseded by the A/B series |
| `config-v11-cbox-n1.yaml` … `config-v11-cbox-n100.yaml` | Cornell box sanity checks sweeping quad count; probably never in the paper |
| `ablation/ablation_asteroids.yaml` | Old scene path (`scenes/2_wildcard_asteroids/` without `experiments/`); very early |
| `ablation/ablation_asteroids3.yaml` | Sweeps `bvhMaxDepth` 5–30; architectural parameter search |
| `ablation/ablation_asteroids3_gaussianSA.yaml` | Sweeps Gaussian SA threshold; architectural search |
| `ablation/ablation_quads_geometries.yaml`, `-2.yaml`, `-3.yaml`, `-4-picks.yaml` | Progressive iterations of quads-with-spotlight ablation on `1_wildcard_spotlight` (not the final scenes) |
| `ablation/ablation_veach.yaml` | Tests `leafSamplingMode` + `bvhMaxDepth` on Veach; parameter search, not the true ablation |
| `ablation/ablation_veach_angularsah.yaml`, `angularsah_config-v9.A6-*.yaml` | Angular-SAH experiments; comparison that did not make the final paper |
| `ablation/ablation_disney_spec.yaml` | Disney specular ablation with `bvhMaxDepth` sweep and `SphericalAABB_anyNode` mode |
