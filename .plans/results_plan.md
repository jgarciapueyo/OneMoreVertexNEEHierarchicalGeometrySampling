# Experimental Plan: Validating BVH-Guided Intermediate Vertex Selection

## Goal

Validate that BVH importance sampling (specular-aware `computeNodeImportance_Unscented_LUT_Specular`)
correctly guides the selection of intermediate vertices in bidirectional path tracing, across a range
of material types, scene geometries, and integrator configurations.

The core comparison is always:
- **Uniform sampling** (`useUniformBVHSampling ≤ 0.0001`): no BVH guidance
- **BVH diffuse-only** (`enableSpecular=0`): BVH importance, diffuse approximation
- **BVH specular-aware** (`enableSpecular=1`): BVH importance, specular-aware approximation (with GGX masking)

---

## Phase 1: BSDF Variation

**Scientific question**: How does the specular-aware approximation compare to diffuse-only and uniform
sampling as material properties change?

**Geometry**: Quad soup, `k=40`, seed=42, 3000 quads (same positions for all variants).

### 1a — Uniform GGX specular (Disney principled, metallic=0)

Already partially done in `config-v9-disney.yaml`. Extend with uniform baseline:

| Sweep axis | Values |
|---|---|
| `geometry` | `quads-disney-rough0.1-k40.xml`, `quads-disney-rough0.3-k40.xml`, `quads-disney-rough0.6-k40.xml` |
| `enableSpecular` | 0, 1 |
| `useUniformBVHSampling` | 0.0001 (uniform), 1 (importance) |

Config: `config-v10-disney-uniform.yaml`

### 1b — Uniform metallic (Disney principled, metallic=1)

Generate geometry files: `quads-disney-metallic-rough0.1-k40.xml`, `...rough0.3...`, `...rough0.6...`

Command template:
```
python scenes/experiments/scene_geometries/create_geometry_disney.py \
  -n 3000 -f scenes/experiments/scene_geometries/quad_2tri.obj \
  -d -1 0 0 -k 40 -a 0 -10 1 -b 10 10 3 \
  --bsdf principled -r 0.8 --roughness <R> --specular 0.5 --metallic 1.0 \
  -o scenes/experiments/scene_geometries/quads-disney-metallic-rough<R>-k40.xml -s 42
```

> Note: `create_geometry_disney.py` currently hardcodes `metallic=0.0`. Add a `--metallic` flag.

| Sweep axis | Values |
|---|---|
| `geometry` | metallic rough0.1/0.3/0.6 |
| `enableSpecular` | 0, 1 |
| `useUniformBVHSampling` | 0.0001, 1 |

Config: `config-v11-disney-metallic.yaml`

### 1c — Heterogeneous materials (random per-quad BSDF)

Generate a single geometry file where each quad gets a randomly sampled BSDF:
- Randomly choose diffuse / specular / metallic per quad
- Roughness sampled uniformly from [0.1, 0.9]

This requires extending `create_geometry_disney.py` to support per-quad random BSDF assignment
(new `--bsdf random` mode or a separate script).

Config: `config-v12-disney-hetero.yaml`

---

## Phase 2: Geometry Variation (kappa sweep)

**Scientific question**: Does the approximation degrade gracefully as BVH node normals become less
concentrated (lower kappa → wider spread → `mu_n` is a worse representative)?

**BSDF**: Fixed principled, roughness=0.3, metallic=0 (from Phase 1 baseline).

### 2a — kappa sweep

Generate geometry at several concentration levels, same seed (42), same BSDF:

| File | kappa | Description |
|---|---|---|
| `quads-disney-rough0.3-k1.xml` | 1 | Near-uniform orientation |
| `quads-disney-rough0.3-k5.xml` | 5 | Moderate concentration |
| `quads-disney-rough0.3-k20.xml` | 20 | High concentration |
| `quads-disney-rough0.3-k40.xml` | 40 | Very tight (current default) |

| Sweep axis | Values |
|---|---|
| `geometry` | k1, k5, k20, k40 |
| `enableSpecular` | 0, 1 |
| `useUniformBVHSampling` | 0.0001, 1 |

Config: `config-v13-kappa.yaml`

Expected outcome: lower kappa should degrade specular-aware quality more than diffuse-only, since
the Gaussian quadrature relies on `mu_n` being representative.

---

## Phase 3: Ablation Study

**Scientific question**: What is the cost/quality tradeoff of each BVH integrator parameter?

**Scene**: Fixed `wildcard_disney_quads.xml`, roughness=0.3, k=40, `enableSpecular=1`.

### 3a — maxLeafSize

Controls BVH granularity. Smaller = finer leaves = better approximation, more traversal.

| Sweep axis | Values |
|---|---|
| `maxLeafSize` | 1, 2, 4, 8, 16 |

Config: `config-v14-leafsize.yaml`

### 3b — gaussianSAThreshold

Controls when to evaluate the node importance with 1 sigma point vs 7 (full Gaussian quadrature).
Larger threshold = more 1-point evaluations = faster but less accurate.

| Sweep axis | Values |
|---|---|
| `gaussianSAThreshold` | 0.001, 0.01, 0.031 (default), 0.1, 0.5 |

Config: `config-v15-gaussian.yaml`

### 3c — defensivePDF

Controls the defensive mixture weight (prevents zero-probability samples).

| Sweep axis | Values |
|---|---|
| `defensivePDF` | 0.0, 0.000001 (default), 0.001, 0.01 |

Config: `config-v16-defensive.yaml`

### 3d — maxDepth

Does BVH guidance provide more benefit at higher path depths?

| Sweep axis | Values |
|---|---|
| `maxDepth` | 2, 3, 4 |
| `enableSpecular` | 0, 1 |
| `useUniformBVHSampling` | 0.0001, 1 |

Config: `config-v17-depth.yaml`

---

## Execution Order

```
Phase 1a  →  Phase 1b  →  Phase 2a  →  Phase 3a  →  Phase 3b  →  Phase 3c  →  Phase 3d  →  Phase 1c
```

Phase 1c (heterogeneous) is last because it requires script changes and is most exploratory.

---

## Shared Infrastructure

All experiments use:
- Scene: `scenes/experiments/3_wildcard_disney/wildcard_disney_quads.xml`
- Reference: `wildcard_disney_quads_pathref.xml` (path tracer, 8192 SPP, per sweep config)
- `spp_schedule: [4, 8, 16, 32, 64, 128, 256, 512, 1024]`
- `repetitions_per_spp: 3`
- `analysis.average_bbox: [0.3, 0.3, 0.7, 0.7]` (lit spotlight region)
- `baseline_config_name: baseline_path`

---

## Code Changes Required

| Phase | Script/File | Change |
|---|---|---|
| 1a | `config-v10-disney-uniform.yaml` | New YAML (extend v9 with useUniformBVHSampling sweep) |
| 1b | `create_geometry_disney.py` | Add `--metallic` flag (currently hardcoded to 0.0) |
| 1b | `config-v11-disney-metallic.yaml` | New YAML |
| 2a | Geometry generation | Generate k1/k5/k20 variants |
| 2a | `config-v13-kappa.yaml` | New YAML |
| 3a–3d | `config-v14` through `config-v17` | New YAMLs |
| 1c | `create_geometry_disney.py` | Add `--bsdf random` mode or separate script |
| 1c | `config-v12-disney-hetero.yaml` | New YAML |
