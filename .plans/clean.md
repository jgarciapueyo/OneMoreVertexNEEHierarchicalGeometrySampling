# Repository Cleanup Plan

Goal: produce a clean, public-facing repository for the EGSR 2026 paper
"One-More-Vertex NEE with Hierarchical Geometry Sampling."

---

## What we implemented (inventory of changes)

### 1. Core contribution: GeometryBVH (`include/mitsuba/render/bvh/`, `src/librender/bvh/`)

The central data structure — a BVH over scene geometry where each node stores
geometric aggregates (surface area, albedo, spherical AABB, VMF-based directional
importance) used to importance-sample sparse geometry without relying on ray casting.

Key files:
- `include/mitsuba/render/bvh/bvh.h` — main class definition, node layout, sampling API
- `include/mitsuba/render/bvh/intensity_grid.h` — VMF-based directional intensity grid
- `include/mitsuba/render/bvh/spherical_aabb.h` — spherical AABB representation
- `include/mitsuba/render/bvh/vmf_*.h` — polynomial/tensor/unrolled VMF weight tables
- `src/librender/bvh/bvh_build.cpp` — BVH construction and leaf-node aggregation
- `src/librender/bvh/bvh_fitting.cpp` — VMF fitting to leaf normals
- `src/librender/bvh/bvh_importance.cpp` — per-node importance computation (traversal weights)
- `src/librender/bvh/bvh_sample.cpp` — hierarchical sampling traversal (Primitive and SphericalAABB modes)
- `src/librender/bvh/spherical_aabb.cpp` — solid-angle AABB sampling
- `src/geometrybvh/geometrybvh.cpp` — thin Mitsuba plugin wrapper (lets scenes declare `<geometrybvh>`)

### 2. New integrators (`src/integrators/`)

- `additionalvertex/additionalvertex.cpp` — **main integrator**: path tracer extended with
  one-more-vertex NEE step that samples a surface point from the GeometryBVH and connects
  it as an intermediate vertex.
- `additionalvertex/doublestep.cpp` — variant that performs two geometry-BVH sampling steps.
- `additionalvertex/fastdoublestep.cpp` — optimised double-step variant.
- `additionalvertex/sampleGeometryExplicit.cpp/.h` — brute-force reference sampler (iterates
  all triangles; used to validate BVH sampling).
- `glinttracer/glinttracer.cpp/.proc` — particle-tracing integrator for glint rendering.
- `glinttracermis/glinttracermis.cpp` etc. — MIS-combined glint tracer (camera + light paths).
- `path/pathindirect.cpp` — path tracer that skips direct illumination (used as a baseline component).

### 3. New plugins

- `src/bsdfs/basicprincipled.cpp` — simplified Disney-principled BSDF used in experiment scenes.
- `src/emitters/aabblight.cpp` — axis-aligned bounding-box area light.
- `src/emitters/area_projected.cpp` — projected-area emitter variant.

### 4. VMF LUT

A precomputed Von Mises-Fisher lookup table used for fast directional importance evaluation:
- `data/vmf_lut.bin` — binary LUT (authoritative copy)
- `scripts/LUT/` — Python scripts that generate and fit the LUT

### 5. Modifications to existing Mitsuba 0.6

Small, surgical changes to integrate the BVH into the renderer:
- `include/mitsuba/render/scene.h` + `src/librender/scene.cpp` — added `m_geometryBVH` member,
  `getGeometryBVH()` accessor, and build-at-load-time logic.
- `include/mitsuba/render/scenehandler.h` + `src/librender/scenehandler.cpp` — XML parser
  recognises `<geometrybvh>` child elements.
- `include/mitsuba/render/bsdf.h` + `src/librender/bsdf.cpp` — minor albedo/roughness API additions.
- `include/mitsuba/render/emitter.h` — minor additions.
- `include/mitsuba/core/fwd.h` — forward declarations for BVH types.
- `include/mitsuba/core/matrix.h` — added matrix utility functions.
- `src/librender/skdtree.cpp` — removed an earlier prototype (kdtree-based aggregates) superseded by GeometryBVH.
- `src/libcore/vmf.cpp` — VMF distribution helpers.
- `src/shapes/obj.cpp`, `src/shapes/ply.cpp` — store per-primitive albedo/area during mesh loading.
- `data/schema/scene.xsd` — XML schema updated for new plugin types.
- `SConstruct` — added `src/geometrybvh/` to the build.

### 6. Tests (`src/tests/`)

Unit tests covering BVH phases: aggregation, construction, intersection, sampling (area,
AABB, primitive), importance computation, and SGGX moments.

### 7. Scenes (`scenes/`)

Scenes used in the paper experiments:
- `scenes/experiments/0_bvhintegrators_spotlight/` — single spotlight illuminating quads.
- `scenes/experiments/1_wildcard_spotlight/` — wildcard geometry with spotlight.
- `scenes/experiments/2_wildcard_asteroids/` — asteroid-shaped sparse quads.
- `scenes/experiments/22_wildcard_asteroids_smallerspotlight/` — tighter spotlight variant.
- `scenes/experiments/3_wildcard_disney/` — Disney BRDF ablation quads.
- `scenes/experiments/4_wildcard_veach/` — Veach MIS scene variant.
- `scenes/experiments/5_trees/`, `6_cbox/`, `7_glints*/`, `9_rebuttal_sphere/` — additional scenes.
- `scenes/basic/` — minimal scenes for quick smoke tests and sanity checks.

### 8. Experiment scripts (`scripts/experiments/`)

- `run_experiments.py` — sweeps parameter configs, runs Mitsuba inside Docker, saves EXRs.
- `analyze_results.py` — computes PSNR/MSE/FLIP metrics, generates plots and LaTeX assets.
- `comparison.sh` — convenience wrapper.
- `config-v*.yaml` and `ablation/*.yaml` — immutable run configs; many are historical dev versions.

### 9. Docker build (`Dockerfile`, `docker-compose.yml`, `compile_mitsuba.sh`)

Containerised build environment based on Ubuntu 20.04. `compile_mitsuba.sh` copies the
right config and invokes `scons`.

---

## Problems to fix (the actual cleanup steps)

### Step 1 — Remove internal development debris

Files that should never appear in a public repo:

| Path | Reason |
|---|---|
| `src/librender/bvh_OLD/` | (Done - removed) Superseded prototype BVH; all 6 files replaced by `src/librender/bvh/` |
| `src/librender/bvh/bvh_importance-nancatching.cpp` | (Done - removed) Debug variant, not compiled (absent from SConscript) |
| `scenes_old/` | (Done - removed but now they were in old_scenes_old) Development scratch scenes from earlier iterations |
| `data/vmf_lut_old.bin` | (Done - Removed) Superseded LUT |
| `include/mitsuba/render/bvh/vmf_lut.bin` | (Done - Removed) Duplicate of `data/vmf_lut.bin` (different path) |
| `scripts/LUT/vmf_lut.bin`, `scripts/LUT/vmf_lut.feather` | (Done - Removed) Yet more LUT copies |
| `scripts/LUT/old/` | (Done - removed) Old scripts to generate the LUT |
| `scripts/polyscope/` | (Don't remove yet) BVH debug visualisation dumps (CSV, PNG, ini files); not needed by users |
| `.plans/` (this directory) | (Don't remove yet) Internal implementation planning notes — move `clean.md` out, delete the rest |
| `ppg/` | (Don't remove) Full copy of Müller et al. 2017 PPG source — replace with a git submodule or note |
| `scripts/experiments/wip.md` | (Done - moved for now into `old_scripts`) Internal work-in-progress notes |
| `scripts/experiments/analyze_results copy.py` | (Done - moved for now into `old_scripts`) Accidental duplicate file |

| `implementation_plan_additional_vertex.md` | (Done) Already deleted in commit; verify it's gone |
| `src/integrators/additionalvertex/fastdoublestep-bkbad.cpp` | (Done - removed) "bkbad" — explicitly a broken backup |

Also audit:
- `scripts/experiments/config-v0.yaml` through `config-v9.*` — many are historical dev sweeps;
  decide which ones (if any) correspond to paper results and keep only those.

### Step 2 — Consolidate the VMF LUT

There are at least three copies of `vmf_lut.bin`. Decide on a single location
(`data/vmf_lut.bin` is the natural home) and make the code load from there.
Update `scripts/LUT/` to write there, and delete the duplicates.

### Step 3 — Write a new `README.md`

Replace the stock Mitsuba README with one that:
1. **What** — one paragraph explaining the method and paper citation.
2. **How it works** — brief technical sketch (sparse geometry → BVH → importance-sampled
   intermediate vertex → unbiased path integral).
3. **Repository structure** — callout box naming the key files/directories.
4. **Prerequisites** — Docker + docker-compose (no native build needed).
5. **Compile** — `docker compose run mitsuba bash compile_mitsuba.sh` (or equivalent).
6. **Quick test** — run one scene from `scenes/basic/`.
7. **Replicate paper results** — point to `scripts/experiments/README.md` and name
   which `config-v*.yaml` files correspond to each paper figure/table.
8. **Citation** — BibTeX block.

### Step 4 — Improve `compile_mitsuba.sh` and Docker setup

Current `compile_mitsuba.sh` is two lines with a comment mixed with `.gitignore` content
(file appears broken — the `.gitignore` suffix starts mid-file). Fix / expand it:
- Verify `build/config-linux-gcc.py` exists; fail with a helpful message if not.
- Accept an optional `--debug` flag.
- Print build directory on success.

Also verify `docker-compose.yml` works end-to-end and the `setpath.sh` sourcing is correct.

### Step 5 — Add a `run_scene.sh` helper

A simple script that:
- Takes a scene XML path (inside the container) and an output path.
- Sources `setpath.sh`.
- Invokes `mitsuba`.

This is the missing link between "I compiled Mitsuba" and "I can render something."

### Step 6 — Clean up experiment configs

- Identify which `config-v*.yaml` files correspond to the paper (likely the `v9.B*` sweep and
  the Veach + spotlight scenes based on the ablation structure).
- Move paper configs to a `scripts/experiments/paper/` subdirectory.
- Archive or delete the rest (or move to a `scripts/experiments/historical/` subdirectory if
  the results are needed for reproducibility).

### Step 7 — Add comments to the C++ code

Focus on non-obvious places:
- `include/mitsuba/render/bvh/bvh.h` — document the node layout and the aggregate fields.
- `src/librender/bvh/bvh_importance.cpp` — explain the importance formula (Section X of the paper).
- `src/librender/bvh/bvh_sample.cpp` — explain the hierarchical traversal and the two leaf-sampling modes.
- `src/librender/bvh/bvh_fitting.cpp` — explain VMF fitting.
- `src/integrators/additionalvertex/additionalvertex.cpp` — mark the "one extra vertex" loop
  extension clearly.
- `src/integrators/additionalvertex/sampleGeometryExplicit.cpp` — note it is a reference/debug
  sampler, not used in production.
- `src/geometrybvh/geometrybvh.cpp` — explain why this is a plugin wrapper rather than living
  in `librender`.

Remove surviving developer notes (e.g. "Nestor notes:", `// TODO:`, `// FIXME:`,
debug `printf`/`Log` calls) that are not meaningful to external readers.

### Step 8 — Tidy the `scenes/` directory

- Rename experiment sub-directories to drop numeric prefixes (or document what the numbers mean).
- Add a `scenes/README.md` that briefly describes what each scene is and where it appears
  in the paper.
- Check that all scene XML files reference assets (meshes, textures) with paths that work
  relative to the repo root inside the Docker container.

### Step 9 — (Optional) Submodule for PPG baseline

If comparisons against PPG are shown in the paper, `ppg/` should be a proper git submodule
pointing to the original repository, not a full inline copy. If comparisons are not in the
paper, remove it entirely.

---

## Suggested order of execution

1. Step 1 (debris removal) — biggest immediate win, can be reviewed in one commit.
2. Step 2 (LUT dedup) — quick, reduces confusion.
3. Step 3 (README) — highest impact for first-time readers.
4. Steps 4 + 5 (build/run scripts) — needed before anyone can try the code.
5. Step 6 (experiment configs) — needed before anyone can replicate results.
6. Step 7 (comments) — can be done incrementally file-by-file.
7. Step 8 (scenes) — polish pass.
8. Step 9 (PPG submodule) — decide based on paper content.
