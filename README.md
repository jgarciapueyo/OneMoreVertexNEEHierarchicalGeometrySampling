<p align="center">
  <h1 align="center"><a href="https://graphics.unizar.es/projects/OneMoreVertex26/">One-more-vertex Next-Event Estimation with Hierarchical Geometry Sampling</a></h1>
  <img src="images/teaser.png" alt="Teaser: our method (2k spp, 105 s) vs path tracing (100k spp, 36 min)" width="100%">
  <p align="center">
    Eurographics Symposium on Rendering (EGSR) 2026 - Computer Graphics Forum
    <br />
    <a href="https://jgarciapueyo.github.io/"><strong>Jorge Garcia-Pueyo*<sup>1</sup></strong></a>
    ·
    <a href="https://nestor98.github.io/"><strong>Nestor Monzon*<sup>1</sup></strong></a>
    ·
    <a href="https://orcid.org/0000-0001-9000-0466"><strong>Adrián Jarabo<sup>2</sup></strong></a>
    ·
    <a href="http://adolfo-munoz.com/"><strong>Adolfo Muñoz<sup>1</sup></strong></a>
    <br />
    *Equal Contribution
    &nbsp;&nbsp;
    <sup>1</sup>
      Universidad de Zaragoza, I3A, Spain
    &nbsp;&nbsp;
    <sup>2</sup>Independent Researcher, Spain
  </p>

  <p align="center">
    <a href='https://graphics.unizar.es/projects/OneMoreVertex26/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
    <a href='https://graphics.unizar.es/projects/OneMoreVertex26/static/omv_nee_paper_26.pdf' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='https://github.com/jgarciapueyo/OneMoreVertexNEEHierarchicalGeometrySampling' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Code-GitHub-green?style=flat-square' alt='Code'>
    </a>
  </p>
</p>

---

### Overview

Robust next-event estimation (NEE) remains a challenge in scenes characterized by **sparse or small-scale geometry** where indirect illumination is the primary transport mechanism. In these scenes, traditional path construction, which relies on local directional sampling, often fails to find intersections with the sparse geometry, and standard NEE also struggles as it typically connects vertices directly to emitters, failing when those connections are occluded or require intermediate bounces.

We propose a novel approach that constructs paths via **direct geometry sampling**. Instead of relying on stochastic ray casting, we repurpose the scene's bounding volume hierarchy (BVH) as a **hierarchical sampling structure**. By performing a stochastic top-down traversal, we transform the selection of the next path vertex into a hierarchical problem. To prioritize high-throughput connections, the traversal is guided by a **proxy contribution function** evaluated at each internal node, which leverages aggregated statistics of the geometry to efficiently estimate contribution during traversal. We demonstrate orders of magnitude improvements in complex scenarios such as indirect illumination from sparse geometry or rendering discrete scattering media.

---

### Method
<p align="center">
  <img src="images/method1.png" alt="One-more-vertex NEE diagram" width="50%">
</p>

Standard next-event estimation connects sensor vertex $\mathbf{x}_s$ directly to an emitter vertex $\mathbf{x}_e$. In scenes where $\mathbf{x}_s$ is mostly illuminated through an indirect light from the scene geometry $\mathcal{G}$, that direct connection has low to zero contribution (red dashed line). Our technique tackles this by importance-sampling an intermediate vertex $\mathbf{x}_n$ on the scene geometry $\mathcal{G}$. This allows us to efficiently sample high-throughput indirect paths (green) where the geometry acts as the primary source of reflected illumination for $\mathbf{x}_s$. Note that our approach also accounts for the case when $\mathbf{x}_s$ is on the sensor $\mathbf{x}_s$ = $\mathbf{x}_0$.

Formally, we estimate the following integral over the scene geometry:

$$L(\mathbf{x}_s \leftrightarrow \mathbf{x}_e) = \int_\mathcal{G} f(\mathbf{x}_s \leftrightarrow \mathbf{x}_n \leftrightarrow \mathbf{x}_e)\, dA(\mathbf{x}_n)$$

where $f(\mathbf{x}_s \leftrightarrow \mathbf{x}_n \leftrightarrow \mathbf{x}_e)$ encodes the BSDF, geometry, visibility, and radiance terms along the three-vertex subpath.

To importance-sample $\mathbf{x}_n$ efficiently, we build a hierarchical structure over the scene primitives that aggregates directional, spatial, and material statistics at each BVH node. At render time, a stochastic top-down traversal selects a child node $\mathcal{N}_j^{d+1}$ from its parent $\mathcal{N}_i^d$ proportionally to an approximated importance weight:

$$p\left(\mathcal{N}_j^{d+1} \mid \mathcal{N}_i^d\right) = \frac{w\left(\mathcal{N}_j^{d+1}, \mathbf{x}_s, \mathbf{x}_e\right)}{w\left(\mathcal{N}_i^d, \mathbf{x}_s,\, \mathbf{x}_e\right)}$$

<p align="center">
  <img src="images/method2.png" alt="Hierarchical sampling overview" width="90%">
</p>

> **Hierarchical One-more-vertex Sampling Overview**. (a) We wish to importance-sample high-throughput three-vertex subpaths between vertices $\mathbf{x}_s$, $\mathbf{x}_e$. Our method is based on a hierarchical structure of the scene primitives that (b) aggregates its directional, spatial, and material statistics (represented by the gray sectors) at each node bottom-up. (c) These statistics enable guiding the hierarchical sampling of the intermediate vertex $\mathbf{x}_n$ by evaluating the approximated per-node importance weight (yellow bars) for each node's children and choosing one proportionally to its weight.

---

### Results

<p align="center">
  <img src="images/fig2.png" alt="Comparison results on the Asteroids scene" width="90%">
</p>

Comparison against path tracing (PT), bidirectional path tracing (BDPT), and practical path guiding (PPG) on the Asteroids scene. At equal samples (16 SPP) our method produces visibly cleaner results at a fraction of the time. At equal time (2 s) the gap is even more pronounced, with our method achieving orders-of-magnitude lower MSE on both the SPP and time convergence curves.

---

### Quickstart

*Requirements*: [Docker](https://docs.docker.com/get-docker/) with Docker Compose. Everything else (Mitsuba 0.6 dependencies, the Python packages of the scripts and the PPG baseline) is installed in the image.

#### 1. Build the image and compile

```bash
git clone https://github.com/jgarciapueyo/OneMoreVertexNEEHierarchicalGeometrySampling.git
cd OneMoreVertexNEEHierarchicalGeometrySampling
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose build
docker compose run --rm mitsuba bash scripts/build_mitsuba.sh
```

The repository is mounted at `/home/mitsuba` inside the container, and the container runs with your user and group ids, so the build output (`build/`, `dist/`) and all results belong to you on the host. The image also contains the [Practical Path Guiding](https://github.com/Tom94/practical-path-guiding) baseline, built from a pinned commit with [`patches/ppg.patch`](patches/ppg.patch) (installed as `mitsuba-ppg`).

Quick sanity check: render the *AsteroidsCloseLight* scene with our method at 16 spp:

```bash
docker compose run --rm mitsuba mitsuba scenes/22_wildcard_asteroids_smallerspotlight/wildcard_spotlight.xml \
    -Dspp=16 -DleafSamplingMode=Primitive -o asteroids_ours.exr
```

For interactive work, open a shell with `docker compose run --rm mitsuba` (`mitsuba` is on the `PATH`).

#### 2. Reproduce the results of the paper

```bash
docker compose run --rm mitsuba bash scripts/reproduce.sh fig9           # full experiment
docker compose run --rm mitsuba bash scripts/reproduce.sh fig9 --quick   # a few minutes, same trends
```

[`scripts/reproduce.sh`](scripts/reproduce.sh) first generates the geometry that is not stored in the repository ([`scripts/prepare_scenes.sh`](scripts/prepare_scenes.sh)), then renders every technique of the figure, computes the metrics (relative MSE, FLIP, ...) and writes the plots and LaTeX assets to the `output_root` of each experiment config (under `scenes/*/results/`) or to `results/`. The full configs render a 16k-spp reference and sweep up to 2048 spp with 3 repetitions per technique, so a full figure takes several hours; `--quick` uses a short spp schedule, one repetition and a 256-spp reference. The pipeline itself is documented in [`scripts/experiments/README.md`](scripts/experiments/README.md).

| Paper figure | Command | What it runs |
|---|---|---|
| Fig. 1 (*Lighthouse*, thin lens) | `reproduce.sh fig1` | `scenes/7_glints_lighthouse/lighthouse_thinlens_{oneehgsptracermis,path}.xml` |
| Fig. 4 (importance visualization) | `reproduce.sh fig4` | `scenes/0_importance_quads/` + [`scripts/figures/fig4_importance_tvd.py`](scripts/figures/fig4_importance_tvd.py); 3D view with [`scripts/polyscope/view_bvh.py`](scripts/polyscope/view_bvh.py) |
| Fig. 5 (ablation, three rows) | `reproduce.sh fig5` | `ablationv2_asteroids.yaml`, `ablation_asteroids6_smallerspotlight_trueablation.yaml`, `ablation2_veach.yaml` |
| Fig. 6 (specular component) | `reproduce.sh fig6` | `ablation_disney_spec.yaml` |
| Fig. 7 (BVH depth, solid-angle threshold) | `reproduce.sh fig7` | `ablation_asteroids3.yaml`, `ablation_asteroids3_gaussianSA.yaml` |
| Fig. 8 (sparsity analysis) | `reproduce.sh fig8` | `scenes/9_rebuttal_sphere/` (`run_hemispheres.py`, `analyze_hemisphere_runs.py`) |
| Fig. 9 (*AsteroidsFarLight*: PT, BDPT, PPG, ours) | `reproduce.sh fig9` | `ablation_asteroids3_ppg.yaml` |
| Fig. 10 (*AsteroidsCloseLight*: PT, BDPT, PPG, ours) | `reproduce.sh fig10` | `ablation_asteroids4_smallerspotlight_ppg.yaml` |
| Fig. 11 (*Lighthouse*, equal time) | `reproduce.sh fig11` | `scenes/7_glints_lighthouse/lighthouse_{oneehgsptracermis,path}.xml` |
| Fig. 12 (Cornell box) | `reproduce.sh fig12` | `config-v11-cbox-n100.yaml` |

The `.yaml` configs are in [`scripts/experiments/`](scripts/experiments/); [`supplementary/`](scripts/experiments/supplementary/) contains experiments that are not in the paper. See the [notes](#notes-on-reproducibility) below for the differences with the published figures.

#### 3. Use the method in your own scene

Add the integrator and the sampling BVH to a Mitsuba 0.6 scene:

```xml
<integrator type="oneehgspath_2step">
    <integer name="maxDepth" value="3"/>
</integrator>

<geometrybvh type="samplingbvh">
    <integer name="maxDepth" value="16"/>              <!-- maximum BVH depth (Sec. 6.4) -->
    <string name="samplingMode" value="Primitive"/>    <!-- leaf sampling (Sec. 5) -->
    <float name="gaussianSAThreshold" value="0.0314"/> <!-- Sec. 6.4 -->
</geometrybvh>
```

and render it with `mitsuba path/to/scene.xml` inside the container. All `<geometrybvh>` options are optional; the table in [`scripts/experiments/README.md`](scripts/experiments/README.md#parameters-of-the-scenes) lists them. The scenes in `scenes/` write their parameters as `$name:default`, which can be overridden on the command line with `-Dname=value`.

---

### Code structure

This repository is [Mitsuba 0.6](https://github.com/mitsuba-renderer/mitsuba) (commit `10af06f`) plus the following additions and changes.

#### Hierarchical geometry sampling (HGS)

| Path | Contents |
|---|---|
| [`include/mitsuba/render/hgs/bvh.h`](include/mitsuba/render/hgs/bvh.h) | `SamplingBVH` class, node statistics (Table 1), sigma points of the unscented transform |
| [`include/mitsuba/render/hgs/vmf_lut.h`](include/mitsuba/render/hgs/vmf_lut.h), `vmf_poly_unrolled_deg4.h` | VMF cosine integral of the diffuse term (Eq. 14), a degree-4 polynomial fitted by [`scripts/LUT/`](scripts/LUT/) |
| [`include/mitsuba/render/hgs/spherical_aabb.h`](include/mitsuba/render/hgs/spherical_aabb.h) | Solid-angle sampling of AABBs (spherical rectangles) |
| [`src/librender/hgs/bvh_build.cpp`](src/librender/hgs/bvh_build.cpp) | Binned-SAH construction, per-leaf alias tables, bottom-up statistics |
| [`src/librender/hgs/bvh_importance.cpp`](src/librender/hgs/bvh_importance.cpp) | Node importance weights (Eqs. 12-21) and Monte Carlo references (Sec. 6.1) |
| [`src/librender/hgs/bvh_sample.cpp`](src/librender/hgs/bvh_sample.cpp) | Stochastic traversal (Eqs. 8-9), leaf sampling, densities, three-vertex contributions |
| [`src/librender/hgs/bvh_debug.cpp`](src/librender/hgs/bvh_debug.cpp) | Per-node CSV dump of Fig. 4 |
| [`src/hgs/samplingbvh.cpp`](src/hgs/samplingbvh.cpp) | Plugin registration (`<geometrybvh type="samplingbvh">`) |
| [`src/tests/test_hgs.cpp`](src/tests/test_hgs.cpp) | Tests: sampling and PDF evaluation agree for every leaf mode, and the densities are normalized (`docker compose run --rm mitsuba mtsutil test_hgs`) |

#### Plugins

| Plugin | Description | Path |
|---|---|---|
| `oneehgspath_2step` | Path tracer with one-more-vertex NEE: at every vertex, MIS between the BVH-sampled intermediate vertex and BSDF sampling + NEE (Sec. 5). **"Ours" in all path-tracing results.** | [`src/integrators/oneehgspath/`](src/integrators/oneehgspath/) |
| `oneehgsptracermis` | One-more-vertex NEE with x_s on the sensor, combined with camera paths through MIS (*Lighthouse*, Figs. 1 and 11) | [`src/integrators/oneehgsptracermis/`](src/integrators/oneehgsptracermis/) |
| `oneehgsptracer` | Only the camera-side technique of the above (direct illumination of the visible geometry); used by `scenes/7_glints/` | [`src/integrators/oneehgsptracer/`](src/integrators/oneehgsptracer/) |
| `pathindirect` | Path tracer without direct illumination at the first vertex (baseline of Fig. 12) | [`src/integrators/path/pathindirect.cpp`](src/integrators/path/pathindirect.cpp) |
| `basicprincipled` | Simplified principled BSDF (diffuse + GGX specular, metallic) of the Disney scenes | [`src/bsdfs/basicprincipled.cpp`](src/bsdfs/basicprincipled.cpp) |

The integrators report `Integrator::needsSamplingBVH()`, and the scene only builds the SamplingBVH for them.

#### Changes to Mitsuba 0.6 files

| File | Change |
|---|---|
| `include/mitsuba/render/integrator.h` | `needsSamplingBVH()` |
| `include/mitsuba/render/scene.h`, `src/librender/scene.cpp` | The scene owns a `SamplingBVH` (configured by a `<geometrybvh>` child) and builds it when the integrator needs it |
| `include/mitsuba/render/scenehandler.h`, `src/librender/scenehandler.cpp` | `<geometrybvh>` tag; scene parameters with defaults (`$name:default`) |
| `data/schema/scene.xsd` | `<geometrybvh>` element |
| `include/mitsuba/render/bsdf.h`, `src/librender/bsdf.cpp` | `getMetallic()`, `getSpecularF0()`, `getRawReflectance()` for the node statistics |
| `include/mitsuba/core/matrix.h`, `include/mitsuba/core/fwd.h` | `Matrix3x3d` (double-precision accumulation of the node Gaussians) |
| `src/libcore/vmf.cpp` | VMF sampling in double precision (Monte Carlo references) |
| `src/shapes/obj.cpp` | Process-wide cache of parsed OBJ files (scenes instancing one OBJ thousands of times) |
| `src/shapes/ply.cpp` | Loading messages at debug level |
| `src/librender/renderjob.cpp` | "Render time" is the wall-clock time of `render()` (used by the equal-time comparisons; it excludes scene loading and the BVH construction) |
| `src/films/hdrfilm.cpp` | No Mitsuba banner by default |
| `SConstruct`, `src/*/SConscript` | Build of the new plugins and sources |

#### Scripts and data

| Path | Contents |
|---|---|
| `scenes/` | Scenes of all experiments; generated geometry is created by `scripts/prepare_scenes.sh` |
| `scripts/experiments/` | Experiment runner and analysis (metrics, plots, LaTeX) |
| `scripts/reproduce.sh`, `scripts/prepare_scenes.sh`, `scripts/build_mitsuba.sh` | Entry points (run inside the container) |
| `scripts/figures/` | Figure-specific scripts (Fig. 4 TVD, *Lighthouse* slicing) |
| `scripts/LUT/` | Tabulation and polynomial fit of the VMF cosine integral |
| `scripts/polyscope/` | Visualization of the BVH debug CSV |
| `Dockerfile`, `docker-compose.yml`, `patches/ppg.patch` | Build and run environment |

---

### Notes on reproducibility

- **Random geometry.** The quads of the *Veach* scene (`quads-k5.xml`) are generated by `scripts/prepare_scenes.sh` with a fixed seed; the placement used for the paper was not recorded, so the scene is statistically equivalent but not identical.
- **Lighthouse assets.** The paper renders of Figs. 1 and 11 used a snowflake particle system and a mesh (`Plane.003`) exported from Blender that are not distributed. The scenes use 50k snowflakes generated with `create_scene.py` instead (`-Dsnowflakes=...` selects another file), so the images differ from the paper.
- **Corrections after publication.** The code in this repository fixes issues found while preparing the release (see the commit history for details):
  - `oneehgsptracermis` (Figs. 1 and 11): the camera paths stopped at the first vertex, so the rendered images lacked all indirect illumination, and the MIS weights of the camera technique did not sum to one (about -1.6% in a test scene). Both are fixed.
  - `samplingMode=SphericalAABB_anyNode` (two variants of Fig. 6) sampled leaves inconsistently with the density used for MIS (about +0.5% in the Disney scene); fixed.
  - `oneehgspath_2step` counted the BSDF-sampled hits of area lights twice after the one-more-vertex estimate (relative change of ~4e-5 in the Cornell box of Fig. 12); fixed. Scenes with spot lights are not affected.
- **Values that differ from the text of the paper.** All configs use `gaussianSAThreshold = 0.0314` (pi/100), and the test is trace(Sigma) < threshold * distance^2; the sigma points are built from the Cholesky factor of the covariance; the specular albedo of the implementation does not include the factor 1/4 of Eq. (17). The scenes of Fig. 8 render 4 spp unless `-Dspp` is given.

---

### Citation

```bibtex
@article{onemorevertex26,
  journal = {Computer Graphics Forum},
  title = {{One-more-vertex Next-Event Estimation with Hierarchical Geometry Sampling}},
  author = {Garcia-Pueyo, Jorge and Monzon, Nestor and Jarabo, Adrian and Muñoz, Adolfo},
  year = {2026},
  publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
  ISSN = {1467-8659},
  DOI = {10.1111/cgf.70539}
}
```
