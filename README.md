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

*Requirements*: [Docker](https://docs.docker.com/get-docker/) with Docker Compose. No other dependencies are needed on the host; everything is encapsulated in the container.

#### 1. Docker build and compilation

Build the image and start the container:

```bash
docker compose build
docker compose up -d
```

Enter the container:

```bash
docker exec -it onemorevertexneehierarchicalgeometrysampling-mitsuba-1 bash
```

Inside the container, compile Mitsuba:

```bash
cp build/config-linux-gcc.py config.py
python2.7 $(which scons)
source setpath.sh
```

<details>
<summary>Building the PPG baseline (optional, needed for paper comparisons)</summary>

The practical path guiding (PPG) baseline requires a separate Mitsuba build inside the container:

```bash
ln -s /home/mitsuba/data /home/mitsuba/ppg/mitsuba/data
cd ppg/mitsuba
cp build/config-linux-gcc.py config.py
python2.7 $(which scons)
```

</details>

#### 2. Reproduce paper results

From **outside** the container, run the full comparison pipeline for any experiment config:

```bash
bash scripts/experiments/comparison.sh scripts/experiments/ablation_asteroids4_smallerspotlight_ppg.yaml
```

This will render all technique variants, compute metrics (MSE, FLIP), and produce comparison figures and LaTeX assets under the output directory specified in the YAML. See [`scripts/experiments/README.md`](scripts/experiments/README.md) for the full pipeline documentation and config format.

#### 3. Create your own scene and run it

Inside the container (after compiling and sourcing `setpath.sh`), render any scene XML directly:

```bash
mitsuba scenes/experiments/1_wildcard_spotlight/wildcard_spotlight_quads.xml
```

To use the one-more-vertex integrator in your own Mitsuba scene, add the following integrator block to your scene XML:

```xml
<integrator type="oneehgspath">
    <integer name="maxDepth" value="3"/>
</integrator>
```

---

### Code structure

This repository is based on [Mitsuba 0.6](https://github.com/mitsuba-renderer/mitsuba). All source code outside the paths listed below is unmodified Mitsuba 0.6.

#### HGS core

| Group | Description | Path |
|-------|-------------|------|
| Headers | `SamplingBVH` class declaration; `VMFLUT` (VMF intensity via degree-4 unrolled polynomial); spherical AABB | `include/mitsuba/render/hgs/` |
| Implementation | BVH construction, per-node statistics fitting, importance weight computation, stochastic traversal sampling | `src/librender/hgs/` |
| Plugin | Mitsuba plugin registration wrapper for `SamplingBVH` | `src/hgs/` |

#### Integrators

| Description | Path |
|-------------|------|
| One-more-vertex path tracer | `src/integrators/oneehgspath/` |
| One-more-vertex particle tracer | `src/integrators/oneehgsptracer/` |
| One-more-vertex particle tracer with MIS | `src/integrators/oneehgsptracermis/` |

#### Scenes, scripts, and infrastructure

| Component | Description | Path |
|-----------|-------------|------|
| Scenes | Mitsuba XML scene files for all paper experiments (Asteroids, Disney, Veach, Cornell box, Glints, …) | `scenes/` |
| Experiment pipeline | Shell and Python scripts to render all technique variants, compute MSE/FLIP metrics, and produce comparison figures and LaTeX assets | `scripts/experiments/` |
| LUT tools | Python scripts to generate the VMF intensity LUT and fit the degree-4 polynomial approximation | `scripts/LUT/` |
| PPG baseline | Practical path guiding as a self-contained Mitsuba build used for paper comparisons | `ppg/` |
| Docker | Containerized build and run environment; no host dependencies beyond Docker and Docker Compose | `Dockerfile`, `docker-compose.yml` |

---

### Citation

```bibtex
Coming soon.
```
