# Plan: Glints Integrator (`glinttracer`)

## Goal

Render camera-visible glints by connecting the **camera origin x_o** to an **emitter x_e**
through an **intermediate surface vertex x_n** found via hierarchical BVH traversal.
This differs from the existing `additionalvertex`/`doublestep` integrators (which treat
x_s as a surface vertex with a BSDF) because x_o is a pinhole camera: no surface, no BSDF.
The contribution to a pixel is determined _after_ sampling x_n, by checking whether the ray
x_o → x_n maps to a valid pixel via `sampleAttenuatedSensorDirect`.

## Architecture

Sensor model assumed: **pinhole perspective camera** for all experiments.

```
x_o  ----(BVH sample)---->  x_n  ----(shadow ray)---->  x_e
camera origin              geometry vertex           emitter position
  |                              |
  +--sampleAttenuatedSensorDirect--> pixel UV + W_e
```

## Component 1: `evalGlintContribution`

**Files:** `include/mitsuba/render/bvh/bvh.h` (declaration) +
           `src/librender/bvh/bvh_sample.cpp` (implementation)

Analogue of `evalLength2Contribution` but without the x_s BSDF term.

```cpp
MTS_EXPORT_RENDER Spectrum evalGlintContribution(
    const Scene *scene,
    const Intersection &its_xn,       // sampled geometry vertex
    const PositionSamplingRecord &pRec_xe,  // emitter sample
    const Vector &d_n_to_o,           // direction x_n → camera (= dRec.d from sampleAttenuatedSensorDirect)
    bool checkVisibility = true        // shadow ray x_n → x_e only
);
```

Returns: `f_xn(wi=d_n_to_o, wo=d_n_to_e) * G_ne * Le`
- **does NOT** divide by `pRec_xe.pdf` (caller does, consistent with `evalLength2Contribution`)
- **does NOT** check visibility x_n → x_o (handled by `sampleAttenuatedSensorDirect`)

## Component 2: `SamplingBVH::sampleGeometryCamera`

**Files:** `include/mitsuba/render/bvh/bvh.h` (declaration in class body) +
           `src/librender/bvh/bvh_sample.cpp` (implementation)

```cpp
bool SamplingBVH::sampleGeometryCamera(
    const Scene *scene,
    Sampler *sampler,
    const Point &x_o,                      // camera origin (replaces its_xs.p)
    const PositionSamplingRecord &pRec_xe,  // emitter vertex
    Intersection &its_xn,                  // output: sampled intermediate vertex
    Float &pdf                             // output: solid angle PDF
);
```

Structural copy of `sampleGeometry` with two changes:
1. **Traversal importance**: replaced by a **stub** (`nodeInfo.surfaceArea`-weighted) since
   `computeNodeImportance_Camera` is deferred. This gives area-proportional sampling,
   which is a valid first-pass (not guided by camera direction, but still non-uniform).
2. **Leaf sampling point**: `x_o` passed instead of `its_xs.p` to all leaf functions.

Lightcuts and `m_probmult`/`m_defensive_pdf` are preserved unchanged.

**TODO (next step):** replace stub with `computeNodeImportance_Camera(scene, bvh, idx, x_o, pRec_xe)`.

## Component 3: `glinttracer` integrator

**Files:** `src/integrators/glinttracer/glinttracer.cpp` (new, self-contained)

Structure mirrors `AdjointParticleTracer` (`ptracer.cpp`) + `ptracer_proc.cpp`.
Classes are renamed with `Glint` prefix to avoid duplicate RTTI when both plugins are loaded.

### `GlintWorkResult` (mirrors `CaptureParticleWorkResult`)
Thin `ImageBlock` subclass that also stores the `RangeWorkUnit` pointer.

### `GlintWorker` (custom `WorkProcessor`, does NOT extend `ParticleTracer`)
Per-particle loop:
```
1. Compute x_o = sensor->getWorldTransform(t=0)(Point(0,0,0))
2. sampleEmitterPosition(pRec_xe, sample)  →  xe, pRec_xe.pdf
3. bvh->sampleGeometryCamera(scene, sampler, x_o, pRec_xe, its_xn, pdf_xn)
4. sampleAttenuatedSensorDirect(dRec, its_xn, ...)  →  W_e, dRec.uv, dRec.d
5. its_xn.wi = its_xn.toLocal(dRec.d)  // direction toward camera, in local frame
6. evalGlintContribution(scene, its_xn, pRec_xe, dRec.d, checkVisibility=true)
7. value = W_e * contrib / (pdf_xn * pRec_xe.pdf)
8. workResult->put(dRec.uv, value)
```

### `GlintProcess` (mirrors `CaptureParticleProcess`)
Subclasses `ParticleProcess`. Key overrides:
- `createWorkProcessor()` → returns `GlintWorker`
- `processResult()` → accumulates `GlintWorkResult` into `m_accum` (copy from ptracer)
- `bindResource("sensor")` → sets up `m_film` and `m_accum` (copy from ptracer)
- `develop()` → normalizes by `W*H / receivedResultCount` (copy from ptracer)

### `GlintTracer` (mirrors `AdjointParticleTracer`)
Parameters: `maxDepth` (unused for now), `rrDepth` (unused for now), `granularity`.
`preprocess()`: computes `m_sampleCount = spp * W * H`.
`render()`: creates `GlintProcess`, binds resources, schedules, waits.

## Component 4: Build system

**File:** `src/integrators/SConscript`

Add:
```python
plugins += env.SharedLibrary('glinttracer', ['glinttracer/glinttracer.cpp'])
```

## Component 5: BVH construction

**No changes.** The existing BVH aggregates (spatial Gaussians, VMF normals, material
statistics) are geometry-only and do not depend on the camera or lighting configuration.

## Deferred (not in this implementation)

- `computeNodeImportance_Camera` — importance function guided by camera frustum + sensor
  sensitivity (FoV culling + VMF LUT for x_n → x_e side using camera direction as d_n_to_o).
- `pdfGeometryCamera` — PDF of the BVH traversal for MIS.
- MIS between BVH sampling and BSDF sampling at x_n.

## Implementation order

1. `evalGlintContribution` — **DONE** (`src/librender/bvh/bvh_sample.cpp` + declaration in `bvh.h`)
2. `sampleGeometryCamera` — **DONE** (`src/librender/bvh/bvh_sample.cpp` + declaration in `bvh.h`)
3. `glinttracer.cpp` — **DONE** (`src/integrators/glinttracer/glinttracer.cpp`)
4. `SConscript` — **DONE** (entry added to `src/integrators/SConscript`)

## Next steps (after testing)

5. `computeNodeImportance_Camera` — importance function guided by camera FoV and sensor
   sensitivity, replacing the area-only stub in `sampleGeometryCamera`.
6. FoV culling — skip nodes whose AABB projects entirely outside the camera frustum.
7. MIS + `pdfGeometryCamera` — once a second sampling strategy (e.g. BSDF at x_n) is added.
