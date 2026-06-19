# Implementation Plan: Additional Vertex Integrator with Geometry-Aware Sampling

## Overview

Implement an integrator that intelligently samples geometry directly using BVH (Bounding Volume Hierarchy), avoiding empty-space ray tracing in sparse geometry scenes. The BVH stores aggregate information (albedo, area, normal distribution, center of mass) that, when evaluated by probabilistically traversing the tree, guides paths towards high-contribution geometry.

**Key Innovation:** The importance function is **context-dependent**, considering:
- **SurfaceSample (xs):** Current shading point (position, normal, BSDF)
- **EmitterSample (xe):** Sampled light source (position, normal, emission)
- **Node aggregates:** Pre-computed albedo, area, normal distribution, center of mass

This allows BVH traversal to adapt to the specific path configuration, weighting nodes by actual contribution potential rather than static albedo×area values.

**File Organization:**
```
include/mitsuba/render/bvh/
└── bvh.h                    # All structures: BVHNodeInfo, BVHNode, SamplingBVH

src/librender/bvh/
├── bvh_build.cpp            # BVH construction + aggregate propagation
├── bvh_importance.cpp       # Context-dependent importance evaluation
└── bvh_sample.cpp           # (planned) Geometry sampling + reverse-PDF for MIS

src/integrators/additionalvertex/
└── additionalvertex.cpp     # (to modify) Integrate sampling strategy
```

---

## Phase 1: BVH Infrastructure (COMPLETE)

**Status:** Implemented. Tests: `src/tests/test_bvh_phase1_bvhconstruction.cpp`, `src/tests/test_bvh_phase1_aggregation.cpp`

This phase covers two activities: BVH tree construction and aggregate computation/propagation.

### 1.1 BVH Tree Construction

Triangle-level BVH using object-median split:
- Collects all triangles from `Scene::getMeshes()` with AABB + centroid caching
- Splits via `std::nth_element` on longest centroid-bounds axis
- Depth-first node storage (left child = parent + 1)
- Configurable leaf size with degenerate centroid fallback-to-leaf

**Data Structures** (`bvh.h`):

```cpp
struct BVHPrimitive {
    uint32_t meshIndex;
    uint32_t triangleIndex;
    AABB bounds;
    Point centroid;
    const BSDF *bsdf;
};

struct BVHNode {
    AABB bounds;
    union {
        uint32_t primitivesOffset;   // Leaf: offset into primitive array
        uint32_t secondChildOffset;  // Interior: offset to second child
    };
    uint32_t nSubtreePrimitives;     // Total primitives in this subtree
    uint16_t nLeafPrimitives;        // 0 = interior, >0 = leaf
    uint8_t splitAxis;
    uint8_t pad;
    
    bool isLeaf() const { return nLeafPrimitives > 0; }
};

class SamplingBVH {
public:
    void build(const Scene *scene);
    void buildAggregates(const Scene *scene);
    // Accessors for nodes, primitives, nodeInfos...
};
```

**Build Algorithm:**
1. Compute bounds of primitives in range
2. If count ≤ maxLeafSize → create leaf
3. Compute centroid bounds, choose split axis (longest dimension)
4. If degenerate (all centroids same) → create leaf
5. Partition via `std::nth_element` at median
6. Recursively build children (left at index+1, right stored in secondChildOffset)

### 1.2 Aggregate Computation & Propagation

Bottom-up computation of node aggregates:

**Per-Node Aggregates** (`BVHNodeInfo`):
- `surfaceArea`: Total triangle area
- `diffuseAlbedo`: Area-weighted average albedo (from BSDF sampling)
- `normalDistribution.m1`: E[n] (area-weighted first moment)
- `normalDistribution.m2`: E[n nᵀ] (area-weighted second moment matrix)
- `centerOfMass`: Area-weighted centroid

**Leaf Aggregates:** Computed from actual triangles via `computeTriangleInfo()`:
- Area from cross product
- Normal from triangle vertices
- Albedo from BSDF sampling

**Interior Aggregates:** Combined from children via `BVHNodeInfo::accumulate()`:
- Area-weighted averaging for moments
- Variance via parallel-axis theorem

### 1.3 Importance Function

Context-dependent evaluation in `bvh_importance.cpp`:

```cpp
Float computeNodeImportance(nodeInfo, nodeBounds, xs, xe) {
    Float baseImportance = nodeInfo.diffuseAlbedo * nodeInfo.surfaceArea;
    if (baseImportance == 0) return 0;
    
    // Geometric term using center of mass as representative point
    Point nodeCenter = nodeInfo.centerOfMass;  // fallback to bounds.getCenter()
    Vector d_xs = nodeCenter - xs.p;
    Vector d_xe = xe.p - nodeCenter;
    Float geometricTerm = 1 / (dist_xs² * dist_xe² + ε);
    
    // BSDF alignment
    Float cosTheta = dot(nodeInfo.normalDistribution.meanDirection(), normalize(d_xs));
    Float bsdfAlignment = max(0, cosTheta);
    
    // Variance penalty
    Float variance = tr(m2) - ||m1||²;
    Float variancePenalty = 1 / (1 + variance);
    
    return baseImportance * geometricTerm * (bsdfAlignment + 0.1) * variancePenalty;
}
```

### 1.4 Scene Integration

- `Scene` owns `ref<SamplingBVH> m_geometryBVH` with `getSamplingBVH()` accessor
- Constructors allocate SamplingBVH with optional `maxLeafSize` property
- Build in `Scene::initialize()` by passing a reference to the `Scene` class itself.

---

## Phase 2: Geometry Sampling (COMPLETE)

Implement importance-driven geometry sampling with PDF computation for MIS.

### 2.1 Sampling Algorithm (COMPLETE)

```
Input: BVH root, shading point xs, emitter sample xe, random samples (s1, s2)
Output: position p, normal n, pdf_geom

// Importance-driven tree traversal
nodeIdx = 0
pdf_traversal = 1.0

while not nodes[nodeIdx].isLeaf():
    leftIdx = nodeIdx + 1
    rightIdx = nodes[nodeIdx].secondChildOffset
    
    left_importance = computeNodeImportance(nodeInfos[leftIdx], nodes[leftIdx].bounds, xs, xe)
    right_importance = computeNodeImportance(nodeInfos[rightIdx], nodes[rightIdx].bounds, xs, xe)
    
    total = left_importance + right_importance
    left_prob = (total > 0) ? left_importance / total : 0.5
    
    if s1 < left_prob:
        nodeIdx = leftIdx
        pdf_traversal *= left_prob
        s1 = s1 / left_prob  // reuse random number
    else:
        nodeIdx = rightIdx
        pdf_traversal *= (1 - left_prob)
        s1 = (s1 - left_prob) / (1 - left_prob)

// Sample primitive in leaf (uniform)
primIdx = floor(s2 * nPrimitives)
// Sample point on triangle (uniform barycentric)

pdf_geom = pdf_traversal * (1/nPrimitives) * (1/triangle_area)
return (p, n, pdf_geom)
```

**Complexity:** O(log N) per sample.
Implemented in `src/librender/bvh/bvh_sample.cpp`. Supporting both `Primitive` (uniform) and `SphericalAABB` leaf sampling modes.

### 2.2 Reverse PDF Query (COMPLETE)

For MIS, computed via `pdfGeometry(xs, xe, meshIndex, triangleIndex, p_hit)`.

**Design:**
- **Triangle Lookup:** Uses a precomputed `m_triangleToPrim` map ($O(1)$) to find the target's position in the global primitive array.
- **Traversal:** Navigates from root to leaf by tracking the `nodeStart` offset and comparing the target index against `leftNode.nSubtreePrimitives`.
- **Consistency:** Re-evaluates `computeNodeImportance` at every step to ensure the reverse PDF exactly matches the forward branching probabilities.

**Complexity:** O(log N).

### 2.3. Integration of spherical axis-aligned bounding boxes
When a leaf node is reached during importance-driven BVH traversal we can either (A) sample a primitive inside the leaf directly (triangle sampling), or (B) perform a spherical-axis-aligned-bounding-box (spherical-AABB) sampling strategy that samples directions/points on the six faces of the leaf's AABB as seen from the shading point. Option (B) trades direct primitive selection for an inexpensive direction-based sampling that (1) naturally accounts for solid-angle visibility from the shading point and (2) allows sampling geometry that lies behind occluders within the AABB via explicit ray cast.

The spherical-AABB method proceeds in three conceptual steps:
1. For each of the six faces of the AABB, compute the face's projected solid angle as seen from the shading point `xs.p`.
2. Sample one face with probability proportional to its projected solid angle.
3. Sample a point uniformly on the chosen rectangular face (area-parametrization), form the direction toward that sample from `xs.p`, trace a ray along that direction, and accept the first intersection inside the AABB (if any). The resulting intersection point is treated as the geometry sample.

Compared to uniform triangle sampling, spherical-AABB focuses effort on directions that occupy large solid angles from the shading point — a beneficial bias when geometry is small but subtends a noticeable solid angle (e.g., small high-albedo objects in a large room).

#### 2.3.1. Face solid-angle computation and sampling (concrete algorithm)
1. Build the 8 corner vertices of the leaf AABB in world coordinates.
2. For each of the 6 faces, obtain its 4 corner vertices (ordered to form a rectangle). Convert each corner to the unit direction from `xs.p`: u_i = normalize(v_i - xs.p).
3. Compute the spherical quad solid angle for the face. A robust and simple approach is to split the quad into two triangles (u0,u1,u2) and (u0,u2,u3) and compute each triangle's solid angle using the vector triple-product atan2 formula:

    omega_triangle(u,v,w) = 2 * atan2(|u·(v×w)|, 1 + u·v + v·w + w·u)

    where u,v,w are unit vectors to the triangle vertices. Sum the two triangle omegas to get face omega.

4. Clamp and floor: if omega ≤ 0 due numerical issues, set a small epsilon; if `xs.p` is inside the AABB, treat total solid angle as 4π and fall back to area-based face weights.
5. Normalize omegas across the six faces to get face selection probabilities P_face.
6. Draw a 1D sample `s` to select a face according to P_face; then draw 2D samples `(u,v)` to sample a point uniformly on the chosen rectangular face (area parameterization).

Notes on numerical robustness and edge cases:
- If `xs.p` lies extremely close to a face plane, the unit directions may be nearly collinear; use double precision and clamp dot products to [-1,1].
- If `xs.p` is inside the AABB, projected solid angles are not meaningful; instead sample a face uniformly weighted by its area or sample primitives directly.

#### 2.3.2. Ray casting, acceptance and computing the pdf (practical recipe)
1. After sampling a point `p_rect` on the chosen face, form the ray direction `d = normalize(p_rect - xs.p)` and cast a ray from `xs.p` towards `d`.
2. If the ray intersects geometry at point `p_hit` with triangle `T` and the intersection distance falls within the AABB extent along that ray, accept `p_hit` as the sampled geometry position. If the ray misses geometry inside the box, report a missed sample (return pdf = 0) and treat the sample path as a wasted geometry sample (counts toward traversal cost).
3. To keep estimates unbiased we must compute the forward pdf for the accepted geometry sample. The forward pdf decomposes as:

    pdf_geom = pdf_traversal * P_face * pdf_on_face_area * J_area->surface

    where:
    - `pdf_traversal` is the product of interior-child-selection probabilities produced during the BVH traversal (same as existing traversal PDF).
    - `P_face` is the normalized projected-solid-angle probability for the chosen face.
    - `pdf_on_face_area` = 1 / area(face) for uniform sampling on the rectangle surface.
    - `J_area->surface` is the Jacobian converting the density on the sampled rectangle area `dA_rect` to density on the surface area of the hit triangle. If the ray hits triangle `T` at `p_hit`, the mapping from a differential rectangle area `dA_rect` around `p_rect` to the differential surface area on `T` is given by:

         dA_T = (cos_theta_hit / |p_rect - xs.p|^2) * dω  and dA_rect = (cos_theta_rect / |p_rect - xs.p|^2) * dω

    Eliminating `dω` (solid angle), the Jacobian for converting `pdf_on_face_area` into `pdf_on_surface_area` is:

         J_area->surface = (cos_theta_rect / cos_theta_hit)

    where `cos_theta_rect` = abs(dot(n_rect, normalize(xs.p - p_rect))) (normal of the rectangle face) and `cos_theta_hit` = abs(dot(n_T, -d)) (triangle normal vs incoming direction). The distance terms cancel because both use the same direction and range.

4. In practice `cos_theta_hit` can be near zero for grazing intersections; clamp `cos_theta_hit` to a small epsilon to avoid division by zero. When the ray hits an edge or a different surface not intended by the rectangle projection, compute the Jacobian based on the actual hit geometry.

5. For MIS and correctness, the reverse PDF (pdfGeometry called for a given triangle ID) must be able to recompute the same value by (a) recomputing traversal probabilities down the BVH to the leaf and (b) summing/integrating all possible rectangle samples that would produce a ray hitting this triangle at `p_hit` — this is complex in closed form. As a practical alternative we will implement an exact reverse PDF for the triangle-sampling mode and an approximate, but consistent, reverse PDF for the spherical-AABB mode by deterministically simulating the same face selection + area sampling path for the given `p_hit` (i.e., compute which face sample `p_rect` would generate direction hitting `p_hit` and evaluate the corresponding `P_face` and `1/area(face)` factors).

#### 2.3.3. Integration into leaf sampling and BVH APIs
1. Extend the `sampleGeometry` API to accept a sampling-mode flag (e.g., `Mode::Primitive` or `Mode::SphericalAABB`) and return both the sampled intersection and the forward PDF.
2. In `bvh_sample.cpp` implement a two-path leaf sampler:
    - Primitive leaf sampling (existing): choose a primitive uniformly or area-weighted and sample a point on triangle; pdf computed as traversal_pdf * (1/nPrimitives) * (1/triangle_area) or traversal_pdf * area-weighted expression if area-weighted.
    - Spherical-AABB leaf sampling (new): follow the face omega sampling → area-sample-rectangle → trace → accept/reject → compute `pdf_geom` as described above. Return lost/missed rays as pdf=0 to the caller.
3. Store in `SamplingBVH` a lightweight face cache (AABB corners + face areas) to avoid recomputing rectangle areas every sample. Provide accessors to compute face omegas given `xs.p` efficiently (vectorized dot/cross computations).
4. Provide a configuration switch at integrator level (`m_geometrySamplingMode`) so `additionalvertex` can choose between the two leaf strategies per-sample (or probabilistically mix them).

#### 2.3.4. Summary of the Spherical AABB Leaf Sampling implementation
The `SamplingBVH` class now supports two strategies for sampling geometry in leaf nodes: primitive mode where a single point in a primitive is chosen and spherical aabb mode where a ray is casted aginst the aabb of the geometry.

To use it:
```cpp
auto mode = SamplingBVHSamplingMode::SphericalAABB; // or SamplingBVHSamplingMode::Primitive
SamplingBVH bvh(4, mode);
bvh.build(scene);
bvh.buildAggregates(scene);

// Rest of code identical - mode is transparent to caller
bvh.sampleGeometry(scene, xs, xe, sample1, sample2, 
                  position, normal, pdf);
```

#### 2.3.4. Testing plan for spherical-AABB integration
Unit & integration tests to validate correctness, stability and efficiency:
- Forward/Reverse PDF consistency tests
  - For a grid of shading points `xs.p` and deterministic RNG seeds, sample N geometry points via spherical-AABB. For each accepted sample compute the forward pdf returned by `sampleGeometry` and compute a reconstructed pdf via `pdfGeometry(xs, xe, meshIndex, triIndex, p_hit)` using the deterministic reconstruction (face intersection). Expect relative error < small tolerance for accepted samples.
- Equivalence with triangle-sampling in the limit
  - In tightly filled leaves (many primitives filling the AABB), the spherical-AABB sampling should approximate triangle sampling. Build a test scene where all triangles closely tile an AABB face and compare histograms of sampled hit positions between the two methods.
- Numeric edge-case tests
  - `xs.p` inside the AABB: ensure fallback behavior (area-based face sampling or direct primitive sampling). 
  - Extremely near-field cases: verify clamping prevents NaNs (dot products, division by small cosines).
- (Not needed for now) Acceptance rate and wasted-sample measuring
  - Measure the fraction of spherical-AABB samples that miss (ray hits no geometry inside the AABB). Report acceptance ratio and average cost (ray casts per accepted sample).
- (Not needed for now) Performance microbench
  - Time per sample for spherical-AABB vs primitive sampling on representative scenes (sparse small object in large room, dense cluster) to guide when to enable method.

These tests will live under `src/tests/` as `test_bvh_phase2_sampletraversal_with_sphericalaabbsampling.cpp` and use deterministic RNG seeds to ensure reproducibility.

### 2.4 Implementation Tasks (Phase 2 — concrete)

Phase 2 implements geometry sampling and spherical-AABB integration. Concrete tasks and file targets:

1. Implement sampling primitives and spherical-AABB in `src/librender/bvh/bvh_sample.cpp`:
    - `sampleGeometry(const SamplingBVH &bvh, const Scene *scene, const SurfaceSample &xs, const EmitterSample &xe, Point2 &s1, Point2 &s2, Mode mode, GeometrySample &out) -> bool` — returns whether a valid geometry sample was produced and sets `out.pdf`.
    - `pdfGeometry(const SamplingBVH &bvh, const Scene *scene, const SurfaceSample &xs, const EmitterSample &xe, meshIndex, triangleIndex, const Point &p_hit, Mode mode) -> Float` — reverse-PDF. Implement exact reverse-PDF for `Mode::Primitive` and deterministic reconstruction for `Mode::SphericalAABB`.

2. Add spherical-AABB utilities in `include/mitsuba/render/bvh/spherical_aabb.h` and implement math helpers (face solid-angle, triangle solid-angle, rectangle sampling) in `src/librender/bvh/spherical_aabb.h`.

3. (TODO(jorge): not really needed?) Add BVH caches/accessors in `include/mitsuba/render/bvh/bvh.h` / `bvh.cpp` to expose face corners and face areas for each node (lightweight: 6 faces × 4 corner indices). If not stored, compute on-the-fly with the node AABB (cheap) but provide inline helpers to reduce branching.

4. Integrator changes (in `src/integrators/additionalvertex/additionalvertex.cpp`):
    - Add `m_geometrySamplingMode` and `m_enableSphericalAABB` flags and parsing.
    - Call `sampleGeometry(..., Mode::SphericalAABB)` when sampling geometry; blend with BSDF samples via MIS as described in Phase 3.

5. Tests
    - Add `src/tests/test_bvh_phase2_sample_aabb.cpp` to test spherical-AABB functions.
    - Add `src/tests/test_bvh_phase2_sample.cpp` for complete test of bvh sampling, including both primitive and spherical-AABB sampling.

6. Build system
    - Update `src/librender/SConscript` to compile new bvh sampling files.

Notes on correctness and trade-offs:
- The reverse-PDF for spherical-AABB is reconstructed deterministically from `xs.p` and `p_hit` by computing which rectangle sample `p_rect` would have generated the same ray; implement carefully to match the forward path probabilities (face omega, area sampling, Jacobian).
- If the deterministic reverse reconstruction is ambiguous (e.g., floating point tolerance places `p_rect` slightly outside face), treat it as zero and log diagnostic failures for debugging tests; aim to eliminate ambiguity by tolerant epsilon checks.

---

## Phase 2.5: Stochastic Baseline Sampling

Before moving to the high-performance BVH-based integrator, we will implement a "Brute Force" stochastic baseline. This will serve as the ground truth for our importance-driven sampling logic.

### 2.5.1 Algorithm

For every path vertex $(x_s, x_e)$ in the integrator:
1.  **PDF Construction:** Iterate over all $N$ triangles in the scene.
2.  **Importance Evaluation:** For each triangle $T_i$, compute its importance $I_i = \text{computeNodeImportance}(T_i, x_s, x_e)$.
3.  **Discrete PDF:** Normalize these importance values to build a discrete PDF $P$ over all triangles.
4.  **Sampling:** Sample a triangle $T_k$ from $P$, and then sample a point $p$ uniformly on $T_k$.
5.  **PDF Value:** The total PDF is $P_k \times (1/\text{Area}(T_k))$.

### 2.5.2 Purpose
- **Validation:** Provides a baseline that uses the exact same `computeNodeImportance` logic as the BVH but without the approximation of node aggregates.
- **Debugging:** Helps identify if convergence issues are due to the importance function itself or the BVH traversal/approximations.
- **Speed:** This method will be very slow ($O(N)$ per vertex), but it is essential for verifying Phase 3.

### 2.5.4 Integration Strategies

We consider two primary ways to integrate this geometry sampling into the integrator:

#### Option 1: Length-2 Next Event Estimation (NEE)
This approach connects the current shading point $x_s$ to a sampled emitter $x_e$ through an intermediate geometry point $p$.
- **Mechanism:** $L = L + \text{throughput} \times f_r(x_s, p) \times \frac{G(x_s, p)}{PDF_{geom}} \times f_r(p, x_e) \times G(p, x_e) \times L_e(x_e)$.
- **Role:** It acts as an additional "direct illumination" term that accounts for one-bounce reflections towards light sources.
- **Decision:** **This will be implemented first as a ground truth baseline.**

#### Option 2: Next-Vertex Sampling
This approach uses the sampled geometry point $p$ to continue the path, replacing or complementing the standard BSDF sampling.
- **Mechanism:** The next ray is traced directly towards $p$. MIS is used to combine the area-based PDF of $p$ with the solid-angle PDF of the BSDF.
- **Role:** Guides the entire path towards important geometry, effectively acting as an importance-sampled vertex selection.

---

## Phase 3: Integrator Integration

**Status:** Not started (Starting with Length-2 NEE Baseline)


### 3.1 Modified Li() Structure

```cpp
Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
    while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
        // ... standard direct illumination (NEE) ...
        
        // === Geometry-Aware Sampling ===
        bool useGeometrySampling = false;
        Point geomPosition;
        Normal geomNormal;
        Float geomPdf = 0.0f;
        
        if (m_geometrySamplingFraction > 0 && rRec.nextSample1D() < m_geometrySamplingFraction) {
            SurfaceSample xs{its.p, its.shFrame.n, its.getBSDF()};
            
            // Sample emitter for importance context
            DirectSamplingRecord dRec(its);
            Spectrum Le = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
            if (!Le.isZero()) {
                EmitterSample xe{dRec.p, dRec.n, Le};
                
                useGeometrySampling = m_geometrySampler.sampleGeometry(
                    scene->getSamplingBVH(), scene, xs, xe,
                    rRec.nextSample2D(), geomPosition, geomNormal, geomPdf);
            }
        }
        
        // === Standard BSDF Sampling ===
        BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
        Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
        
        // === Choose/MIS sampling strategies ===
        // ... blend geometry and BSDF samples with MIS weights ...
    }
    return Li;
}
```

### 3.2 Configuration Parameters

```cpp
Float m_geometrySamplingFraction;  // [0,1] probability of geometry sampling per vertex
Bool m_enableGeometricTerm;        // Weight nodes by distance/angles (default: true)
Bool m_enableBSDFAlignment;        // Weight by BSDF lobe alignment (default: true)
```

### 3.3 Implementation Tasks

1. Add includes and members to `additionalvertex.cpp`
2. Parse parameters in constructor
3. Implement sampling strategy blending with MIS
4. Update `toString()` for diagnostics

---

## Phase 4: Validation & Tuning

1. Test on sparse geometry scene (Stanford bunny in large room)
2. Measure convergence vs. standard path tracer
3. Benchmark BVH construction overhead
4. Tune `m_geometrySamplingFraction` parameter
5. Verify PDF consistency: `avg(1/pdf)` should match expected value

---

## Design Decisions

### Why Context-Dependent Importance?

**Traditional approach (context-free):**
```cpp
importance = albedo × area  // Static, precomputed
```

**Our approach (context-dependent):**
```cpp
importance = albedo × area × geometric_term(xs, node, xe) × bsdf_alignment(xs, node)
```

**Benefits:**
1. **Geometric configuration awareness:** Far nodes get lower importance (1/r²), unfavorable angles down-weighted
2. **BSDF-aware sampling:** Aligns with incident/outgoing directions, incorporates glossy lobe orientations
3. **Emitter-aware:** Considers emitter position xe, can weight by intensity
4. **Variance reduction:** Samples geometry that contributes to the specific path, avoids wasted samples

**Trade-off:** Cannot pre-compute (depends on xs, xe), but still O(log N) per sample.

### Normal / Variance Representation

Per-node storage:
- `m1 = E[n]` (area-weighted first moment)
- `m2 = E[n nᵀ]` (area-weighted second moment matrix)
- Derived: `Var = tr(m2) - ||m1||²` (scalar dispersion proxy)

This is SGGX-ready for future anisotropic normal statistics.

**Rough normals handling:**
- Store variance to indicate surface roughness
- Option: Weight sampling by `1 / (1 + variance)` to prefer flatter surfaces
- Or: Track distribution and weight by alignment to expected normal

---

## Known Limitations & Future Extensions

1. **Diffuse-only:** No specular/glossy contribution to geometry selection. Future: blend with specular color.
2. **No occlusion:** Can sample invisible surfaces. Future: hierarchical occlusion tests.
3. **Uniform leaf sampling:** All primitives in leaf sampled uniformly. Future: area-weighted selection.
4. **Static aggregates:** For animated scenes, must recompute per frame.
5. **SGGX simplified:** Currently isotropic. Future: full anisotropic NDF.
6. **Build cost:** Object-median is fast; SAH would be higher quality but slower.

---

## Testing Strategy

### Importance Function Validation
- **Geometric term:** Verify importance decreases with distance² for equal albedo/area
- **BSDF alignment:** Aligned normals (cosθ≈1) >> perpendicular normals (cosθ≈0)
- **Context sensitivity:** Same node, different xs → different importance

### Test Scenes
- **Cornell box:** Basic convergence test with 1-2 objects
- **Sparse scene:** Large room with small high-albedo object (geometry sampling should dominate)
- **Heterogeneous:** Objects with varying albedos (0.1 to 0.9)

### Metrics
- Sample-to-convergence ratio
- BVH aggregation time
- Variance reduction (MSE vs. standard path tracer)
- PDF consistency: forward and reverse must match

---

## Appendix: Deprecated Design Notes

Earlier versions contained "`ShapeBVH` with SAH + per-shape aggregates" design. Current implementation uses **triangle-level `SamplingBVH`** with object-median split. SAH sections removed; can be reintroduced if needed for future optimization.
