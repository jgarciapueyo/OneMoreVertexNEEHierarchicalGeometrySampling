# Implementation Plan: Additional Vertex Integrator with Geometry-Aware Sampling

## Overview
Implement an integrator that intelligently samples geometry directly using KDTree aggregates, avoiding empty-space ray tracing in sparse geometry scenes. Geometry-aware aggregates (albedo, area, normal distribution) are built post-construction on the ShapeKDTree and used to probabilistically traverse the tree, sampling high-contribution geometry.

---

## 1. Data Structures

### 1.1 KDNodeInfo (Per-Node Aggregates)

```cpp
// File: mitsuba/include/mitsuba/render/kdnodeinfo.h

struct KDNodeInfo {
    /// Diffuse albedo (Spectrum converted to average reflectance)
    Float diffuseAlbedo;
    
    /// Total surface area (for area-weighted sampling)
    Float surfaceArea;
    
    /// SGGX normal distribution function parameters
    /// Stores mean normal + covariance/shape matrix for anisotropic NDF
    struct SGGX {
        Normal meanNormal;      // Principal normal direction
        Float  variance;        // Overall spread (isotropic approximation for now)
        // Extended: could add covariance matrix for full anisotropy
    } normalDistribution;
    
    /// Validity flag (true if node contains actual geometry)
    bool valid;
    
    KDNodeInfo()
        : diffuseAlbedo(0.5f), surfaceArea(0.0f), valid(false) {
        normalDistribution.meanNormal = Normal(0, 1, 0);
        normalDistribution.variance = 1.0f;
    }
};
```

### 1.2 Extended ShapeKDTree Structure

Add to `ShapeKDTree` class header:

```cpp
// mitsuba/include/mitsuba/render/skdtree.h (additions)

private:
    /// Parallel array: m_nodeInfo[node_index] = KDNodeInfo
    /// Index computed as: node_index = (node_ptr - m_nodes)
    std::vector<KDNodeInfo> m_nodeInfo;
    
    /// Pre-computed total area across all shapes
    Float m_totalArea;
    
public:
    /// Build geometric aggregates (call immediately after build())
    void buildGeometricAggregates();
    
    /// Retrieve aggregate info for a node
    inline const KDNodeInfo& getNodeInfo(const KDNode *node) const {
        size_t idx = node - m_nodes;
        return m_nodeInfo[idx];
    }
    
    /// Access node pointer by index (for reverse lookup)
    inline const KDNode* getNodeByIndex(size_t idx) const {
        return m_nodes + idx;
    }
    
    /// Total surface area
    inline Float getTotalArea() const { return m_totalArea; }
```

### 1.3 Sampler Utility for Geometry

```cpp
// mitsuba/include/mitsuba/render/geometrysampler.h

class GeometrySampler {
public:
    /// Sample a point on geometry using KDTree aggregates
    /// Traverses tree probabilistically based on diffuse reflectance + area
    bool sampleGeometry(
        const ShapeKDTree *kdtree,
        const Scene *scene,
        const Point2 &sample,
        Point &position,          ///< Output: sampled 3D position
        Normal &normal,           ///< Output: surface normal at position
        Float &pdf                ///< Output: probability of sampling this point
    ) const;
    
    /// Compute PDF of a given geometry point (reverse query)
    Float pdfGeometry(
        const ShapeKDTree *kdtree,
        const Point &position,
        const Normal &normal
    ) const;
};
```

---

## 2. Pre-Processing Pipeline: Building Geometric Aggregates

### 2.1 Workflow

After `scene->getKDTree()->build()` (in scene initialization or integrator setup):

1. **Traverse KDTree** (post-order DFS)
2. **For each leaf node:**
   - Extract shapes/primitives in range
   - Compute aggregates per shape: area, diffuse albedo, mean normal
   - Accumulate into leaf `KDNodeInfo`
3. **For each internal node:**
   - Combine children aggregates (area-weighted)
   - Propagate upward to root
4. **Normalize** diffuse albedo and normal distribution

### 2.2 Implementation Steps

#### Step 1: Compute Per-Shape Aggregates

```cpp
struct ShapeAggregate {
    Float area;
    Float diffuseAlbedo;     // Average over surface
    Vector meanNormal;       // Average normal direction
    Float normalVariance;    // Spread of normals
};

ShapeAggregate computeShapeAggregate(const Shape *shape);
```

**For triangle meshes:**
- Area: sum of triangle areas (efficient)
- Diffuse albedo: sample BSDF at surface points (e.g., 64 stratified samples)
- Mean normal: area-weighted average of vertex/face normals
- Normal variance: compute deviation from mean (approximation for SGGX parameters)

**For non-triangle shapes:**
- Area: `shape->getSurfaceArea()` if available
- Diffuse albedo: sample BSDF at random positions + average reflectance
- Mean normal: sample surface normals randomly, compute mean/variance
- (Requires PositionSamplingRecord + BSDFSamplingRecord introspection)

#### Step 2: Recursive Node Aggregation

```cpp
void buildGeometricAggregates_Recursive(
    const KDNode *node,
    std::vector<KDNodeInfo> &nodeInfo,
    const std::vector<ShapeAggregate> &shapeAggregates
);
```

**Leaf node:** Directly set `KDNodeInfo` from shapes in primitive range.

**Internal node:** 
- Retrieve children aggregates
- Combine area-weighted: 
  ```
  node.area = left.area + right.area
  node.albedo = (left.area * left.albedo + right.area * right.albedo) / node.area
  node.normal = (left.area * left.normal + right.area * right.normal) / node.area
  node.variance = weighted average of variances + variance of means
  ```

---

## 3. Geometry Sampling Algorithm

### 3.1 Adaptive KDTree Traversal

**Goal:** Sample a point on geometry, weighted by diffuse albedo and area, with PDF for MIS.

**Algorithm:**

```
Input: KDTree root node, 2D random sample (s1, s2)
Output: position p, normal n, PDF pdf_geom

// Phase 1: Adaptive tree traversal
node = root
pdf_traversal = 1.0

while not node.isLeaf():
    left_child, right_child = node children
    left_info = m_nodeInfo[left_child_index]
    right_info = m_nodeInfo[right_child_index]
    
    // Compute branch weights (albedo * area)
    left_weight = left_info.diffuseAlbedo * left_info.surfaceArea
    right_weight = right_info.diffuseAlbedo * right_info.surfaceArea
    
    total_weight = left_weight + right_weight
    left_prob = left_weight / total_weight
    
    // Branch probabilistically
    if s1 < left_prob:
        node = left_child
        pdf_traversal *= left_prob
        s1 = s1 / left_prob
    else:
        node = right_child
        pdf_traversal *= (1 - left_prob)
        s1 = (s1 - left_prob) / (1 - left_prob)

// Phase 2: Sample primitive in leaf
primitives = node.getPrimitiveRange()
primitive_idx = s2 * primitives.count()  // uniform among primitives in leaf

shape_index, local_primitive = resolve_primitive(primitive_idx)
shape = scene.getShapes()[shape_index]

// Phase 3: Sample position on shape
PositionSamplingRecord pRec;
pRec.uv = (next_random(), next_random())  // or use remaining randomness from s1, s2
shape->samplePosition(pRec, uv_sample)

p = pRec.p
n = pRec.n
pdf_shape = pRec.pdf * (1.0 / num_primitives_in_leaf)

// Phase 4: Combine PDFs
pdf_geom = pdf_traversal * pdf_shape

return (p, n, pdf_geom)
```

**Complexity:** O(tree depth) ≈ O(log N) for N primitives.

### 3.2 Reverse PDF Query

For MIS weighting, we need `pdfGeometry(point, normal)`:

```
Input: position p, normal n
Output: pdf_geom

// Find leaf node containing p
node = find_leaf_node(p)

// Reconstruct PDF path from root to leaf
pdf_traversal = 1.0
node = root
while not node.isLeaf():
    left_info, right_info = child aggregates
    left_weight = left_info.albedo * left_info.area
    right_weight = right_info.albedo * right_info.area
    total_weight = left_weight + right_weight
    
    if point_in_aabb(p, left_child_aabb):
        pdf_traversal *= left_weight / total_weight
        node = left_child
    else:
        pdf_traversal *= right_weight / total_weight
        node = right_child

// Shape PDF within leaf
pdf_shape = (1.0 / shape_area) * (1.0 / num_primitives_in_leaf)

return pdf_traversal * pdf_shape
```

---

## 4. Additional Vertex Integrator Integration

### 4.1 Modified Li() Structure

Base on `additionalvertex.cpp` but add geometry sampling:

```cpp
Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
    // ... initialization ...
    
    while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
        // ... standard direct illumination (NEE) ...
        
        // === NEW: Geometry-Aware Sampling ===
        
        bool useGeometrySampling = false;
        Point geomPosition;
        Normal geomNormal;
        Float geomPdf = 0.0f;
        
        if (m_geometrySamplingFraction > 0 && rRec.nextSample1D() < m_geometrySamplingFraction) {
            useGeometrySampling = m_geometrySampler.sampleGeometry(
                scene->getKDTree(),
                scene,
                rRec.nextSample2D(),
                geomPosition,
                geomNormal,
                geomPdf
            );
        }
        
        // === Standard BSDF Sampling ===
        Float bsdfPdf;
        BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
        Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
        
        if (bsdfWeight.isZero() && !useGeometrySampling)
            break;
        
        // === Choose/mix sampling strategies ===
        Vector wo;
        Float finalPdf;
        
        if (useGeometrySampling && rRec.nextSample1D() < 0.5f) {
            // Geometry sampling: cast ray from current vertex to sampled geometry
            wo = normalize(geomPosition - its.p);
            finalPdf = geomPdf;
        } else if (bsdfWeight.isZero()) {
            break;  // Fallback required but failed
        } else {
            // BSDF sampling
            wo = its.toWorld(bRec.wo);
            finalPdf = bsdfPdf;
        }
        
        // ... continue with standard path (ray cast, MIS weight, etc.) ...
        throughput *= bsdfVal;
        
        // ... Russian roulette, depth increment ...
    }
    
    return Li;
}
```

### 4.2 Configuration Parameters

Add to integrator properties:

```cpp
// Float m_geometrySamplingFraction [0, 1]
//    Probability of using geometry sampling at each vertex
//    0.0 = standard path tracer
//    1.0 = always geometry sampling
//    0.5 = 50/50 mix (MIS both strategies)

// Bool m_enableNDFWeighting
//    If true, further weight geometry sampling by normal distribution
//    matching the sampled point normal to current surface orientation
```

---

## 5. Implementation Steps (Sequence)

### Phase 1: Data Structures & Pre-Processing

1. Create `kdnodeinfo.h` with `KDNodeInfo` struct
2. Extend `skdtree.h` with:
   - `m_nodeInfo` vector
   - `buildGeometricAggregates()` method signature
3. Implement `buildGeometricAggregates()` in `skdtree.cpp`:
   - `computeShapeAggregate()` helper (sample BSDF, compute mean normal, variance)
   - Recursive tree traversal with area-weighted aggregation
4. Expose node access methods (`getNodeInfo`, `getNodeByIndex`)

### Phase 2: Scene Integration

8. Modify `Scene::initialize()` to call:
   ```cpp
   m_kdtree->buildGeometricAggregates();
   ```

### Phase 3: Geometry Sampler

5. Create `geometrysampler.h` and `geometrysampler.cpp`
6. Implement `GeometrySampler::sampleGeometry()`:
   - Adaptive tree traversal (albedo × area weighting)
   - Primitive selection and shape sampling
   - PDF computation
7. Implement `GeometrySampler::pdfGeometry()`:
   - Reverse traversal to reconstruct PDF

### Phase 4: Additional Vertex Integrator

9. Update `additionalvertex.cpp`:
   - Add member `GeometrySampler m_geometrySampler`
   - Add member `Float m_geometrySamplingFraction` (parameter)
   - Add parameter parsing in constructor
10. Modify `Li()` to call geometry sampler
11. Implement sampling strategy blending (choice or MIS)
12. Update `toString()` for diagnostics

### Phase 5: Validation & Tuning

13. Test on sparse geometry scene (e.g., Stanford bunny in large room)
14. Measure convergence vs. standard path tracer
15. Benchmark KDTree aggregation overhead
16. Tune `m_geometrySamplingFraction` parameter

---

## 6. File Structure & Locations

```
mitsuba/
├── include/mitsuba/render/
│   ├── kdnodeinfo.h          (NEW)
│   ├── geometrysampler.h      (NEW)
│   └── skdtree.h              (MODIFIED: add methods, m_nodeInfo)
│
├── src/librender/
│   ├── scene.cpp              (MODIFIED: call buildGeometricAggregates)
│   ├── skdtree.cpp            (MODIFIED: implement buildGeometricAggregates)
│   └── geometrysampler.cpp    (NEW)
│
└── src/integrators/additionalvertex/
    └── additionalvertex.cpp   (MODIFIED: integrate sampling strategy)
```

---

## 7. Normal Distribution Function (NDF) Representation

### 7.1 SGGX Simplified (Isotropic)

For now, store per node:
- `meanNormal`: principal normal direction
- `variance`: standard deviation from mean

**Sampling from NDF (optional future enhancement):**
```cpp
Vector sampleNDF(const Normal &meanN, Float variance, const Point2 &sample) {
    // Hemispherical Gaussian lobe centered on meanN
    Float phi = 2 * M_PI * sample.x;
    Float cosTheta = exp(-variance * (1 - cos(theta))); // Gaussian envelope
    Vector localDir = spherical_to_cartesian(phi, cosTheta);
    return frame(meanN).toWorld(localDir);
}
```

**For future:** Extend to full SGGX with covariance matrix for anisotropic normals.

### 7.2 Rough Normals Handling

If a surface region has high normal variance (curved geometry):
- Store `variance` to indicate roughness
- Option: Weighted geometry sampling by `1 / (1 + variance)` to prefer flatter surfaces
- Or: Track normal distribution and weight incoming rays by alignment to expected normal

---

## 8. Known Limitations & Future Extensions

1. **Diffuse-only:** No specular/glossy contribution to geometry selection. Future: blend with specular color.
2. **No occlusion:** Geometry sampling ignores visibility (can sample invisible surfaces). Future: hierarchical occlusion tests via shadow rays.
3. **Uniform primitives in leaf:** All primitives in a leaf are sampled uniformly. Future: further subdivision by shape or material.
4. **Static aggregates:** Assumes static geometry. For animated scenes, aggregates must be recomputed per frame.
5. **SGGX simplified:** Currently isotropic. Future: full anisotropic NDF for better normal estimation.

---

## 9. Testing Strategy

### 9.1 Simple Test Scene
- Cornell box with 1-2 small objects (bunny, dragon)
- Test convergence with/without geometry sampling

### 9.2 Sparse Scene
- Large empty room with small high-albedo object
- Geometry sampling should dominate BSDF sampling

### 9.3 Metrics
- Sample-to-convergence ratio
- KDTree aggregation time
- Sample generation time per ray (geometry vs. BSDF)
- Variance reduction (MSE vs. standard path tracer)

---

## 10. Code Interdependencies

- **skdtree.h/cpp** ← foundational (must be done first)
- **geometrysampler.h/cpp** ← depends on kdnodeinfo, skdtree
- **additionalvertex.cpp** ← depends on geometrysampler
- **scene.cpp** ← minimal: just call buildGeometricAggregates()

---

## Summary

A phased approach to build geometry-aware aggregates on the existing ShapeKDTree, implement intelligent geometry sampling via tree traversal, and integrate into a new Additional Vertex Integrator. The integrator can blend standard BSDF sampling with direct geometry sampling, with parameter control for sparse vs. dense geometry scenes.
