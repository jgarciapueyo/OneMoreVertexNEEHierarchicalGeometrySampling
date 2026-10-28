/*
    Hierarchical geometry sampling: stochastic top-down traversal of the
    SamplingBVH (Sec. 4.1 of the paper), leaf sampling (Sec. 5) and the
    evaluation of the corresponding densities.
*/

#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/render/hgs/spherical_aabb.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/trimesh.h>
#include <cmath>

/* Define HGS_DEBUG_NAN to reject (and log) samples whose traversal weights or
   densities are not finite, instead of letting them propagate silently */
// #define HGS_DEBUG_NAN

#if defined(HGS_DEBUG_NAN)
#define HGS_REJECT_IF_INVALID(value, what) do { \
        if (!std::isfinite(value) || (value) < 0) { \
            SLog(EWarn, "SamplingBVH: invalid %s (%f)", what, (double) (value)); \
            return 0; \
        } \
    } while (0)
#else
#define HGS_REJECT_IF_INVALID(value, what) do { } while (0)
#endif

MTS_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------
// Traversal (Eqs. 8 and 9)
// ---------------------------------------------------------------------------

/* Descends from the root, choosing each child with probability proportional
   to its importance weight w(N, x_s, x_e) (plus the defensive term). Returns
   the node where the traversal stopped (a leaf, or an interior node when
   lightcuts are enabled) and accumulates the discrete traversal probability. */
template <typename ImportanceFn>
size_t SamplingBVH::sampleTraversal(Sampler *sampler, const ImportanceFn &importance,
        Float &pdf_traversal) const {
    size_t nodeIndex = 0;
    pdf_traversal = 1.0f;

    while (!m_nodes[nodeIndex].isLeaf()) {
        const size_t leftIdx = nodeIndex + 1;
        const size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;

        const Float left_importance = importance(leftIdx) * m_probmult + m_defensive_pdf;
        const Float right_importance = importance(rightIdx) * m_probmult + m_defensive_pdf;
        const Float total_importance = left_importance + right_importance;

        /* Lightcuts: stop when both children have negligible weight */
        if (m_enableLightcuts && total_importance < m_lightcutThreshold * 2.f * m_defensive_pdf)
            break;

        const Float left_prob = left_importance / total_importance;
        if (sampler->next1D() < left_prob) {
            nodeIndex = leftIdx;
            pdf_traversal *= left_prob;
        } else {
            nodeIndex = rightIdx;
            pdf_traversal *= (1.0f - left_prob);
        }
    }
    return nodeIndex;
}

/* Probability with which sampleTraversal() reaches the node containing the
   leaf-ordered primitive primIdx. Returns the node where the traversal stops
   and, in nodeStart, the index of its first primitive. Returns -1 when the
   probability is zero. */
template <typename ImportanceFn>
size_t SamplingBVH::pdfTraversal(uint32_t primIdx, const ImportanceFn &importance,
        Float &pdf_traversal, uint32_t &nodeStart) const {
    size_t nodeIndex = 0;
    nodeStart = 0;
    pdf_traversal = 1.0f;

    while (!m_nodes[nodeIndex].isLeaf()) {
        const size_t leftIdx = nodeIndex + 1;
        const size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;

        const Float left_imp = importance(leftIdx) * m_probmult + m_defensive_pdf;
        const Float right_imp = importance(rightIdx) * m_probmult + m_defensive_pdf;
        const Float total = left_imp + right_imp;

        if (total <= 0.0f)
            return (size_t) -1;

        if (m_enableLightcuts && total < m_lightcutThreshold * 2.f * m_defensive_pdf)
            break;

        const uint32_t leftCount = m_nodes[leftIdx].nSubtreePrimitives;
        if (primIdx < nodeStart + leftCount) {
            pdf_traversal *= (left_imp / total);
            nodeIndex = leftIdx;
        } else {
            pdf_traversal *= (right_imp / total);
            nodeIndex = rightIdx;
            nodeStart += leftCount;
        }
    }
    return nodeIndex;
}

bool SamplingBVH::sampleFromNode(const Scene *scene, Sampler *sampler, size_t nodeIndex,
        const Point &from, Intersection &its, Float &pdf_sa) const {
    const BVHNode &node = m_nodes[nodeIndex];
    if (node.isLeaf()) {
        if (node.nLeafPrimitives == 0)
            return false;
        if (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB ||
            m_samplingMode == SamplingBVHSamplingMode::SphericalAABB_anyNode)
            return sampleLeafSphericalAABB(scene, node, nodeIndex, sampler->next2D(), from, its, pdf_sa);
        return sampleLeafPrimitive(scene, node, nodeIndex, sampler->next1D(), sampler->next2D(), from, its, pdf_sa);
    } else if (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB_anyNode) {
        /* The traversal stopped at an interior node (lightcuts) */
        return sampleLeafSphericalAABB(scene, node, nodeIndex, sampler->next2D(), from, its, pdf_sa);
    } else {
        /* The traversal stopped at an interior node (lightcuts): descend by area */
        return sampleNodePrimitive(scene, nodeIndex, sampler, from, its, pdf_sa, true);
    }
}

Float SamplingBVH::pdfFromNode(const Scene *scene, size_t nodeIndex, uint32_t nodeStart,
        uint32_t primIdx, uint32_t meshIndex, uint32_t triangleIndex,
        const Point &position, const Point &from) const {
    const BVHNode &node = m_nodes[nodeIndex];
    const BVHNodeInfo &nodeInfo = m_nodeInfos[nodeIndex];

    if (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB) {
        /* Mimic the area-proportional descent of sampleNodePrimitive() down to
           the leaf that contains the primitive (only taken with lightcuts) */
        Float descent_prob = 1.0f;
        size_t currentIdx = nodeIndex;
        uint32_t currentStart = nodeStart;

        while (!m_nodes[currentIdx].isLeaf()) {
            const size_t leftIdx = currentIdx + 1;
            const size_t rightIdx = m_nodes[currentIdx].secondChildOffset;

            const float left_area = m_nodeInfos[leftIdx].surfaceArea;
            const float right_area = m_nodeInfos[rightIdx].surfaceArea;
            if ((left_area + right_area) <= 0.0f)
                return 0.0f;

            const float left_prob = left_area / (left_area + right_area);
            const uint32_t leftCount = m_nodes[leftIdx].nSubtreePrimitives;

            if (primIdx < currentStart + leftCount) {
                descent_prob *= left_prob;
                currentIdx = leftIdx;
            } else {
                descent_prob *= (1.0f - left_prob);
                currentIdx = rightIdx;
                currentStart += leftCount;
            }
        }

        Vector wo = position - from;
        const Float dist = std::sqrt(wo.lengthSquared());
        if (dist <= Epsilon)
            return 0.0f;
        wo /= dist;

        Float pdf_solid_angle = 0.0f;
        AABBSphPdf(m_nodes[currentIdx].bounds, from, wo, pdf_solid_angle);
        return descent_prob * pdf_solid_angle;
    } else if (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB_anyNode) {
        const Vector d = position - from;
        if (d.lengthSquared() <= Epsilon * Epsilon)
            return 0.0f;
        Float pdf_solid_angle = 0.0f;
        AABBSphPdf(node.bounds, from, normalize(d), pdf_solid_angle);
        return pdf_solid_angle;
    } else {
        /* Primitive: Prob(leaf | node) * Prob(prim | leaf) * pdf(p | prim) =
           (leaf_area / node_area) * (prim_area / leaf_area) * (1 / prim_area) = 1 / node_area */
        if (nodeInfo.surfaceArea <= 0.0f)
            return 0.0f;
        const Float pdf_area = 1.0f / nodeInfo.surfaceArea;

        const TriMesh *mesh = scene->getMeshes()[meshIndex];
        const Triangle &tri = mesh->getTriangles()[triangleIndex];
        const Point *verts = mesh->getVertexPositions();
        const Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
        const Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
        const Normal n = normalize(cross(e1, e2));

        /* Convert the area density into solid angle at 'from' */
        const Vector d = position - from;
        const Float dist_sq = d.lengthSquared();
        const Float dist = std::sqrt(dist_sq);
        if (dist <= Epsilon)
            return 0.0f;
        const Vector wo = d / dist;
        const Float cos_theta = std::abs(dot(n, -wo));
        if (cos_theta <= Epsilon)
            return 0.0f;

        return pdf_area * (dist_sq / cos_theta);
    }
}

uint32_t SamplingBVH::lookupPrimitive(const Intersection &its, uint32_t &meshIndex) const {
    meshIndex = 0xFFFFFFFFu;
    const TriMesh *hitMesh = dynamic_cast<const TriMesh *>(its.shape);
    if (hitMesh == NULL)
        return 0xFFFFFFFFu;
    meshIndex = getMeshIndex(hitMesh);
    if (meshIndex == 0xFFFFFFFFu || its.primIndex >= hitMesh->getTriangleCount()
        || meshIndex >= m_meshOffsets.size())
        return 0xFFFFFFFFu;
    const uint32_t lookupIdx = m_meshOffsets[meshIndex] + its.primIndex;
    if (lookupIdx >= m_triangleToPrim.size())
        return 0xFFFFFFFFu;
    return m_triangleToPrim[lookupIdx];
}

// ---------------------------------------------------------------------------
// Sampling and PDF evaluation from a surface vertex x_s
// ---------------------------------------------------------------------------

bool SamplingBVH::sampleGeometry(
    const Scene *scene,
    Sampler *sampler,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    Intersection &its_xp,
    Float &pdf) const
{
    if (m_nodes.empty() || m_primitives.empty())
        return false;

    const SamplingBVH *bvh = this;
    auto importance = [&](size_t nodeIdx) {
        return computeNodeImportance(m_scene, bvh, (int) nodeIdx, its_xs, pRec_xe);
    };

    Float pdf_traversal;
    size_t nodeIndex = sampleTraversal(sampler, importance, pdf_traversal);
    HGS_REJECT_IF_INVALID(pdf_traversal, "traversal probability");

    Float pdf_leaf = 0.0f;
    if (!sampleFromNode(scene, sampler, nodeIndex, its_xs.p, its_xp, pdf_leaf))
        return false;

    /* Density per unit solid angle at x_s */
    pdf = pdf_traversal * pdf_leaf;
    HGS_REJECT_IF_INVALID(pdf, "sample density");
    return true;
}

Float SamplingBVH::pdfGeometry(
    const Scene *scene,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    const Intersection &its_xp
) const {
    if (its_xp.shape == NULL || m_nodes.empty())
        return 0.0f;

    uint32_t meshIndex;
    const uint32_t primIdx = lookupPrimitive(its_xp, meshIndex);
    if (primIdx == 0xFFFFFFFFu)
        return 0.0f;

    const SamplingBVH *bvh = this;
    auto importance = [&](size_t nodeIdx) {
        return computeNodeImportance(m_scene, bvh, (int) nodeIdx, its_xs, pRec_xe);
    };

    Float pdf_traversal;
    uint32_t nodeStart;
    const size_t nodeIndex = pdfTraversal(primIdx, importance, pdf_traversal, nodeStart);
    if (nodeIndex == (size_t) -1)
        return 0.0f;

    const Float pdf_leaf = pdfFromNode(scene, nodeIndex, nodeStart, primIdx, meshIndex,
        its_xp.primIndex, its_xp.p, its_xs.p);
    return pdf_traversal * pdf_leaf;
}

// ---------------------------------------------------------------------------
// Sampling and PDF evaluation from the camera (x_s on the sensor)
// ---------------------------------------------------------------------------

bool SamplingBVH::sampleGeometryCamera(
    const Scene *scene,
    Sampler *sampler,
    const Point &x_o,
    const Sensor *sensor,
    const PositionSamplingRecord &pRec_xe,
    Intersection &its_xn,
    Float &pdf) const
{
    if (m_nodes.empty() || m_primitives.empty())
        return false;

    const SamplingBVH *bvh = this;
    auto importance = [&](size_t nodeIdx) {
        return computeNodeImportanceGlint(scene, bvh, (int) nodeIdx, x_o, sensor, pRec_xe);
    };

    Float pdf_traversal;
    size_t nodeIndex = sampleTraversal(sampler, importance, pdf_traversal);

    Float pdf_leaf = 0.0f;
    if (!sampleFromNode(scene, sampler, nodeIndex, x_o, its_xn, pdf_leaf))
        return false;

    // pdf_leaf is per unit solid angle at x_o; return a density per unit area at x_n
    Vector d = its_xn.p - x_o;
    Float dist2 = d.lengthSquared();
    if (dist2 <= 0.0f) return false;
    Float cosTheta = absDot(its_xn.geoFrame.n, d) / std::sqrt(dist2);
    if (cosTheta <= 0.0f) return false;

    pdf = pdf_traversal * pdf_leaf * cosTheta / dist2;
    return true;
}

Float SamplingBVH::pdfGeometryCamera(
    const Scene *scene,
    const Point &x_o,
    const Sensor *sensor,
    const PositionSamplingRecord &pRec_xe,
    const Intersection &its_xn
) const {
    if (its_xn.shape == NULL || m_nodes.empty())
        return 0.0f;

    uint32_t meshIndex;
    const uint32_t primIdx = lookupPrimitive(its_xn, meshIndex);
    if (primIdx == 0xFFFFFFFFu)
        return 0.0f;

    const SamplingBVH *bvh = this;
    auto importance = [&](size_t nodeIdx) {
        return computeNodeImportanceGlint(scene, bvh, (int) nodeIdx, x_o, sensor, pRec_xe);
    };

    Float pdf_traversal;
    uint32_t nodeStart;
    const size_t nodeIndex = pdfTraversal(primIdx, importance, pdf_traversal, nodeStart);
    if (nodeIndex == (size_t) -1)
        return 0.0f;

    const Float pdf_leaf = pdfFromNode(scene, nodeIndex, nodeStart, primIdx, meshIndex,
        its_xn.primIndex, its_xn.p, x_o);
    return pdf_traversal * pdf_leaf;
}

// ---------------------------------------------------------------------------
// Leaf sampling (Sec. 5)
// ---------------------------------------------------------------------------

// Samples a primitive of the leaf proportionally to its area (alias table),
// then a uniform point on it
bool SamplingBVH::sampleLeafPrimitive(
    const Scene *scene,
    const BVHNode &leafNode,
    size_t leafNodeIndex,
    const Float &sample1,
    const Point2 &sample2,
    const Point &xs,
    Intersection &its_xp,
    Float &pdf_xp,
    bool return_solidangle_pdf) const
{
    if (leafNode.nLeafPrimitives == 0) return false;

    // 1) O(1) triangle selection via per-leaf alias table
    const bool hasAliasMeta = (leafNodeIndex < m_leafAliasTableOffsets.size());
    const uint32_t aliasOffset = hasAliasMeta ? m_leafAliasTableOffsets[leafNodeIndex] : 0xFFFFFFFFu;
    uint32_t localPrimIdx = 0;

    if (aliasOffset != 0xFFFFFFFFu && leafNode.nLeafPrimitives > 1) {
        const Float u = std::min(std::max(sample1, 0.0f), std::nextafter(1.0f, 0.0f));
        const Float scaled = u * (Float) leafNode.nLeafPrimitives;
        const uint32_t bucket = (uint32_t) std::min(
            (uint32_t) std::floor(scaled),
            leafNode.nLeafPrimitives - 1
        );
        const Float frac = scaled - (Float) bucket;

        const Float q = m_leafAliasProb[aliasOffset + bucket];
        const uint32_t a = m_leafAliasAlias[aliasOffset + bucket];
        localPrimIdx = (frac < q) ? bucket : a;
    } else if (leafNode.nLeafPrimitives > 1) {
        // Correctness fallback: preserve area-proportional sampling even if alias data is missing.
        Float total_leaf_area = 0.0f;
        for (uint32_t i = 0; i < leafNode.nLeafPrimitives; ++i) {
            const BVHPrimitive &p = getPrimitive(leafNode.primitivesOffset + i);
            const TriMesh *m = scene->getMeshes()[p.meshIndex];
            const Triangle &t = m->getTriangles()[p.triangleIndex];
            total_leaf_area += t.surfaceArea(m->getVertexPositions());
        }

        if (total_leaf_area <= 0.0f)
            return false;

        const Float target_area = sample1 * total_leaf_area;
        Float accumulated_area = 0.0f;
        localPrimIdx = leafNode.nLeafPrimitives - 1;

        for (uint32_t i = 0; i < leafNode.nLeafPrimitives; ++i) {
            const BVHPrimitive &p = getPrimitive(leafNode.primitivesOffset + i);
            const TriMesh *m = scene->getMeshes()[p.meshIndex];
            const Triangle &t = m->getTriangles()[p.triangleIndex];

            accumulated_area += t.surfaceArea(m->getVertexPositions());
            if (accumulated_area >= target_area) {
                localPrimIdx = i;
                break;
            }
        }
    }

    // 2. Fetch the chosen primitive and sample a point on it
    const BVHPrimitive &prim = getPrimitive(leafNode.primitivesOffset + localPrimIdx);
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    const TriMesh *mesh = meshes[prim.meshIndex];
    const Triangle& tri = mesh->getTriangles()[prim.triangleIndex];

    PositionSamplingRecord pRec;
    pRec.p = tri.sample(mesh->getVertexPositions(), mesh->getVertexNormals(),
        mesh->getVertexTexcoords(), pRec.n, pRec.uv, sample2);

    // 3. Prob(prim | leaf) * pdf(p | prim) = (prim_area / leaf_area) * (1 / prim_area) = 1 / leaf_area
    const Float total_leaf_area = (Float) m_nodeInfos[leafNodeIndex].surfaceArea;
    if (total_leaf_area <= 0.0f) return false;
    Float pdf_area = 1.0f / total_leaf_area;

    // 4. Fill the intersection record
    its_xp.p = pRec.p;
    its_xp.uv = pRec.uv;
    its_xp.shape = mesh;
    its_xp.instance = NULL;
    its_xp.primIndex = prim.triangleIndex;
    its_xp.t = 1.f;

    const Point *verts = mesh->getVertexPositions();
    Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
    Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
    its_xp.geoFrame = Frame(normalize(cross(e1, e2)));
    its_xp.shFrame = Frame(pRec.n);
    its_xp.wi = its_xp.toLocal(normalize(xs - its_xp.p));

    if (return_solidangle_pdf) {
        // 5. Convert Area PDF -> Solid Angle PDF
        // P_sa = P_area * (dist^2 / cos_theta)
        Vector d = its_xp.p - xs;
        Float dist_sq = d.lengthSquared();
        Float dist = std::sqrt(dist_sq);
        Vector wo = d / dist;
        Float cos_theta = std::abs(dot(its_xp.geoFrame.n, -wo));
        if (cos_theta <= Epsilon)
            return false;

        pdf_xp = pdf_area * (dist_sq / cos_theta);
    } else {
        pdf_xp = pdf_area;
    }

    return true;
}

// Descends from an interior node to a leaf by area, then samples the leaf
bool SamplingBVH::sampleNodePrimitive(
    const Scene *scene,
    size_t nodeIdx,
    Sampler* sampler,
    const Point &xs,
    Intersection &its_xp,
    Float &pdf_xp,
    bool return_solidangle_pdf
) const {
    float pdf_leaf = 1.f;
    while(!m_nodes[nodeIdx].isLeaf()) {
        const size_t leftIdx = nodeIdx + 1;
        const size_t rightIdx = m_nodes[nodeIdx].secondChildOffset;
        float left_area = m_nodeInfos[leftIdx].surfaceArea;
        float right_area = m_nodeInfos[rightIdx].surfaceArea;

        float left_prob = left_area / (left_area + right_area);

        // Pick a child proportionally to its surface area
        if (sampler->next1D() < left_prob) {
            nodeIdx = leftIdx;
            pdf_leaf *= left_prob;
        } else {
            nodeIdx = rightIdx;
            pdf_leaf *= (1.0f - left_prob);
        }

    }
    bool success;
    if (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB) {
        success = sampleLeafSphericalAABB(scene, m_nodes[nodeIdx], nodeIdx, sampler->next2D(), xs, its_xp, pdf_xp);
    } else {
        success = sampleLeafPrimitive(scene, m_nodes[nodeIdx], nodeIdx, sampler->next1D(), sampler->next2D(), xs, its_xp, pdf_xp, return_solidangle_pdf);
    }
    pdf_xp *= pdf_leaf;

    return success;
}

// Samples a direction in the solid angle of the node's AABB and ray-casts it;
// the hit must belong to the node
bool SamplingBVH::sampleLeafSphericalAABB(
    const Scene *scene,
    const BVHNode &leafNode,
    size_t leafNodeIndex,
    const Point2 &sample2,
    const Point &xs,
    Intersection &its_xp,
    Float &pdf_xp) const
{
    if (m_samplingMode != SamplingBVHSamplingMode::SphericalAABB_anyNode && leafNode.nLeafPrimitives == 0) return false;

    Vector wo;
    Float pdf_aabb_solid_angle;

    // Naturally samples in Solid Angle
    AABBSphSample(leafNode.bounds, xs, sample2.x, sample2.y, wo, pdf_aabb_solid_angle);


    if (pdf_aabb_solid_angle <= 0.0f || !std::isfinite(pdf_aabb_solid_angle)) return false;

    Ray ray(xs, wo, Epsilon, std::numeric_limits<Float>::infinity(), 0.f);
    if (!scene->rayIntersect(ray, its_xp)) return false;

    if (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB_anyNode) {
        // In this mode, the "leafNode" may not necessarily be a leaf!
        if (!intersectionInNode(scene, its_xp, leafNodeIndex)) return false;
    }
    else {
        if (!intersectionInLeafNode(scene, its_xp, leafNodeIndex)) return false;
    }
    // The density stays in solid angle at xs (an area density would need the
    // cos / dist^2 factor, which vanishes at grazing angles)
    pdf_xp = pdf_aabb_solid_angle;

    return true;
}

bool SamplingBVH::intersectionInNode(
    const Scene *scene,
    const Intersection &its,
    size_t targetNodeIndex) const
{
    if (!scene || targetNodeIndex >= m_nodes.size() || its.shape == NULL)
        return false;

    uint32_t meshIndex;
    const uint32_t primIdx = lookupPrimitive(its, meshIndex);
    if (primIdx == 0xFFFFFFFFu)
        return false;

    // Top-down traversal towards the primitive, checking whether targetNodeIndex is on the way
    size_t currNodeIndex = 0;
    uint32_t nodeStart = 0;

    while (true) {
        // If we hit the target node along the path to the primitive, it belongs to this subtree!
        if (currNodeIndex == targetNodeIndex) {
            return true;
        }

        const BVHNode &currNode = m_nodes[currNodeIndex];

        // If we reach a leaf and haven't matched the target, the primitive is somewhere else
        if (currNode.isLeaf()) {
            return false;
        }

        // Fetch children
        size_t leftIdx = currNodeIndex + 1;
        size_t rightIdx = currNode.secondChildOffset;

        uint32_t leftCount = m_nodes[leftIdx].nSubtreePrimitives;

        // Route the traversal toward the primitive's actual location
        if (primIdx < nodeStart + leftCount) {
            currNodeIndex = leftIdx;
        } else {
            currNodeIndex = rightIdx;
            nodeStart += leftCount;
        }
    }

    return false;
}

bool SamplingBVH::intersectionInLeafNode(
    const Scene *scene,
    const Intersection &its,
    size_t leafNodeIndex) const
{
    if (!scene || leafNodeIndex >= m_nodes.size() || its.shape == NULL)
        return false;

    const BVHNode &leafNode = m_nodes[leafNodeIndex];
    if (!leafNode.isLeaf())
        return false;

    uint32_t meshIndex;
    const uint32_t primIdx = lookupPrimitive(its, meshIndex);
    if (primIdx == 0xFFFFFFFFu)
        return false;

    const uint32_t leafStart = leafNode.primitivesOffset;
    const uint32_t leafEnd = leafStart + leafNode.nLeafPrimitives;
    return primIdx >= leafStart && primIdx < leafEnd;
}

// ---------------------------------------------------------------------------
// Contribution of the three-vertex subpaths
// ---------------------------------------------------------------------------

Spectrum evalLength2Contribution(
    const Scene *scene,
    const Intersection &its_xs,
    const Intersection &its_xp,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility
) {
    if (!pRec_xe.object)
        return Spectrum(0.0f);
    // 1. Evaluate visibility if needed
    // 1.1. Visibility between xs and xp
    Vector d_sp = its_xp.p - its_xs.p;
    Float dist_sp = d_sp.length();
    d_sp /= dist_sp;

    Ray shadowRay_sp(its_xs.p, d_sp, Epsilon, dist_sp * (1.0f - ShadowEpsilon), 0.f);
    if (checkVisibility && scene->rayIntersect(shadowRay_sp)) {
        return Spectrum(0.0f);
    }

    // 1.2. Visibility between xp and xe
    Vector d_pe = pRec_xe.p - its_xp.p;
    Float dist_pe = d_pe.length();
    d_pe /= dist_pe;

    Ray shadowRay_pe(its_xp.p, d_pe, Epsilon, dist_pe * (1.0f - ShadowEpsilon), 0.f);
    if (checkVisibility && scene->rayIntersect(shadowRay_pe)) {
        return Spectrum(0.0f);
    }
    // 2. Evaluate contribution of this subpath: xs -> xp -> xe
    // 2.1. Evaluate bsdf
    // 2.1.1. Evaluate BSDF at xs for the direction towards xp
    Vector l_xs_to_xp = its_xs.toLocal(d_sp);
    // 2.1.2. Local directions at xp
    Vector l_xp_in  = its_xp.toLocal(-d_sp);
    Vector l_xp_out = its_xp.toLocal(d_pe);

    BSDFSamplingRecord bRec_xs(its_xs, l_xs_to_xp);
    const BSDF *bsdf_xs = its_xs.getBSDF();
    if (!bsdf_xs)
        return Spectrum(0.0f);
    Spectrum f_xs = bsdf_xs->eval(bRec_xs);
    BSDFSamplingRecord bRec_xp(its_xp, l_xp_in, l_xp_out);


    const BSDF *bsdf_xp = its_xp.getBSDF();
    if (!bsdf_xp)
        return Spectrum(0.0f);
    Spectrum f_xp = bsdf_xp->eval(bRec_xp);

    if (f_xs.isZero() || f_xp.isZero()) {
        return Spectrum(0.0f);
    }

    // 2.2. Evaluate emitter radiance at xe in the direction of xp
    const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
    DirectionSamplingRecord dRec_xe(-d_pe);
    dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
    Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);

    if (Le.isZero())
        return Spectrum(0.0f);

    // 2.3. Geometry terms. BSDF::eval includes the cosine of its outgoing
    // direction: f_xs carries cos(theta_s -> p) and f_xp carries
    // cos(theta_p -> e); evalDirection carries the cosine at x_e. The
    // remaining factors are the cosine at x_p towards x_s and the inverse
    // squared distances.
    Float cos_theta_xp_in = std::abs(dot(its_xp.geoFrame.n, -d_sp));
    Float G_sp = cos_theta_xp_in / (dist_sp * dist_sp);
    Float G_pe = 1.f / (dist_pe * dist_pe);

    return f_xs * G_sp * f_xp * G_pe * Le;
}

Spectrum evalGlintContribution(
    const Scene *scene,
    const Intersection &its_xn,
    const PositionSamplingRecord &pRec_xe,
    const Vector &d_n_to_o,
    bool checkVisibility
) {
    if (!pRec_xe.object) {
        return Spectrum(0.f);
    }

    const BSDF *bsdf_xn = its_xn.getBSDF();
    if (!bsdf_xn) {
        return Spectrum(0.f);
    }

    // Direction and distance x_n → x_e
    Vector d_n_to_e = pRec_xe.p - its_xn.p;
    Float dist_ne = d_n_to_e.length();
    if (dist_ne < Epsilon) return Spectrum(0.f);
    d_n_to_e /= dist_ne;

    // Shadow ray x_n → x_e  (x_n → x_o visibility is done by sampleAttenuatedSensorDirect)
    if (checkVisibility) {
        Ray shadowRay(its_xn.p, d_n_to_e, Epsilon, dist_ne * (1.f - ShadowEpsilon), 0.f);
        if (scene->rayIntersect(shadowRay)) return Spectrum(0.f);
    }

    // BSDF at x_n: wi = toward camera (d_n_to_o), wo = toward emitter (d_n_to_e)
    // In Mitsuba's ERadiance convention, eval() returns f_r * cos(theta_wo) = f_r * cos(theta_xn->xe).
    // The rendering equation also requires cos(theta_xn->xo) for the G(x_n, x_o) geometry term.
    // ptracer gets this via bsdf->eval in EImportance mode (wi=toward sensor), but we evaluate
    // in ERadiance mode for the emitter side, so we must multiply cos(theta_xn->xo) explicitly.
    Vector l_n_in  = its_xn.toLocal(d_n_to_o);
    Vector l_n_out = its_xn.toLocal(d_n_to_e);
    Float cosCamera = Frame::cosTheta(l_n_in);  // cos(theta_xn -> x_o)
    if (cosCamera <= 0) return Spectrum(0.f);    // x_n faces away from camera
    BSDFSamplingRecord bRec(its_xn, l_n_in, l_n_out);
    Spectrum f_xn = bsdf_xn->eval(bRec);
    if (f_xn.isZero()) return Spectrum(0.f);

    // Emitter radiance at x_e toward x_n
    const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
    DirectionSamplingRecord dRec_xe(-d_n_to_e);
    dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
    Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);
    if (Le.isZero()) return Spectrum(0.f);

    // Geometry term x_n → x_e:
    //   cos(theta_xn->xe) absorbed into f_xn via BSDF eval (ERadiance: cos of wo)
    //   cos(theta_xe)     absorbed into Le via evalDirection
    //   1/dist_ne^2       explicit here
    Float G_ne = 1.f / (dist_ne * dist_ne);

    return cosCamera * f_xn * G_ne * Le;
}

MTS_NAMESPACE_END
