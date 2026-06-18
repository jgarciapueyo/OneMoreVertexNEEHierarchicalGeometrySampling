/*
    This file is part of Mitsuba, a physically based rendering system.

    BVH Geometry Sampling - Importance-driven geometry sampling via BVH traversal
*/

#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/render/bvh/spherical_aabb.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/trimesh.h>

MTS_NAMESPACE_BEGIN

// const Float DEFENSIVE_PDF = 1e-2f; // Minimum PDF to avoid 0 prob
// ^Nes: if we do MIS, we have to account for this defensive pdf when computing the pdf of a node
//       this will be hard according to Adrian, he recommends to skip MIS for now
// TO DO: also, we should compare the magnitude of this value to the typical node importance values 
//        we get from computeNodeImportance, to make sure it is not dominating the sampling


// #define ENABLE_LIGHTCUTS

#define LIGHTCUT_THRESHOLD 2.f // If the importance of a node is very low, treat it as a leaf to avoid unnecessary traversal. This threshold should be related to the DEFENSIVE_PDF to be effective.
// When enabled, directly samples nodes as if they were leafs when their importance is very low


bool GeometryBVH::sampleGeometry(
    const Scene *scene,
        Sampler *sampler,
        const Intersection &its_xs,
        const PositionSamplingRecord &pRec_xe,
        Intersection &its_xp,
        Float &pdf)
{
    if (m_nodes.empty() || m_primitives.empty()) {
        return false;
    }

    // 1. Importance-driven tree traversal
    size_t nodeIndex = 0;
    Float pdf_traversal = 1.0f;

    while (!m_nodes[nodeIndex].isLeaf()) {
        const size_t leftIdx = nodeIndex + 1;
        const size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;

        // 2. Compute importance weights for both children
        const Float left_importance = computeNodeImportance(
            m_scene,
            this,
            leftIdx,
            its_xs,
            pRec_xe,
            m_nodeInfos[leftIdx],
            m_nodes[leftIdx].bounds,
            m_sggxCrossSectionScale, m_solidAngleScale, m_variancePenaltyScale, m_cosinePenaltyScale
        ) * m_probmult + m_defensive_pdf;
        
        const Float right_importance = computeNodeImportance(
            m_scene,
            this,
            rightIdx,
            its_xs,
            pRec_xe,
            m_nodeInfos[rightIdx],
            m_nodes[rightIdx].bounds,
            m_sggxCrossSectionScale, m_solidAngleScale, m_variancePenaltyScale, m_cosinePenaltyScale
        ) * m_probmult + m_defensive_pdf;

        const Float total_importance = left_importance + right_importance;

        // // Error handling: if both importances are zero, we cannot continue
        // if (total_importance > 2.301f * m_defensive_pdf) {
        //     std::cout << "Yeeea!!!!!!\n" 
        //               << "xe.p: (" << xe.p.x << ", " << xe.p.y << ", " << xe.p.z << ")\n"
        //               << "xs.p: (" << xs.p.x << ", " << xs.p.y << ", " << xs.p.z << ")\n"
        //               << "xs.n: (" << xs.n.x << ", " << xs.n.y << ", " << xs.n.z << ")\n"
        //               << "left_importance: " << left_importance << "\n"
        //               << "right_importance: " << right_importance << "\n";
        //     exit(0);
        // }
        
        //     // Log(EWarn, "GeometryBVH::sampleGeometry: All node importances are zero at depth, cannot sample geometry.\n");
         
        //     // ^Nes: this log was likely slowing the rendering a lot when DEFENSIVE_PDF was not there, and right now this code is dead anyways, see note:

        //     // -------------- Nestor's notes: --------------
        //     // Note: this is impossible right now due to DEFENSIVE_PDF.
        //     // What happens now when both computeNodeImportance return 0 is a 50/50 chance of both children
        //     // Do we really want this? Some alternatives:
        //     // 1) Return false WITHOUT the log, handle this case robustly somehow 
        //     //    (could be hard, our computeNodeImportance should be very careful to avoid returning 0 for valid nodes!) 
        //     // 2) Sample this full node uniformly with whatever strategy (area/solidangle) as if it was a leaf. I think this would be nice
        //     // ---------------------------------------------

        //     return false;
        // }
        #ifdef ENABLE_LIGHTCUTS 
        if (total_importance < LIGHTCUT_THRESHOLD * 2.f * m_defensive_pdf) { // If importance is very low, treat this node as a leaf to avoid unnecessary traversal
            break;
        }   
        #endif

        // Compute branching probability
        const Float left_prob = left_importance / total_importance;

        // Probabilistic branch selection
        Float s1 = sampler->next1D();
        if (s1 < left_prob) {
            nodeIndex = leftIdx;
            pdf_traversal *= left_prob;
            // s1 = s1 / left_prob;  // Reuse random number
        } else {
            nodeIndex = rightIdx;
            pdf_traversal *= (1.0f - left_prob);
            // s1 = (s1 - left_prob) / (1.0f - left_prob);  // Reuse random number
        }
    }

    // 2. Sample leaf using configured sampling mode
    const BVHNode &leafNode = m_nodes[nodeIndex];
    
    if (leafNode.nLeafPrimitives == 0) {
        return false;
    }

    Float pdf_leaf = 0.0f;
    bool success = false;

    if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
        // Use spherical AABB sampling
        success = sampleLeafSphericalAABB(scene, leafNode, nodeIndex, sampler->next2D(), its_xs.p, its_xp, pdf_leaf);
        // success = sampleNodePrimitive(nodeIndex, scene, sampler->next2D(), its_xs.p, its_xp, pdf_leaf); // Sample primitives in the entire node (not just the leaf) using the same BVH sampling strategy. This is more expensive but should give better results when we have large leaf nodes.
    } else {
        // Use primitive sampling (default)
        success = sampleLeafPrimitive(scene, leafNode, sampler->next1D(), sampler->next2D(), its_xs.p, its_xp, pdf_leaf);
        // success = sampleNodePrimitive(nodeIndex, scene, sampler->next2D(), its_xs.p, its_xp, pdf_leaf); // Sample primitives in the entire node (not just the leaf) using the same BVH sampling strategy. This is more expensive but should give better results when we have large leaf nodes.
    }

    if (!success) return false;

    // Phase 3: Final PDF is in Solid Angle
    pdf = pdf_traversal * pdf_leaf;
    
    return true;
}

// Updated: Now accepts xs to perform Area->SolidAngle conversion
bool GeometryBVH::sampleLeafPrimitive(
    const Scene *scene,
    const BVHNode &leafNode,
    const Float &sample1,
    const Point2 &sample2,
    const Point &xs,
    Intersection &its_xp,
    Float &pdf_xp) 
{
    if (leafNode.nLeafPrimitives == 0) return false;

    const uint32_t primIdx = std::min(
        static_cast<uint32_t>(sample1 * leafNode.nLeafPrimitives),
        static_cast<uint32_t>(leafNode.nLeafPrimitives - 1)
    );
    
    const BVHPrimitive &prim = scene->getGeometryBVH()->getPrimitive(leafNode.primitivesOffset + primIdx);
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    const TriMesh *mesh = meshes[prim.meshIndex];
    const Triangle& tri = mesh->getTriangles()[prim.triangleIndex];

    PositionSamplingRecord pRec;
    pRec.p = tri.sample(mesh->getVertexPositions(), mesh->getVertexNormals(),
        mesh->getVertexTexcoords(), pRec.n, pRec.uv, sample2);

    const Float triangle_area = tri.surfaceArea(mesh->getVertexPositions());
    if (triangle_area <= 0.0f) return false;

    // 1. Basic Area PDF
    Float pdf_area = 1.0f / (leafNode.nLeafPrimitives * triangle_area);
    
    // Fill intersection
    its_xp.p = pRec.p;
    its_xp.uv = pRec.uv;
    its_xp.shape = mesh;
    its_xp.instance = mesh;
    its_xp.primIndex = prim.triangleIndex;
    
    const Point *verts = mesh->getVertexPositions();
    Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
    Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
    its_xp.geoFrame = Frame(normalize(cross(e1, e2)));
    its_xp.shFrame = Frame(pRec.n);

    // 2. Convert Area PDF -> Solid Angle PDF
    // P_sa = P_area * (dist^2 / cos_theta)
    Vector d = its_xp.p - xs;
    Float dist_sq = d.lengthSquared();
    Float dist = std::sqrt(dist_sq);
    Vector wo = d / dist;
    
    Float cos_theta = std::abs(dot(its_xp.geoFrame.n, -wo));

    if (cos_theta <= Epsilon) {
        // Grazing angle on uniform sampling: Probability of generating this direction is Infinite 
        // (technically). We can return 0 to reject or a huge number.
        // Returning 0 rejects the sample (safe).
        return false; 
    }

    pdf_xp = pdf_area * (dist_sq / cos_theta);
    return true;
}

bool GeometryBVH::sampleLeafSphericalAABB(
    const Scene *scene,
    const BVHNode &leafNode,
    size_t leafNodeIndex,
    const Point2 &sample2,
    const Point &xs,
    Intersection &its_xp,
    Float &pdf_xp)
{
    if (leafNode.nLeafPrimitives == 0) return false;

    Vector wo;
    Float pdf_aabb_solid_angle;

    // Naturally samples in Solid Angle
    AABBSphSample(leafNode.bounds, xs, sample2.x, sample2.y, wo, pdf_aabb_solid_angle);
    
    if (pdf_aabb_solid_angle <= 0.0f || !std::isfinite(pdf_aabb_solid_angle)) return false;

    Ray ray(xs, wo, Epsilon, std::numeric_limits<Float>::infinity(), 0.f);
    if (!scene->rayIntersect(ray, its_xp)) return false;

    if (!intersectionInLeafNode(scene, its_xp, leafNodeIndex)) return false;

    // FIX: Do NOT convert to Area. Keep it in Solid Angle.
    // This avoids the (cos / dist^2) multiplication which vanishes at grazing angles.
    pdf_xp = pdf_aabb_solid_angle; 
    
    return true;
}

// Samples uniformly in the primitives inside of the node and returns the pdf in Solid Angle
bool GeometryBVH::sampleNodePrimitive(
    const Scene *scene, 
    size_t nodeIdx,
    const Float &sample1,
    const Float &sample2,
    const Point2 &sample3,
    const Point &xs,
    Intersection &its_xp,
    Float &pdf_xp
) {
    float pdf_leaf = 1.f;
    float sample = sample1;
    while(!m_nodes[nodeIdx].isLeaf()) {
        const size_t leftIdx = nodeIdx + 1;
        const size_t rightIdx = m_nodes[nodeIdx].secondChildOffset;
        float left_area = m_nodeInfos[leftIdx].surfaceArea;
        float right_area = m_nodeInfos[rightIdx].surfaceArea;
        float left_prob = left_area / (left_area + right_area);
        // Uniformly sample one of the two children
        if (sample < left_prob) {
            // remap sample to be in [0,1] for the next iteration
            sample = sample / left_prob;
            nodeIdx = leftIdx;
            pdf_leaf *= left_prob; // Update pdf_leaf to account for the probability of choosing this child
        } else {
            sample = (sample - left_prob) / (1.f - left_prob);
            nodeIdx = rightIdx;
            pdf_leaf *= (1.0f - left_prob); // Update pdf_leaf to account for the probability of choosing this child
        }
    }

    bool success = sampleLeafPrimitive(scene, m_nodes[nodeIdx], sample2, sample3, xs, its_xp, pdf_xp);
    if (!success) return false;
    pdf_xp *= pdf_leaf;
    return true; 
}

bool GeometryBVH::intersectionInLeafNode(
    const Scene *scene,
    const Intersection &its,
    size_t leafNodeIndex)
{   
    // Check that the leafNodeIndex is valid
    if (!scene || leafNodeIndex >= m_nodes.size()) {
        return false;
    }

    // Check that the node is actually a leaf
    const BVHNode &leafNode = m_nodes[leafNodeIndex];
    if (!leafNode.isLeaf()) {
        return false;
    }

    // Check that the intersection is on a valid shape and that the shape is a TriMesh
    if (its.shape == NULL) {
        return false;
    }

    const TriMesh *hitMesh = dynamic_cast<const TriMesh*>(its.shape);
    if (hitMesh == NULL) {
        return false;
    }

    // Find the mesh index in the scene's mesh list
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    uint32_t meshIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < meshes.size(); ++i) {
        if (meshes[i] == hitMesh) {
            meshIndex = i;
            break;
        }
    }

    if (meshIndex == 0xFFFFFFFF) {
        return false;
    }

    // Check that the triangle index is valid (must be less than the number of triangles in the mesh)
    if (its.primIndex >= hitMesh->getTriangleCount()) {
        return false;
    }
    
    // Check that the triangle index corresponds to a primitive in this leaf node
    if (meshIndex >= m_meshOffsets.size()) {
        return false;
    }
    uint32_t lookupIdx = m_meshOffsets[meshIndex] + its.primIndex;
    if (lookupIdx >= m_triangleToPrim.size()) {
        return false;
    }
    uint32_t primIdx = m_triangleToPrim[lookupIdx];
    if (primIdx == 0xFFFFFFFF) {
        return false;
    }

    const uint32_t leafStart = leafNode.primitivesOffset;
    const uint32_t leafEnd = leafStart + leafNode.nLeafPrimitives;
    return primIdx >= leafStart && primIdx < leafEnd;
}


// Just like pdfGeometry, but finds the indices from the intersection directly
Float GeometryBVH::pdfGeometry(
    const Scene *scene,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    const Intersection &its_xp
) {
    if (its_xp.shape == NULL) {
        Log(EWarn, "pdfGeometry: Intersection has no shape, cannot compute PDF.\n");
        return 0.0f;
    }

    const TriMesh *hitMesh = dynamic_cast<const TriMesh*>(its_xp.shape);
    if (hitMesh == NULL) {
        Log(EWarn, "pdfGeometry: Intersection shape is not a TriMesh, cannot compute PDF.\n");
        return 0.0f;
    }

    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    uint32_t meshIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < meshes.size(); ++i) {
        if (meshes[i] == hitMesh) {
            meshIndex = i;
            break;
        }
    }

    if (meshIndex == 0xFFFFFFFF) {
        return 0.0f;
    }

    if (its_xp.primIndex >= hitMesh->getTriangleCount()) {
        return 0.0f;
    }
    
    // Now we have the mesh index and triangle index, we can call the original pdfGeometry
     return this->pdfGeometry(
        scene, 
        its_xs, 
        pRec_xe,
        its_xp,
        meshIndex, 
        its_xp.primIndex, 
        its_xp.p
    );  
}

Float GeometryBVH::pdfGeometry(
    const Scene *scene,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    const Intersection &its_xp,
    uint32_t meshIndex,
    uint32_t triangleIndex,
    const Point &position
) {
    if (m_nodes.empty() || m_primitives.empty()) return 0.0f;

    // 1. Find Primitive
    if (meshIndex >= m_meshOffsets.size()) return 0.0f;
    uint32_t lookupIdx = m_meshOffsets[meshIndex] + triangleIndex;
    if (lookupIdx >= m_triangleToPrim.size()) return 0.0f;
    uint32_t primIdx = m_triangleToPrim[lookupIdx];
    if (primIdx == 0xFFFFFFFF) return 0.0f;

    // 2. Traversal
    size_t nodeIndex = 0;
    uint32_t nodeStart = 0;
    Float pdf_traversal = 1.0f;

    while (!m_nodes[nodeIndex].isLeaf()) {
        size_t leftIdx = nodeIndex + 1;
        size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;
        const BVHNode &leftNode = m_nodes[leftIdx];
        const BVHNode &rightNode = m_nodes[rightIdx];

        const Float left_imp = computeNodeImportance(
            m_scene,
            this,
            leftIdx,
            its_xs,
            pRec_xe,
            m_nodeInfos[leftIdx],
            m_nodes[leftIdx].bounds,
            m_sggxCrossSectionScale, m_solidAngleScale, m_variancePenaltyScale, m_cosinePenaltyScale // pass importance config params
            ) * m_probmult + m_defensive_pdf;

        const Float right_imp = computeNodeImportance(
            m_scene,
            this,
            rightIdx,
            its_xs,
            pRec_xe,
            m_nodeInfos[rightIdx],
            m_nodes[rightIdx].bounds,
            m_sggxCrossSectionScale, m_solidAngleScale, m_variancePenaltyScale, m_cosinePenaltyScale
        ) * m_probmult + m_defensive_pdf;

        const Float total = left_imp + right_imp;
        if (total <= 0.0f) return 0.0f;

        uint32_t leftCount = leftNode.nSubtreePrimitives;
        bool inLeft = (primIdx < nodeStart + leftCount);

        if (inLeft) {
            pdf_traversal *= (left_imp / total);
            nodeIndex = leftIdx;
        } else {
            pdf_traversal *= (right_imp / total);
            nodeIndex = rightIdx;
            nodeStart += leftCount;
        }
    }

    // 3. Compute leaf-level PDF in SOLID ANGLE
    const BVHNode &leaf = m_nodes[nodeIndex];
    Float pdf_leaf_sa = 0.0f;

    if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
        Vector wo = position - its_xs.p;
        Float dist_sq = wo.lengthSquared();
        Float dist = std::sqrt(dist_sq);
        if (dist <= Epsilon) return 0.0f;
        wo /= dist;

        Float pdf_solid_angle = 0.0f;
        AABBSphPdf(leaf.bounds, its_xs.p, wo, pdf_solid_angle);
        
        // FIX: Return Solid Angle directly. Do not convert to Area.
        pdf_leaf_sa = pdf_solid_angle;

    } else {
        // Primitive (Area) -> Convert to Solid Angle
        const std::vector<TriMesh*> &meshes = scene->getMeshes();
        const TriMesh *mesh = meshes[meshIndex];
        const Triangle &tri = mesh->getTriangles()[triangleIndex];
        Float area = tri.surfaceArea(mesh->getVertexPositions());
        
        if (area <= 0.0f || leaf.nLeafPrimitives == 0) return 0.0f;
        Float pdf_area = 1.0f / (leaf.nLeafPrimitives * area);

        // Convert Area -> Solid Angle
        // Need geometric normal for conversion
        const Point *verts = mesh->getVertexPositions();
        Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
        Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
        Normal n = normalize(cross(e1, e2));
        
        Vector d = position - its_xs.p;
        Float dist_sq = d.lengthSquared();
        Vector wo = normalize(d);
        Float cos_theta = std::abs(dot(n, -wo));

        if (cos_theta <= Epsilon) return 0.0f; // Singularity
        
        pdf_leaf_sa = pdf_area * (dist_sq / cos_theta);
    }

    return pdf_traversal * pdf_leaf_sa;
}

Spectrum evalLength2Contribution(
    const Scene *scene,
    const Intersection &its_xs,
    const Intersection &its_xp,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility
) {
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
    BSDFSamplingRecord bRec_xs(its_xs, its_xs.toLocal(d_sp));
    Spectrum f_xs = its_xs.getBSDF()->eval(bRec_xs);
    // 2.1.2. Evaluate BSDF at xp for the direction towards xe
    BSDFSamplingRecord bRec_xp(its_xp, its_xp.toLocal(-d_sp), its_xp.toLocal(d_pe));
    Spectrum f_xp = its_xp.getBSDF()->eval(bRec_xp);

    if (f_xs.isZero() || f_xp.isZero()) {
        return Spectrum(0.0f);
    }

    // 2.2. Evaluate emitter radiance at xe in the direction of xp
    const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
    DirectionSamplingRecord dRec_xe(-d_pe);
    dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
    Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);
    
    if (Le.isZero()) {
        return Spectrum(0.0f);
    }

    // 2.3. Geometry term
    Float cos_theta_xs_out = std::abs(dot(its_xs.geoFrame.n, d_sp));
    Float cos_theta_xp_in = std::abs(dot(its_xp.geoFrame.n, -d_sp));
    Float cos_theta_xp_out = std::abs(dot(its_xp.geoFrame.n, d_pe));
    Float cos_theta_xe_in = std::abs(dot(pRec_xe.n, -d_pe));

    Float G_sp = (cos_theta_xp_in) / (dist_sp * dist_sp);
    Float G_pe = (1.f) / (dist_pe * dist_pe);
    // TODO(jorge): revise where geometry terms have to be added

    return (f_xs * G_sp * f_xp * G_pe * Le);
}

MTS_NAMESPACE_END
