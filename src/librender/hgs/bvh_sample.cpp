/*
    This file is part of Mitsuba, a physically based rendering system.

    BVH Geometry Sampling - Importance-driven geometry sampling via BVH traversal
*/

#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/render/hgs/spherical_aabb.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/trimesh.h>
#include <cmath>
#include <iostream>

// #define DEBUG_BVH_NAN



MTS_NAMESPACE_BEGIN


#ifdef DEBUG_BVH_NAN
static inline bool bvhInvalidFloat(Float v) {
    return !std::isfinite(v) || std::isnan(v);
}

#define BVH_NAN_LOG(msg) do { std::cerr << "[DEBUG_BVH_NAN] " << msg << std::endl; } while (0)
#endif

// const Float DEFENSIVE_PDF = 1e-2f; // Minimum PDF to avoid 0 prob
// ^Nes: if we do MIS, we have to account for this defensive pdf when computing the pdf of a node
//       this will be hard according to Adrian, he recommends to skip MIS for now
// TO DO: also, we should compare the magnitude of this value to the typical node importance values 
//        we get from computeNodeImportance, to make sure it is not dominating the sampling


// LIGHTCUTS: now these are runtime parameters that can be set from the xml, see m_enableLightcuts and m_lightcutThreshold in bvh.h and bvh_build.cpp
// When enabled, directly samples nodes as if they were leafs when their importance is very low

// #define DEBUG_BVH_SAMPLING 

#ifdef DEBUG_BVH_SAMPLING


#include <mutex>
#include <map>

struct Stats {
    std::map<int, Float> nodeImportanceLeft, nodeImportanceRight; // Sum of importance values for each level
    std::map<int, uint32_t> nodeImportanceCounts; // Count of samples for each level

    std::mutex mutex; // Mutex for thread safety

    void logImportance(int level, Float importanceL, Float importanceR) {
        std::lock_guard<std::mutex> lock(mutex);
        nodeImportanceLeft[level] += importanceL;
        nodeImportanceRight[level] += importanceR;
        nodeImportanceCounts[level] += 1;
    }

    void printAverages() {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << "Average node importance by level:\n";
        for (const auto &entry : nodeImportanceCounts) {
            int level = entry.first;
            float count = float(entry.second);
            Float avgL = nodeImportanceLeft[level] / count;
            Float avgR = nodeImportanceRight[level] / count;
            std::cout << "Level " << level << ": Left Avg Importance = " << avgL 
                      << ", Right Avg Importance = " << avgR 
                      << ", Sample Count = " << entry.second << "\n";
        }
    }
};

static Stats g_stats;

#endif

bool GeometryBVH::sampleGeometry(
    const Scene *scene,
    Sampler *sampler,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    Intersection &its_xp,
    Float &pdf) const
{
    if (m_nodes.empty() || m_primitives.empty()) {
#ifdef DEBUG_BVH_NAN
        BVH_NAN_LOG("sampleGeometry: empty nodes/primitives. nodes=" << m_nodes.size()
            << " primitives=" << m_primitives.size());
#endif
        return false;
    }

    // 1. Importance-driven tree traversal
    size_t nodeIndex = 0;
    Float pdf_traversal = 1.0f;

    #ifdef DO_TIMING_SAMPLING
    auto &importanceTimer = getImportanceTimer();
    uint64_t startTime = importanceTimer.getNanoSeconds();
    importanceTimer.addCounter(); // count the number of times we entered sampleGeometry, useful for averaging timings
    #endif 

    while (!m_nodes[nodeIndex].isLeaf()) {
        const size_t leftIdx = nodeIndex + 1;
        const size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;

        // 2. Compute importance weights for both children
        const Float left_importance_temp = computeNodeImportance(m_scene, this, leftIdx, its_xs, pRec_xe);
        const Float left_importance = left_importance_temp * m_probmult + m_defensive_pdf;

        const Float right_importance_temp = computeNodeImportance(m_scene, this, rightIdx, its_xs, pRec_xe);
        const Float right_importance = right_importance_temp * m_probmult + m_defensive_pdf;

        const Float total_importance = left_importance + right_importance;

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(left_importance_temp) || bvhInvalidFloat(right_importance_temp)
            || bvhInvalidFloat(left_importance) || bvhInvalidFloat(right_importance)
            || bvhInvalidFloat(total_importance) || total_importance <= 0.0f) {
            BVH_NAN_LOG("sampleGeometry traversal invalid importance at node=" << nodeIndex
                << " leftIdx=" << leftIdx
                << " rightIdx=" << rightIdx
                << " left_importance_temp=" << left_importance_temp
                << " right_importance_temp=" << right_importance_temp
                << " left_importance=" << left_importance
                << " right_importance=" << right_importance
                << " total_importance=" << total_importance
                << " xs=" << its_xs.p.toString()
                << " xe=" << pRec_xe.p.toString()
                << " defensive_pdf=" << m_defensive_pdf
                << " probmult=" << m_probmult);
            return false;
        }
#endif

        // --- LIGHTCUTS ---
        if (m_enableLightcuts && total_importance < m_lightcutThreshold * 2.f * m_defensive_pdf) { 
            #ifdef DO_TIMING_SAMPLING
            importanceTimer.addNamedCounter("lightcuts");
            #endif
            break; 
        }   

        // Compute branching probability
        const Float left_prob = left_importance / total_importance;

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(left_prob) || left_prob < 0.0f || left_prob > 1.0f) {
            BVH_NAN_LOG("sampleGeometry traversal invalid left_prob at node=" << nodeIndex
                << " left_prob=" << left_prob
                << " left_importance=" << left_importance
                << " right_importance=" << right_importance
                << " total_importance=" << total_importance
                << " xs=" << its_xs.p.toString()
                << " xe=" << pRec_xe.p.toString());
            return false;
        }
#endif

        // Probabilistic branch selection
        Float s1 = sampler->next1D();
        if (s1 < left_prob) {
            nodeIndex = leftIdx;
            pdf_traversal *= left_prob;
        } else {
            nodeIndex = rightIdx;
            pdf_traversal *= (1.0f - left_prob);
        }

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(pdf_traversal) || pdf_traversal <= 0.0f) {
            BVH_NAN_LOG("sampleGeometry traversal invalid pdf_traversal after branch"
                << " node=" << nodeIndex
                << " left_prob=" << left_prob
                << " s1=" << s1
                << " pdf_traversal=" << pdf_traversal
                << " xs=" << its_xs.p.toString()
                << " xe=" << pRec_xe.p.toString());
            return false;
        }
#endif
    }

    #ifdef DO_TIMING_SAMPLING
    importanceTimer.addElapsed("sampleNodeLoop", startTime); 
    startTime = importanceTimer.getNanoSeconds();
    #endif

    // 2. Sample leaf or subtree using configured sampling mode
    const BVHNode &node = m_nodes[nodeIndex];
    Float pdf_leaf = 0.0f;
    bool success = false;

    if (node.isLeaf()) {
        if (node.nLeafPrimitives == 0) return false;

        if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
            success = sampleLeafSphericalAABB(scene, node, nodeIndex, sampler->next2D(), its_xs.p, its_xp, pdf_leaf);
        } else {
            success = sampleLeafPrimitive(scene, node, nodeIndex, sampler->next1D(), sampler->next2D(), its_xs.p, its_xp, pdf_leaf);
        }
    } // the node is not a leaf (because of pruning/lightcuts)
    else if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB_anyNode) {
        // This strategy can be used directly on non-leaf nodes
        success = sampleLeafSphericalAABB(scene, node, nodeIndex, sampler->next2D(), its_xs.p, its_xp, pdf_leaf);
    }   
    else { // this descends to a leaf by area, then samples the leaf either by Primitive or SphericalAABB, depending on the sampling mode
        // --- LIGHTCUTS FIX: We stopped at a non-leaf node. Sample the subtree. ---
        // sampleNodePrimitive requires 3 random samples (Float, Float, Point2)
        success = sampleNodePrimitive(scene, nodeIndex, sampler, its_xs.p, its_xp, pdf_leaf, true);
    }
    
    #ifdef DO_TIMING_SAMPLING
    if (success) {
        importanceTimer.addElapsed("sampleLeaf", startTime);
        importanceTimer.addNamedCounter("successfulSamples");
    }
    else {
        importanceTimer.addElapsed("sampleLeafFail", startTime);
        importanceTimer.addNamedCounter("failedSamples");
    }

    if (importanceTimer.timer->getSeconds() > 10.f) {
        std::cout << "Sampling timings:\n";
        importanceTimer.report();
        exit(0); 
    }

    #endif

    if (!success) return false;

    // Phase 3: Final PDF is in Solid Angle
    pdf = pdf_traversal * pdf_leaf;

#ifdef DEBUG_BVH_NAN
    if (bvhInvalidFloat(pdf_leaf) || bvhInvalidFloat(pdf) || pdf <= 0.0f) {
        BVH_NAN_LOG("sampleGeometry invalid final pdf"
            << " nodeIndex=" << nodeIndex
            << " success=" << success
            << " pdf_traversal=" << pdf_traversal
            << " pdf_leaf=" << pdf_leaf
            << " pdf=" << pdf
            << " samplingMode=" << (int) m_samplingMode
            << " xs=" << its_xs.p.toString()
            << " xe=" << pRec_xe.p.toString()
            << " xp=" << its_xp.p.toString());
        return false;
    }
#endif
    
    return true;
}
// bool GeometryBVH::sampleGeometry(
//     const Scene *scene,
//         Sampler *sampler,
//         const Intersection &its_xs,
//         const PositionSamplingRecord &pRec_xe,
//         Intersection &its_xp,
//         Float &pdf)
// {
//     // std::cout << "Starting BVH sampling...\n";
//     if (m_nodes.empty() || m_primitives.empty()) {
//         return false;
//     }

//     // 1. Importance-driven tree traversal
//     size_t nodeIndex = 0;
//     Float pdf_traversal = 1.0f;

    
    
//     while (!m_nodes[nodeIndex].isLeaf()) {
//         const size_t leftIdx = nodeIndex + 1;
//         const size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;

//         // 2. Compute importance weights for both children
//         const Float left_importance_temp = computeNodeImportance(
//             m_scene,
//             this,
//             leftIdx,
//             its_xs,
//             pRec_xe
//         );
//         const Float left_importance = left_importance_temp * m_probmult + m_defensive_pdf;

//         /*
//         std::cout << "Left importance (before probmult and defensive): " << left_importance_temp << " at node " << leftIdx << "\n";
//         if (left_importance_temp > 0.001f) {
//             const Float left_imp_gt = computeNodeImportanceGroundTruth(
//                 m_scene,
//                 this,
//                 leftIdx,
//                 its_xs,
//                 pRec_xe,
//                 1000,
//                 false,
//                 m_nodeInfos[leftIdx]
//             );
//             std::cout << "Left importance: " << left_importance_temp << " at node " << leftIdx << "\n";
//             std::cout << "Left importance (ground truth): " << left_imp_gt << " at node " << leftIdx << "\n";
//             // Debug: print points, normals and local directions to help diagnose zero BSDF
//             std::cout << "DEBUG its_xs.p: (" << its_xs.p.x << ", " << its_xs.p.y << ", " << its_xs.p.z << ")\n";
//             std::cout << "DEBUG its_xs.geo.n: (" << its_xs.geoFrame.n.x << ", " << its_xs.geoFrame.n.y << ", " << its_xs.geoFrame.n.z << ")\n";
//             std::cout << "DEBUG its_xp.p: (" << its_xp.p.x << ", " << its_xp.p.y << ", " << its_xp.p.z << ")\n";
//             std::cout << "DEBUG its_xp.geo.n: (" << its_xp.geoFrame.n.x << ", " << its_xp.geoFrame.n.y << ", " << its_xp.geoFrame.n.z << ")\n";
//             exit(0);
//         }
//         */
        
//         const Float right_importance_temp = computeNodeImportance(
//             m_scene,
//             this,
//             rightIdx,
//             its_xs,
//             pRec_xe
//         );
//         const Float right_importance = right_importance_temp * m_probmult + m_defensive_pdf;

//         const Float total_importance = left_importance + right_importance;


// #ifdef DEBUG_BVH_SAMPLING
//         // g_stats.logImportance(bvh.getLevel(nodeIndex), left_importance_temp, right_importance_temp);
//         int level = getLevel(nodeIndex);

//         if (level <= 2 && (left_importance_temp > 0.001f || right_importance_temp > 0.001f)) { // Only log when importance is significant to avoid noise, and only for the top levels to avoid too much data
//             // g_stats.logImportance(level, left_importance_temp, right_importance_temp);
//             std::cout << "Node " << nodeIndex 
//                 << "\nxs: " <<  its_xs.p.toString() << ", xe: " << pRec_xe.p.toString() << "\n"
//                 << ": Left importance = " << left_importance_temp 
//                 << ", Right importance = " << right_importance_temp 
//                 << ", Total importance = " << total_importance 
//                 << ", Level = " << getLevel(nodeIndex) << "\n";

//         }
// #endif

//         // // Error handling: if both importances are zero, we cannot continue
//         // if (total_importance > 2.301f * m_defensive_pdf) {
//         //     std::cout << "Yeeea!!!!!!\n" 
//         //               << "xe.p: (" << xe.p.x << ", " << xe.p.y << ", " << xe.p.z << ")\n"
//         //               << "xs.p: (" << xs.p.x << ", " << xs.p.y << ", " << xs.p.z << ")\n"
//         //               << "xs.n: (" << xs.n.x << ", " << xs.n.y << ", " << xs.n.z << ")\n"
//         //               << "left_importance: " << left_importance << "\n"
//         //               << "right_importance: " << right_importance << "\n";
//         //     exit(0);
//         // }
        
//         //     // Log(EWarn, "GeometryBVH::sampleGeometry: All node importances are zero at depth, cannot sample geometry.\n");
         
//         //     // ^Nes: this log was likely slowing the rendering a lot when DEFENSIVE_PDF was not there, and right now this code is dead anyways, see note:

//         //     // -------------- Nestor's notes: --------------
//         //     // Note: this is impossible right now due to DEFENSIVE_PDF.
//         //     // What happens now when both computeNodeImportance return 0 is a 50/50 chance of both children
//         //     // Do we really want this? Some alternatives:
//         //     // 1) Return false WITHOUT the log, handle this case robustly somehow 
//         //     //    (could be hard, our computeNodeImportance should be very careful to avoid returning 0 for valid nodes!) 
//         //     // 2) Sample this full node uniformly with whatever strategy (area/solidangle) as if it was a leaf. I think this would be nice
//         //     // ---------------------------------------------

//         //     return false;
//         // }
//         if (m_enableLightcuts && total_importance < m_lightcutThreshold * 2.f * m_defensive_pdf) { // If importance is very low, treat this node as a leaf to avoid unnecessary traversal
//             Log(EError, " you need to fix the sampleNodePrimitive below instead of calling sampleLeafPrimitive!!\n");
//             exit(1);
//             break;
//         }   
        

//         // Compute branching probability
//         const Float left_prob = left_importance / total_importance;

//         // Probabilistic branch selection
//         Float s1 = sampler->next1D();
//         if (s1 < left_prob) {
//             nodeIndex = leftIdx;
//             pdf_traversal *= left_prob;
//             // s1 = s1 / left_prob;  // Reuse random number
//         } else {
//             nodeIndex = rightIdx;
//             pdf_traversal *= (1.0f - left_prob);
//             // s1 = (s1 - left_prob) / (1.0f - left_prob);  // Reuse random number
//         }
//     }

//     // 2. Sample leaf using configured sampling mode
//     const BVHNode &leafNode = m_nodes[nodeIndex];
    
//     if (leafNode.nLeafPrimitives == 0) {
//         return false;
//     }

//     Float pdf_leaf = 0.0f;
//     bool success = false;

//     if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
//         // Use spherical AABB sampling
//         success = sampleLeafSphericalAABB(scene, leafNode, nodeIndex, sampler->next2D(), its_xs.p, its_xp, pdf_leaf);
//         // success = sampleNodePrimitive(nodeIndex, scene, sampler->next2D(), its_xs.p, its_xp, pdf_leaf); // Sample primitives in the entire node (not just the leaf) using the same BVH sampling strategy. This is more expensive but should give better results when we have large leaf nodes.
//     } else {
//         // Use primitive sampling (default)
//         success = sampleLeafPrimitive(scene, leafNode, sampler->next1D(), sampler->next2D(), its_xs.p, its_xp, pdf_leaf);
//         // success = sampleNodePrimitive(nodeIndex, scene, sampler->next2D(), its_xs.p, its_xp, pdf_leaf); // Sample primitives in the entire node (not just the leaf) using the same BVH sampling strategy. This is more expensive but should give better results when we have large leaf nodes.
//     }

//     if (!success) return false;

//     // Phase 3: Final PDF is in Solid Angle
//     pdf = pdf_traversal * pdf_leaf;
    
//     return true;
// }

// Updated: Samples primitives proportionally by area, then converts Area->SolidAngle
bool GeometryBVH::sampleLeafPrimitive(
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
            const BVHPrimitive &p = scene->getGeometryBVH()->getPrimitive(leafNode.primitivesOffset + i);
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
            const BVHPrimitive &p = scene->getGeometryBVH()->getPrimitive(leafNode.primitivesOffset + i);
            const TriMesh *m = scene->getMeshes()[p.meshIndex];
            const Triangle &t = m->getTriangles()[p.triangleIndex];

            accumulated_area += t.surfaceArea(m->getVertexPositions());
            if (accumulated_area >= target_area) {
                localPrimIdx = i;
                break;
            }
        }
    }
    
    // 3. Fetch the specifically chosen primitive
    const BVHPrimitive &prim = scene->getGeometryBVH()->getPrimitive(leafNode.primitivesOffset + localPrimIdx);
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    const TriMesh *mesh = meshes[prim.meshIndex];
    const Triangle& tri = mesh->getTriangles()[prim.triangleIndex];

    PositionSamplingRecord pRec;
    pRec.p = tri.sample(mesh->getVertexPositions(), mesh->getVertexNormals(),
        mesh->getVertexTexcoords(), pRec.n, pRec.uv, sample2);

    // 4. The local Area PDF inside this leaf is 1 / area(leaf)
    const Float total_leaf_area = (Float) m_nodeInfos[leafNodeIndex].surfaceArea;
    if (total_leaf_area <= 0.0f) return false;
    Float pdf_area = 1.0f / total_leaf_area;
    
    // Fill intersection
    its_xp.p = pRec.p;
    its_xp.uv = pRec.uv;
    its_xp.shape = mesh;
    its_xp.instance = mesh;
    its_xp.primIndex = prim.triangleIndex;
    its_xp.t = 1.f;
    
    const Point *verts = mesh->getVertexPositions();
    Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
    Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
    its_xp.geoFrame = Frame(normalize(cross(e1, e2)));
    its_xp.shFrame = Frame(pRec.n);
    its_xp.wi = its_xp.p - xs;

    if (return_solidangle_pdf) {
        // 5. Convert Area PDF -> Solid Angle PDF
        // P_sa = P_area * (dist^2 / cos_theta)
        Vector d = its_xp.p - xs;
        Float dist_sq = d.lengthSquared();

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(dist_sq) || dist_sq <= Epsilon * Epsilon) {
            BVH_NAN_LOG("sampleLeafPrimitive invalid distance for area->solid-angle conversion"
                << " dist_sq=" << dist_sq
                << " pdf_area=" << pdf_area
                << " xs=" << xs.toString()
                << " xp=" << its_xp.p.toString()
                << " leafNodeIndex=" << leafNodeIndex
                << " localPrimIdx=" << localPrimIdx);
            return false;
        }
#endif

        Float dist = std::sqrt(dist_sq);
        Vector wo = d / dist;
        
        Float cos_theta = std::abs(dot(its_xp.geoFrame.n, -wo));

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(cos_theta)) {
            BVH_NAN_LOG("sampleLeafPrimitive invalid cos_theta"
                << " cos_theta=" << cos_theta
                << " dist=" << dist
                << " dist_sq=" << dist_sq
                << " pdf_area=" << pdf_area
                << " xs=" << xs.toString()
                << " xp=" << its_xp.p.toString()
                << " n=" << its_xp.geoFrame.n.toString());
            return false;
        }
#endif

        if (cos_theta <= Epsilon) {
            return false; 
        }

        pdf_xp = pdf_area * (dist_sq / cos_theta);

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(pdf_xp) || pdf_xp <= 0.0f) {
            BVH_NAN_LOG("sampleLeafPrimitive invalid solid-angle pdf"
                << " pdf_xp=" << pdf_xp
                << " pdf_area=" << pdf_area
                << " dist_sq=" << dist_sq
                << " cos_theta=" << cos_theta
                << " xs=" << xs.toString()
                << " xp=" << its_xp.p.toString());
            return false;
        }
#endif
    } else {
        pdf_xp = pdf_area;
    }
    
    return true;
}

// Samples uniformly in the primitives inside of the node and returns the pdf in Solid Angle
bool GeometryBVH::sampleNodePrimitive(
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

#ifdef DEBUG_BVH_NAN
        if (!std::isfinite(left_area) || !std::isfinite(right_area) || (left_area + right_area) <= 0.0f) {
            BVH_NAN_LOG("sampleNodePrimitive invalid child areas"
                << " nodeIdx=" << nodeIdx
                << " left_area=" << left_area
                << " right_area=" << right_area
                << " xs=" << xs.toString());
            return false;
        }
#endif

        float left_prob = left_area / (left_area + right_area);

#ifdef DEBUG_BVH_NAN
        if (!std::isfinite(left_prob) || left_prob < 0.0f || left_prob > 1.0f) {
            BVH_NAN_LOG("sampleNodePrimitive invalid left_prob"
                << " nodeIdx=" << nodeIdx
                << " left_prob=" << left_prob
                << " left_area=" << left_area
                << " right_area=" << right_area
                << " xs=" << xs.toString());
            return false;
        }
#endif

        // Uniformly sample one of the two children
        if (sampler->next1D() < left_prob) {
            nodeIdx = leftIdx;
            pdf_leaf *= left_prob; // Update pdf_leaf to account for the probability of choosing this child
        } else {
            nodeIdx = rightIdx;
            pdf_leaf *= (1.0f - left_prob); // Update pdf_leaf to account for the probability of choosing this child
        }

#ifdef DEBUG_BVH_NAN
        if (!std::isfinite(pdf_leaf) || pdf_leaf <= 0.0f) {
            BVH_NAN_LOG("sampleNodePrimitive invalid subtree pdf"
                << " nodeIdx=" << nodeIdx
                << " pdf_leaf=" << pdf_leaf
                << " left_prob=" << left_prob
                << " xs=" << xs.toString());
            return false;
        }
#endif
    }
    bool success;
    if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
        success = sampleLeafSphericalAABB(scene, m_nodes[nodeIdx], nodeIdx, sampler->next2D(), xs, its_xp, pdf_xp);
    } else {
        success = sampleLeafPrimitive(scene, m_nodes[nodeIdx], nodeIdx, sampler->next1D(), sampler->next2D(), xs, its_xp, pdf_xp, return_solidangle_pdf);
    }
    pdf_xp *= pdf_leaf;

#ifdef DEBUG_BVH_NAN
    if (!success || bvhInvalidFloat(pdf_xp) || pdf_xp <= 0.0f) {
        BVH_NAN_LOG("sampleNodePrimitive invalid output"
            << " success=" << success
            << " nodeIdx=" << nodeIdx
            << " pdf_leaf=" << pdf_leaf
            << " pdf_xp=" << pdf_xp
            << " xs=" << xs.toString()
            << " xp=" << its_xp.p.toString());
        return false;
    }
#endif

    return success; 
}

bool GeometryBVH::intersectionInNode(
    const Scene *scene,
    const Intersection &its,
    size_t targetNodeIndex) const
{
    if (!scene || targetNodeIndex >= m_nodes.size()) {
        return false;
    }

    if (its.shape == NULL) {
        return false;
    }

    const TriMesh *hitMesh = dynamic_cast<const TriMesh*>(its.shape);
    if (hitMesh == NULL) {
        return false;
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
        return false;
    }

    if (its.primIndex >= hitMesh->getTriangleCount()) {
        return false;
    }
    
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

    // 2. Fast top-down traversal to check if targetNodeIndex is an ancestor
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
        
        // Use the same partition logic from pdfGeometry
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

bool GeometryBVH::sampleLeafSphericalAABB(
    const Scene *scene,
    const BVHNode &leafNode,
    size_t leafNodeIndex,
    const Point2 &sample2,
    const Point &xs,
    Intersection &its_xp,
    Float &pdf_xp) const
{
    if (m_samplingMode != GeometryBVHSamplingMode::SphericalAABB_anyNode && leafNode.nLeafPrimitives == 0) return false;

    Vector wo;
    Float pdf_aabb_solid_angle;

    // Naturally samples in Solid Angle
    AABBSphSample(leafNode.bounds, xs, sample2.x, sample2.y, wo, pdf_aabb_solid_angle);

    // std::cout << "Sampled direction: (" << wo.x << ", " << wo.y << ", " << wo.z << ") with PDF (solid angle): " << pdf_aabb_solid_angle << "\n";
    
    if (pdf_aabb_solid_angle <= 0.0f || !std::isfinite(pdf_aabb_solid_angle)) return false;

    Ray ray(xs, wo, Epsilon, std::numeric_limits<Float>::infinity(), 0.f);
    if (!scene->rayIntersect(ray, its_xp)) return false;

    if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB_anyNode) {
        // In this mode, the "leafNode" may not necessarily be a leaf!
        if (!intersectionInNode(scene, its_xp, leafNodeIndex)) return false;
    }
    else{    
        if (!intersectionInLeafNode(scene, its_xp, leafNodeIndex)) return false;
    }
    // FIX: Do NOT convert to Area. Keep it in Solid Angle.
    // This avoids the (cos / dist^2) multiplication which vanishes at grazing angles.
    pdf_xp = pdf_aabb_solid_angle; 
    
    return true;
}


bool GeometryBVH::intersectionInLeafNode(
    const Scene *scene,
    const Intersection &its,
    size_t leafNodeIndex) const
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
) const {
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

// Float GeometryBVH::pdfGeometry(
//     const Scene *scene,
//     const Intersection &its_xs,
//     const PositionSamplingRecord &pRec_xe,
//     const Intersection &its_xp,
//     uint32_t meshIndex,
//     uint32_t triangleIndex,
//     const Point &position
// ) {
//     if (m_nodes.empty() || m_primitives.empty()) return 0.0f;

//     // 1. Find Primitive
//     if (meshIndex >= m_meshOffsets.size()) return 0.0f;
//     uint32_t lookupIdx = m_meshOffsets[meshIndex] + triangleIndex;
//     if (lookupIdx >= m_triangleToPrim.size()) return 0.0f;
//     uint32_t primIdx = m_triangleToPrim[lookupIdx];
//     if (primIdx == 0xFFFFFFFF) return 0.0f;

//     // 2. Traversal
//     size_t nodeIndex = 0;
//     uint32_t nodeStart = 0;
//     Float pdf_traversal = 1.0f;

//     while (!m_nodes[nodeIndex].isLeaf()) {
//         size_t leftIdx = nodeIndex + 1;
//         size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;
//         const BVHNode &leftNode = m_nodes[leftIdx];
//         const BVHNode &rightNode = m_nodes[rightIdx];

//         const Float left_imp = computeNodeImportance(
//             m_scene,
//             this,
//             leftIdx,
//             its_xs,
//             pRec_xe
//         ) * m_probmult + m_defensive_pdf;

//         const Float right_imp = computeNodeImportance(
//             m_scene,
//             this,
//             rightIdx,
//             its_xs,
//             pRec_xe
//         ) * m_probmult + m_defensive_pdf;

//         const Float total = left_imp + right_imp;
//         if (total <= 0.0f) return 0.0f;

//         // #ifdef ENABLE_LIGHTCUTS 
//         if (m_enableLightcuts && total < m_lightcutThreshold * 2.f * m_defensive_pdf) { // If importance is very low, treat this node as a leaf to avoid unnecessary traversal
//             // here, lightcuts were activated in the sample function, so we should treat this node as a leaf and compute the pdf accordingly
//             break;
//         }  
//         // #endif

//         uint32_t leftCount = leftNode.nSubtreePrimitives;
//         bool inLeft = (primIdx < nodeStart + leftCount);

//         if (inLeft) {
//             pdf_traversal *= (left_imp / total);
//             nodeIndex = leftIdx;
//         } else {
//             pdf_traversal *= (right_imp / total);
//             nodeIndex = rightIdx;
//             nodeStart += leftCount;
//         }
//     }

//     // 3. Compute leaf-level PDF in SOLID ANGLE
//     const BVHNode &leaf = m_nodes[nodeIndex];
//     const BVHNodeInfo &nodeInfo = m_nodeInfos[nodeIndex];

//     Float pdf_leaf_sa = 0.0f;

//     if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
//         Vector wo = position - its_xs.p;
//         Float dist_sq = wo.lengthSquared();
//         Float dist = std::sqrt(dist_sq);
//         if (dist <= Epsilon) return 0.0f;
//         wo /= dist;

//         Float pdf_solid_angle = 0.0f;
//         AABBSphPdf(leaf.bounds, its_xs.p, wo, pdf_solid_angle);
        
//         // FIX: Return Solid Angle directly. Do not convert to Area.
//         pdf_leaf_sa = pdf_solid_angle;

//     } else {
//         // Primitive (Area) -> Convert to Solid Angle
//         const std::vector<TriMesh*> &meshes = scene->getMeshes();
//         const TriMesh *mesh = meshes[meshIndex];
//         const Triangle &tri = mesh->getTriangles()[triangleIndex];
//         Float area = tri.surfaceArea(mesh->getVertexPositions());
        
//         if (area <= 0.0f || leaf.nLeafPrimitives == 0) return 0.0f;
//         // Float pdf_area = 1.0f / (leaf.nLeafPrimitives * area);
//         // ^ Nestor: WARNING: this is only correct if all primitives in the leaf have the same area, which is not necessarily true!
//         Float pdf_area = 1.0f / nodeInfo.surfaceArea; // Area of the whole leaf.

//         // Convert Area -> Solid Angle
//         // Need geometric normal for conversion
//         const Point *verts = mesh->getVertexPositions();
//         Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
//         Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
//         Normal n = normalize(cross(e1, e2));
        
//         Vector d = position - its_xs.p;
//         Float dist_sq = d.lengthSquared();
//         Vector wo = normalize(d);
//         Float cos_theta = std::abs(dot(n, -wo));

//         if (cos_theta <= Epsilon) return 0.0f; // Singularity
        
//         pdf_leaf_sa = pdf_area * (dist_sq / cos_theta);
//     }

//     return pdf_traversal * pdf_leaf_sa;
// }

Float GeometryBVH::pdfGeometry(
    const Scene *scene,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    const Intersection &its_xp,
    uint32_t meshIndex,
    uint32_t triangleIndex,
    const Point &position
) const 
{
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

        const Float left_imp = computeNodeImportance(m_scene, this, leftIdx, its_xs, pRec_xe) * m_probmult + m_defensive_pdf;
        const Float right_imp = computeNodeImportance(m_scene, this, rightIdx, its_xs, pRec_xe) * m_probmult + m_defensive_pdf;

        const Float total = left_imp + right_imp;

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(left_imp) || bvhInvalidFloat(right_imp) || bvhInvalidFloat(total) || total <= 0.0f) {
            BVH_NAN_LOG("pdfGeometry invalid traversal importances"
                << " nodeIndex=" << nodeIndex
                << " leftIdx=" << leftIdx
                << " rightIdx=" << rightIdx
                << " left_imp=" << left_imp
                << " right_imp=" << right_imp
                << " total=" << total
                << " xs=" << its_xs.p.toString()
                << " xe=" << pRec_xe.p.toString()
                << " xp=" << position.toString()
                << " meshIndex=" << meshIndex
                << " triangleIndex=" << triangleIndex);
            return 0.0f;
        }
#endif

        if (total <= 0.0f) return 0.0f;

        // --- LIGHTCUTS FIX: Break early during PDF evaluation ---
        if (m_enableLightcuts && total < m_lightcutThreshold * 2.f * m_defensive_pdf) { 
            break;
        }  

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

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(pdf_traversal) || pdf_traversal <= 0.0f) {
            BVH_NAN_LOG("pdfGeometry invalid traversal pdf"
                << " nodeIndex=" << nodeIndex
                << " pdf_traversal=" << pdf_traversal
                << " xs=" << its_xs.p.toString()
                << " xe=" << pRec_xe.p.toString()
                << " xp=" << position.toString()
                << " meshIndex=" << meshIndex
                << " triangleIndex=" << triangleIndex);
            return 0.0f;
        }
#endif
    }
    // 3. Compute leaf/subtree PDF in SOLID ANGLE
    const BVHNode &node = m_nodes[nodeIndex];
    const BVHNodeInfo &nodeInfo = m_nodeInfos[nodeIndex];

    Float pdf_leaf_sa = 0.0f;

    if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
        // --- LIGHTCUTS FIX: Mimic the area-based descent of sampleNodePrimitive ---
        Float descent_prob = 1.0f;
        size_t currentIdx = nodeIndex;
        uint32_t currentStart = nodeStart;

        // Traverse down to the actual leaf that contains our primitive
        while (!m_nodes[currentIdx].isLeaf()) {
            size_t leftIdx = currentIdx + 1;
            size_t rightIdx = m_nodes[currentIdx].secondChildOffset;
            
            float left_area = m_nodeInfos[leftIdx].surfaceArea;
            float right_area = m_nodeInfos[rightIdx].surfaceArea;

#ifdef DEBUG_BVH_NAN
            if (!std::isfinite(left_area) || !std::isfinite(right_area) || (left_area + right_area) <= 0.0f) {
                BVH_NAN_LOG("pdfGeometry invalid area descent in SphericalAABB"
                    << " currentIdx=" << currentIdx
                    << " left_area=" << left_area
                    << " right_area=" << right_area
                    << " xs=" << its_xs.p.toString()
                    << " xp=" << position.toString());
                return 0.0f;
            }
#endif

            float left_prob = left_area / (left_area + right_area);

            uint32_t leftCount = m_nodes[leftIdx].nSubtreePrimitives;
            bool inLeft = (primIdx < currentStart + leftCount);

            if (inLeft) {
                descent_prob *= left_prob;
                currentIdx = leftIdx;
            } else {
                descent_prob *= (1.0f - left_prob);
                currentIdx = rightIdx;
                currentStart += leftCount;
            }

#ifdef DEBUG_BVH_NAN
            if (!std::isfinite(descent_prob) || descent_prob <= 0.0f) {
                BVH_NAN_LOG("pdfGeometry invalid descent_prob in SphericalAABB"
                    << " currentIdx=" << currentIdx
                    << " descent_prob=" << descent_prob
                    << " left_prob=" << left_prob
                    << " xs=" << its_xs.p.toString()
                    << " xp=" << position.toString());
                return 0.0f;
            }
#endif
        }

        // Now we have the actual leaf that was evaluated in sampleLeafSphericalAABB
        const BVHNode &actualLeaf = m_nodes[currentIdx];

        Vector wo = position - its_xs.p;
        Float dist_sq = wo.lengthSquared();
        Float dist = std::sqrt(dist_sq);

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(dist_sq) || bvhInvalidFloat(dist) || dist <= Epsilon) {
            BVH_NAN_LOG("pdfGeometry invalid SphericalAABB direction"
                << " dist_sq=" << dist_sq
                << " dist=" << dist
                << " xs=" << its_xs.p.toString()
                << " xp=" << position.toString()
                << " currentIdx=" << currentIdx);
            return 0.0f;
        }
#endif
        
        if (dist > Epsilon) {
            wo /= dist;
            Float pdf_solid_angle = 0.0f;
            AABBSphPdf(actualLeaf.bounds, its_xs.p, wo, pdf_solid_angle);

#ifdef DEBUG_BVH_NAN
            if (bvhInvalidFloat(pdf_solid_angle) || pdf_solid_angle < 0.0f) {
                BVH_NAN_LOG("pdfGeometry invalid AABBSphPdf result"
                    << " pdf_solid_angle=" << pdf_solid_angle
                    << " descent_prob=" << descent_prob
                    << " xs=" << its_xs.p.toString()
                    << " xp=" << position.toString()
                    << " leafIdx=" << currentIdx);
                return 0.0f;
            }
#endif
            
            // Final PDF is the probability of picking this leaf * the AABB solid angle PDF
            pdf_leaf_sa = descent_prob * pdf_solid_angle;
        }

    } 
    else if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB_anyNode) {
        Float pdf_solid_angle = 0.0f;
        Vector d = position - its_xs.p;
        Float dist_sq = d.lengthSquared();

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(dist_sq) || dist_sq <= Epsilon * Epsilon) {
            BVH_NAN_LOG("pdfGeometry invalid SphericalAABB_anyNode direction"
                << " dist_sq=" << dist_sq
                << " xs=" << its_xs.p.toString()
                << " xp=" << position.toString()
                << " nodeIndex=" << nodeIndex);
            return 0.0f;
        }
#endif

        AABBSphPdf(node.bounds, its_xs.p, normalize(d), pdf_solid_angle);

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(pdf_solid_angle) || pdf_solid_angle < 0.0f) {
            BVH_NAN_LOG("pdfGeometry invalid AABBSphPdf result in SphericalAABB_anyNode"
                << " pdf_solid_angle=" << pdf_solid_angle
                << " xs=" << its_xs.p.toString()
                << " xp=" << position.toString()
                << " nodeIndex=" << nodeIndex);
            return 0.0f;
        }
#endif

        pdf_leaf_sa = pdf_solid_angle; // No need to multiply by descent_prob because we are already sampling uniformly in the entire node.
    }
    else {
        // Primitive (Area) -> Convert to Solid Angle
        const std::vector<TriMesh*> &meshes = scene->getMeshes();
        const TriMesh *mesh = meshes[meshIndex];
        const Triangle &tri = mesh->getTriangles()[triangleIndex];
        
        if (nodeInfo.surfaceArea <= 0.0f) return 0.0f;

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat((Float) nodeInfo.surfaceArea)) {
            BVH_NAN_LOG("pdfGeometry invalid node surfaceArea"
                << " nodeIndex=" << nodeIndex
                << " nodeSurfaceArea=" << nodeInfo.surfaceArea
                << " xs=" << its_xs.p.toString()
                << " xp=" << position.toString()
                << " meshIndex=" << meshIndex
                << " triangleIndex=" << triangleIndex);
            return 0.0f;
        }
#endif
        
        // This is mathematically correct for primitive sampling because:
        // Prob(leaf|node) * Prob(prim|leaf) * Pdf(p|prim) = 
        // (leaf_area/node_area) * (prim_area/leaf_area) * (1/prim_area) = 1/node_area
        Float pdf_area = 1.0f / nodeInfo.surfaceArea; 

        // Convert Area -> Solid Angle
        const Point *verts = mesh->getVertexPositions();
        Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
        Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
        Normal n = normalize(cross(e1, e2));
        
        Vector d = position - its_xs.p;
        Float dist_sq = d.lengthSquared();
        Float dist = std::sqrt(dist_sq);

    #ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(dist_sq) || bvhInvalidFloat(dist) || dist <= Epsilon) {
            BVH_NAN_LOG("pdfGeometry invalid primitive direction"
            << " dist_sq=" << dist_sq
            << " dist=" << dist
            << " nodeIndex=" << nodeIndex
            << " xs=" << its_xs.p.toString()
            << " xp=" << position.toString()
            << " meshIndex=" << meshIndex
            << " triangleIndex=" << triangleIndex);
            return 0.0f;
        }
    #endif

        Vector wo = d / dist;
        Float cos_theta = std::abs(dot(n, -wo));

    #ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(cos_theta)) {
            BVH_NAN_LOG("pdfGeometry invalid primitive cos_theta"
            << " cos_theta=" << cos_theta
            << " n=" << n.toString()
            << " wo=" << wo.toString()
            << " xs=" << its_xs.p.toString()
            << " xp=" << position.toString()
            << " meshIndex=" << meshIndex
            << " triangleIndex=" << triangleIndex);
            return 0.0f;
        }
    #endif

        if (cos_theta <= Epsilon) return 0.0f; // Singularity
        
        pdf_leaf_sa = pdf_area * (dist_sq / cos_theta);

#ifdef DEBUG_BVH_NAN
        if (bvhInvalidFloat(pdf_leaf_sa) || pdf_leaf_sa <= 0.0f) {
            BVH_NAN_LOG("pdfGeometry invalid primitive pdf_leaf_sa"
                << " pdf_leaf_sa=" << pdf_leaf_sa
                << " pdf_area=" << pdf_area
                << " dist_sq=" << dist_sq
                << " cos_theta=" << cos_theta
                << " xs=" << its_xs.p.toString()
                << " xp=" << position.toString()
                << " meshIndex=" << meshIndex
                << " triangleIndex=" << triangleIndex);
            return 0.0f;
        }
#endif
    }

    Float final_pdf = pdf_traversal * pdf_leaf_sa;

#ifdef DEBUG_BVH_NAN
    if (bvhInvalidFloat(final_pdf) || final_pdf < 0.0f) {
        BVH_NAN_LOG("pdfGeometry invalid final pdf"
            << " final_pdf=" << final_pdf
            << " pdf_traversal=" << pdf_traversal
            << " pdf_leaf_sa=" << pdf_leaf_sa
            << " nodeIndex=" << nodeIndex
            << " samplingMode=" << (int) m_samplingMode
            << " xs=" << its_xs.p.toString()
            << " xe=" << pRec_xe.p.toString()
            << " xp=" << position.toString()
            << " meshIndex=" << meshIndex
            << " triangleIndex=" << triangleIndex);
        return 0.0f;
    }
#endif

    return final_pdf;
}

// ---------------------------------------------------------------------------
// pdfGeometryCamera (shortcut overload)
// Resolves mesh/triangle indices from the intersection, then delegates to the
// full overload below.  Mirrors the shortcut pdfGeometry overload exactly,
// but passes (x_o, sensor) instead of its_xs.
// ---------------------------------------------------------------------------
Float GeometryBVH::pdfGeometryCamera(
    const Scene *scene,
    const Point &x_o,
    const Sensor *sensor,
    const PositionSamplingRecord &pRec_xe,
    const Intersection &its_xn
) const {
    if (its_xn.shape == NULL) {
        Log(EWarn, "pdfGeometryCamera: Intersection has no shape, cannot compute PDF.");
        return 0.0f;
    }

    const TriMesh *hitMesh = dynamic_cast<const TriMesh *>(its_xn.shape);
    if (hitMesh == NULL) {
        Log(EWarn, "pdfGeometryCamera: Intersection shape is not a TriMesh, cannot compute PDF.");
        return 0.0f;
    }

    const std::vector<TriMesh *> &meshes = scene->getMeshes();
    uint32_t meshIndex = 0xFFFFFFFF;
    for (uint32_t i = 0; i < meshes.size(); ++i) {
        if (meshes[i] == hitMesh) {
            meshIndex = i;
            break;
        }
    }

    if (meshIndex == 0xFFFFFFFF)
        return 0.0f;

    if (its_xn.primIndex >= hitMesh->getTriangleCount())
        return 0.0f;

    return this->pdfGeometryCamera(
        scene, x_o, sensor, pRec_xe,
        its_xn, meshIndex, its_xn.primIndex, its_xn.p);
}

// ---------------------------------------------------------------------------
// pdfGeometryCamera (full overload)
// Mirrors pdfGeometry exactly, with two substitutions:
//   • computeNodeImportance  →  computeNodeImportanceGlint
//   • its_xs.p               →  x_o
// Everything else (primitive lookup, nodeStart tracking, lightcuts, the three
// sampling-mode branches) is identical.
// ---------------------------------------------------------------------------
Float GeometryBVH::pdfGeometryCamera(
    const Scene *scene,
    const Point &x_o,
    const Sensor *sensor,
    const PositionSamplingRecord &pRec_xe,
    const Intersection &its_xn,
    uint32_t meshIndex,
    uint32_t triangleIndex,
    const Point &position
) const {
    if (m_nodes.empty() || m_primitives.empty()) return 0.0f;

    // 1. Find primitive index in flattened BVH array
    if (meshIndex >= m_meshOffsets.size()) return 0.0f;
    uint32_t lookupIdx = m_meshOffsets[meshIndex] + triangleIndex;
    if (lookupIdx >= m_triangleToPrim.size()) return 0.0f;
    uint32_t primIdx = m_triangleToPrim[lookupIdx];
    if (primIdx == 0xFFFFFFFF) return 0.0f;

    // 2. Traverse the BVH, accumulating the probability of the path taken.
    //    Uses computeNodeImportanceGlint so the weights match sampleGeometryCamera.
    size_t nodeIndex = 0;
    uint32_t nodeStart = 0;
    Float pdf_traversal = 1.0f;

    while (!m_nodes[nodeIndex].isLeaf()) {
        const size_t leftIdx  = nodeIndex + 1;
        const size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;
        const BVHNode &leftNode = m_nodes[leftIdx];

        const Float left_imp  = computeNodeImportanceGlint(scene, this, (int) leftIdx,  x_o, sensor, pRec_xe) * m_probmult + m_defensive_pdf;
        const Float right_imp = computeNodeImportanceGlint(scene, this, (int) rightIdx, x_o, sensor, pRec_xe) * m_probmult + m_defensive_pdf;
        const Float total     = left_imp + right_imp;

        if (total <= 0.0f) return 0.0f;

        if (m_enableLightcuts && total < m_lightcutThreshold * 2.f * m_defensive_pdf)
            break;

        const uint32_t leftCount = leftNode.nSubtreePrimitives;
        const bool inLeft = (primIdx < nodeStart + leftCount);

        if (inLeft) {
            pdf_traversal *= (left_imp / total);
            nodeIndex = leftIdx;
        } else {
            pdf_traversal *= (right_imp / total);
            nodeIndex = rightIdx;
            nodeStart += leftCount;
        }
    }

    // 3. Compute leaf/subtree PDF in solid angle.
    //    Uses x_o (camera origin) as the "from" point instead of its_xs.p.
    const BVHNode     &node     = m_nodes[nodeIndex];
    const BVHNodeInfo &nodeInfo = m_nodeInfos[nodeIndex];
    Float pdf_leaf_sa = 0.0f;

    if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
        // Descend by area to find the actual leaf that held our primitive,
        // accumulating the area-proportional descent probability.
        Float descent_prob = 1.0f;
        size_t currentIdx  = nodeIndex;
        uint32_t currentStart = nodeStart;

        while (!m_nodes[currentIdx].isLeaf()) {
            const size_t leftIdx2  = currentIdx + 1;
            const size_t rightIdx2 = m_nodes[currentIdx].secondChildOffset;

            const float left_area  = m_nodeInfos[leftIdx2].surfaceArea;
            const float right_area = m_nodeInfos[rightIdx2].surfaceArea;
            if ((left_area + right_area) <= 0.0f) return 0.0f;

            const float left_prob = left_area / (left_area + right_area);
            const uint32_t leftCount2 = m_nodes[leftIdx2].nSubtreePrimitives;
            const bool inLeft2 = (primIdx < currentStart + leftCount2);

            if (inLeft2) {
                descent_prob *= left_prob;
                currentIdx = leftIdx2;
            } else {
                descent_prob *= (1.0f - left_prob);
                currentIdx = rightIdx2;
                currentStart += leftCount2;
            }
        }

        const BVHNode &actualLeaf = m_nodes[currentIdx];
        Vector wo = position - x_o;
        const Float dist_sq = wo.lengthSquared();
        const Float dist    = std::sqrt(dist_sq);
        if (dist <= Epsilon) return 0.0f;
        wo /= dist;

        Float pdf_solid_angle = 0.0f;
        AABBSphPdf(actualLeaf.bounds, x_o, wo, pdf_solid_angle);
        pdf_leaf_sa = descent_prob * pdf_solid_angle;

    } else if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB_anyNode) {
        const Vector d = position - x_o;
        if (d.lengthSquared() <= Epsilon * Epsilon) return 0.0f;
        Float pdf_solid_angle = 0.0f;
        AABBSphPdf(node.bounds, x_o, normalize(d), pdf_solid_angle);
        pdf_leaf_sa = pdf_solid_angle;

    } else {
        // Primitive (area) mode — convert area PDF to solid angle.
        if (nodeInfo.surfaceArea <= 0.0f) return 0.0f;
        const Float pdf_area = 1.0f / nodeInfo.surfaceArea;

        const std::vector<TriMesh *> &meshes = scene->getMeshes();
        const TriMesh *mesh = meshes[meshIndex];
        const Triangle &tri = mesh->getTriangles()[triangleIndex];
        const Point *verts  = mesh->getVertexPositions();
        const Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
        const Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
        const Normal n  = normalize(cross(e1, e2));

        Vector d = position - x_o;
        const Float dist_sq = d.lengthSquared();
        const Float dist    = std::sqrt(dist_sq);
        if (dist <= Epsilon) return 0.0f;
        const Vector wo       = d / dist;
        const Float cos_theta = std::abs(dot(n, -wo));
        if (cos_theta <= Epsilon) return 0.0f;

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
    if (!pRec_xe.object) {
        std::cerr << "!!!! No emitter at xe intersection, zero contribution.\n";
        std::cerr << "pRec_xe.object must be set to an emitter\n";
        return Spectrum(0.0f);
    }
    // 1. Evaluate visibility if needed
    // 1.1. Visibility between xs and xp
    Vector d_sp = its_xp.p - its_xs.p;
    Float dist_sp = d_sp.length();
    d_sp /= dist_sp;

    // std::cout << "1. Evaluating visibility...\n";

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
    // std::cout << "Visibility check passed.\n";
    // 2. Evaluate contribution of this subpath: xs -> xp -> xe
    // 2.1. Evaluate bsdf
    // 2.1.1. Evaluate BSDF at xs for the direction towards xp
    Vector l_xs_to_xp = its_xs.toLocal(d_sp);
    // 2.1.2. Local directions at xp
    Vector l_xp_in  = its_xp.toLocal(-d_sp);
    Vector l_xp_out = its_xp.toLocal(d_pe);

    // std::cout << "2. Evaluating BSDF...\n";
    BSDFSamplingRecord bRec_xs(its_xs, l_xs_to_xp);
    const BSDF *bsdf_xs = its_xs.getBSDF();
    if (!bsdf_xs) {
        std::cerr << "!!!! No BSDF at xs intersection, zero contribution.\n";
        return Spectrum(0.0f);
    }
    Spectrum f_xs = bsdf_xs->eval(bRec_xs);
    BSDFSamplingRecord bRec_xp(its_xp, l_xp_in, l_xp_out);

    // std::cout << "Local direction xs->xp: (" << l_xs_to_xp.x << ", " << l_xs_to_xp.y << ", " << l_xs_to_xp.z << ")\n";  
    
    const BSDF *bsdf_xp = its_xp.getBSDF();
    if (!bsdf_xp) {
        std::cerr << "!!!! No BSDF at xp intersection, zero contribution.\n";   
        return Spectrum(0.0f);
    }
    Spectrum f_xp = bsdf_xp->eval(bRec_xp);


    // std::cout << "Local direction xp in: (" << l_xp_in.x << ", " << l_xp_in.y << ", " << l_xp_in.z << ")\n";

    if (f_xs.isZero() || f_xp.isZero()) {
        return Spectrum(0.0f);
    }

    // 2.2. Evaluate emitter radiance at xe in the direction of xp
    const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
    DirectionSamplingRecord dRec_xe(-d_pe);
    dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
    Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);
    
    if (Le.isZero()) {
        // std::cout << "Zero contribution due to emitter evaluation.\n";
        return Spectrum(0.0f);
    }

    // std::cout << "Emitter radiance at xe: " << Le.toString() << "\n";

    // 2.3. Geometry term
    Float cos_theta_xs_out = std::abs(dot(its_xs.geoFrame.n, d_sp));
    Float cos_theta_xp_in = std::abs(dot(its_xp.geoFrame.n, -d_sp));
    Float cos_theta_xp_out = std::abs(dot(its_xp.geoFrame.n, d_pe));
    // Float cos_theta_xe_in = std::abs(dot(pRec_xe.n, -d_pe));

    Float G_sp = (cos_theta_xp_in) / (dist_sp * dist_sp);
    Float G_pe = (1.f) / (dist_pe * dist_pe);
    // TODO(jorge): revise where geometry terms have to be added

    Spectrum contribution = f_xs * G_sp * f_xp * G_pe * Le;
    if (!contribution.isValid()) {
        std::stringstream ss;
        ss << "Contribution breakdown:\n"
           << "f_xs: " << f_xs.toString() << "\n"
           << "G_sp: " << G_sp << "\n"
           << "f_xp: " << f_xp.toString() << "\n"
           << "G_pe: " << G_pe << "\n"
           << "Le: " << Le.toString() << "\n"
           << "Contrib: " << (f_xs * G_sp * f_xp * G_pe * Le).toString() << "\n";

        // std::cout << "Zero contribution due to BSDF evaluation.\n";
        ss << " bsdf: " << its_xs.getBSDF()->toString() << "\n";
        ss << " bsdf: " << its_xp.getBSDF()->toString() << "\n";

        // Debug: print points, normals and local directions to help diagnose zero BSDF
        ss << "DEBUG its_xs.p: (" << its_xs.p.x << ", " << its_xs.p.y << ", " << its_xs.p.z << ")\n";
        ss << "DEBUG its_xs.geo.n: (" << its_xs.geoFrame.n.x << ", " << its_xs.geoFrame.n.y << ", " << its_xs.geoFrame.n.z << ")\n";
        ss << "DEBUG its_xp.p: (" << its_xp.p.x << ", " << its_xp.p.y << ", " << its_xp.p.z << ")\n";
        ss << "DEBUG its_xp.geo.n: (" << its_xp.geoFrame.n.x << ", " << its_xp.geoFrame.n.y << ", " << its_xp.geoFrame.n.z << ")\n";
        ss << "DEBUG local xs->xp (its_xs.toLocal(d_sp)): (" << l_xs_to_xp.x << ", " << l_xs_to_xp.y << ", " << l_xs_to_xp.z << ")\n";
        ss << "DEBUG local xp in (its_xp.toLocal(-d_sp)): (" << l_xp_in.x << ", " << l_xp_in.y << ", " << l_xp_in.z << ")\n";
        ss << "DEBUG local xp out (its_xp.toLocal(d_pe)): (" << l_xp_out.x << ", " << l_xp_out.y << ", " << l_xp_out.z << ")\n";

        // Print BSDFSamplingRecord-like info (most relevant: directions)
        ss << "DEBUG bRec_xs.localDir: (" << l_xs_to_xp.x << ", " << l_xs_to_xp.y << ", " << l_xs_to_xp.z << ")\n";
        ss << "DEBUG bRec_xp.localIn: (" << l_xp_in.x << ", " << l_xp_in.y << ", " << l_xp_in.z<< ")"<< "\n";
        ss << "DEBUG bRec_xp.localOut: (" << l_xp_out.x << ", " << l_xp_out.y << ", " << l_xp_out.z << ")\n";
        ss << "\n";
        std::cout << ss.str();
    }
    
    return contribution;
}

// ---------------------------------------------------------------------------
// evalGlintContribution
// Evaluates the x_n → x_e segment of a glint path: x_o → x_n → x_e.
// The camera-side factor W_e (sensor importance for the x_o → x_n direction)
// and the x_o → x_n visibility are handled by sampleSensorDirect in
// the integrator before calling this function.
// Returns f_xn * cos(theta_xn->xo) * G_ne * Le, i.e. the full x_n contribution
// excluding W_e (which lacks only cos(theta_xn->xo)/dist_no^2, provided by W_e times
// this function's cos factor).  Does NOT divide by pRec_xe.pdf (caller's job).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// sampleGeometryCamera
// BVH geometry sampling from the camera origin x_o (glints use case).
// Uses computeNodeImportanceGlint for traversal weights, which incorporates:
//   • Camera FoV (nodes outside view get zero importance)
//   • Emitter visibility direction
//   • Diffuse BSDF at x_n via VMF LUT (both camera-side and emitter-side cosines)
//   • Geometry terms G(x_n,x_o) and G(x_n,x_e)
// Leaf sampling uses x_o as the "from" point (not the surface intersection).
// ---------------------------------------------------------------------------
bool GeometryBVH::sampleGeometryCamera(
    const Scene *scene,
    Sampler *sampler,
    const Point &x_o,
    const Sensor *sensor,
    const PositionSamplingRecord &pRec_xe,
    Intersection &its_xn,
    Float &pdf)
{
    if (m_nodes.empty() || m_primitives.empty()) return false;

    size_t nodeIndex = 0;
    Float pdf_traversal = 1.0f;

    while (!m_nodes[nodeIndex].isLeaf()) {
        const size_t leftIdx  = nodeIndex + 1;
        const size_t rightIdx = m_nodes[nodeIndex].secondChildOffset;

        const Float left_imp_raw  = computeNodeImportanceGlint(
            scene, this, (int) leftIdx,  x_o, sensor, pRec_xe);
        const Float right_imp_raw = computeNodeImportanceGlint(
            scene, this, (int) rightIdx, x_o, sensor, pRec_xe);
        const Float left_imp  = left_imp_raw  * m_probmult + m_defensive_pdf;
        const Float right_imp = right_imp_raw * m_probmult + m_defensive_pdf;
        const Float total_imp = left_imp + right_imp;

        // Lightcuts: prune subtrees whose total importance is negligible
        if (m_enableLightcuts && total_imp < m_lightcutThreshold * 2.f * m_defensive_pdf)
            break;

        const Float left_prob = left_imp / total_imp;
        if (sampler->next1D() < left_prob) {
            nodeIndex = leftIdx;
            pdf_traversal *= left_prob;
        } else {
            nodeIndex = rightIdx;
            pdf_traversal *= (1.f - left_prob);
        }
    }

    // Leaf (or early-stopped interior) sampling using x_o as the "from" point
    const BVHNode &node = m_nodes[nodeIndex];
    Float pdf_leaf = 0.f;
    bool success   = false;

    if (node.isLeaf()) {
        if (node.nLeafPrimitives == 0) return false;
        if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB) {
            success = sampleLeafSphericalAABB(scene, node, nodeIndex,
                                              sampler->next2D(), x_o, its_xn, pdf_leaf);
        } else {
            success = sampleLeafPrimitive(scene, node, nodeIndex,
                                          sampler->next1D(), sampler->next2D(),
                                          x_o, its_xn, pdf_leaf, false);
        }
    } else if (m_samplingMode == GeometryBVHSamplingMode::SphericalAABB_anyNode) {
        success = sampleLeafSphericalAABB(scene, node, nodeIndex,
                                          sampler->next2D(), x_o, its_xn, pdf_leaf);
    } else {
        // Lightcuts stopped at an interior node: descend by area to a leaf
        success = sampleNodePrimitive(scene, nodeIndex, sampler, x_o, its_xn, pdf_leaf, false);
    }

    if (!success) return false;
    pdf = pdf_traversal * pdf_leaf;
    return true;
}

MTS_NAMESPACE_END
