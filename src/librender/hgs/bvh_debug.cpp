/*
    Debug output of the SamplingBVH (per-node CSV used for the importance
    visualization of Fig. 4) and Gaussian sampling for the Monte Carlo
    reference importance functions.
*/

#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <algorithm>
#include <fstream>
#include <mitsuba/core/matrix.h>
#include <cmath>
#include <cstdlib>

#include <mitsuba/core/vmf.h>

MTS_NAMESPACE_BEGIN


/// Inverse CDF of the standard normal distribution, clamped to avoid infinities
static Float safeErf(Float u) {
    return SQRT_TWO * math::erfinv(mitsuba::math::clamp(2.0f * u - 1.0f, -0.999999f, 0.999999f));
}
// -------------------------------------------------------------------------------
Point3f Gaussian3D::sample(const Point3f &sample3) const {
    // 1. Convert uniform sample [0,1]^3 to standard normal
    Vector z(
        safeErf(sample3.x),
        safeErf(sample3.y),
        safeErf(sample3.z)
    );
    
    // -----------
    
    // 2. Robust Cholesky Decomposition of a 3x3 Matrix (sigma)
    // Adds a tiny epsilon to the diagonal for numerical stability
    Float eps = 1e-6f;
    Float zero_tol = 1e-7f; // Tolerance for division
    Matrix3x3 L(0.0f);

    // Compute L(0,0)
    Float d00 = m_covariance(0,0) + eps;
    L(0,0) = std::sqrt(std::max(d00, 0.0f));

    // Compute L(1,0) and L(2,0) guarded by L(0,0)
    if (L(0,0) > zero_tol) {
        L(1,0) = m_covariance(1,0) / L(0,0);
        L(2,0) = m_covariance(2,0) / L(0,0);
    } else {
        L(1,0) = 0.0f;
        L(2,0) = 0.0f;
    }

    // Compute L(1,1)
    Float d11 = m_covariance(1,1) + eps - L(1,0)*L(1,0);
    L(1,1) = std::sqrt(std::max(d11, 0.0f));

    // Compute L(2,1) guarded by L(1,1)
    if (L(1,1) > zero_tol) {
        L(2,1) = (m_covariance(2,1) - L(2,0)*L(1,0)) / L(1,1);
    } else {
        L(2,1) = 0.0f;
    }

    // Compute L(2,2)
    Float d22 = m_covariance(2,2) + eps - L(2,0)*L(2,0) - L(2,1)*L(2,1);
    L(2,2) = std::sqrt(std::max(d22, 0.0f));
    // ------------
    // 3. Apply the transformation: p = mu + L * z
    Vector offset = L * z;

    // z was a standard normal, the result is a sample of N(mean, sigma)
    return m_mean + offset;
}


/// Output of SamplingBVH::print(), read by scripts/polyscope/view_bvh.py
static const std::string BVH_DEBUG_FILE = "bvh_debug.csv";

/**
 * Writes one CSV row per node with its bounds, statistics and several node
 * importance weights for fixed query vertices x_s and x_e: the ground truth
 * (Monte Carlo over the node's primitives, with and without visibility), the
 * weight used for sampling ("ours", computeNodeImportance) and the reference
 * variants of bvh_importance.cpp. scripts/polyscope/view_bvh.py visualizes
 * the file (importance visualization of Fig. 4).
 */
void SamplingBVH::print() {
    const int N_SAMPLES = 2000; // Monte Carlo samples per node for the reference weights

    std::ofstream out(BVH_DEBUG_FILE.c_str());
    if (!out) {
        Log(EError, "Unable to open %s for the BVH debug output", BVH_DEBUG_FILE.c_str());
        return;
    }

    // 1. Fixed query vertices: x_s is "debugQueryPoint" (an upward-facing
    //    surface point that uses the BSDF of shape "debugQueryShape"), and x_e
    //    is a sample on the first emitter of the scene.
    ref<Sampler> sampler = m_scene->getSampler();
    const ref_vector<Shape> &shapes = m_scene->getShapes();
    const ref_vector<Emitter> &emitters = m_scene->getEmitters();
    if (m_debugQueryShape < 0 || (size_t) m_debugQueryShape >= shapes.size() || emitters.empty()) {
        Log(EError, "outputDebugCSV: debugQueryShape=%i is out of range (%i shapes) or the scene "
            "has no emitter", m_debugQueryShape, (int) shapes.size());
        return;
    }
    const Shape *shape = shapes[m_debugQueryShape].get();
    const Emitter *emitter = emitters[0].get();

    Intersection its_xs;
    its_xs.p = m_debugQueryPoint;
    its_xs.shFrame = Frame(Normal(0, 0, 1));
    its_xs.geoFrame = Frame(Normal(0, 0, 1));
    its_xs.wi = Vector(0.f, 0.f, 1.f);
    its_xs.shape = shape;

    PositionSamplingRecord pRec_xe;
    emitter->samplePosition(pRec_xe, sampler->next2D());
    pRec_xe.object = emitter;

    Log(EInfo, "Writing %s: x_s = %s on shape %i, x_e = %s", BVH_DEBUG_FILE.c_str(),
        its_xs.p.toString().c_str(), m_debugQueryShape, pRec_xe.p.toString().c_str());

    out << "node_index,level,aabb_min_x,aabb_min_y,aabb_min_z,aabb_max_x,aabb_max_y,aabb_max_z,"
           "n_subtree_primitives,n_leaf_primitives,node_type,valid,surface_area,"
           "gaussian_mu_x,gaussian_mu_y,gaussian_mu_z,gaussian_cov_xx,gaussian_cov_xy,gaussian_cov_xz,gaussian_cov_yy,gaussian_cov_yz,gaussian_cov_zz,"
           "vmf_mean_x,vmf_mean_y,vmf_mean_z,vmf_kappa,"
           "imp_gt_vis,imp_gt_novis,imp_ours,imp_ours_mc_vis,imp_ours_mc_novis,imp_lut_vis,imp_lut_novis,imp_unscented_mc_vis,imp_unscented_mc_novis,imp_unscented_lut_vis,imp_unscented_lut_novis\n";
    for (size_t i = 0; i < m_nodes.size(); ++i) {
        // m_csvEnabledLevels is a mask of 32 bits where each bit indicates whether to output CSV for that level. For example, if m_csvEnabledLevels = 0b111 (7 in decimal), then levels 0, 1 and 2 will be outputted, but not deeper levels.
        size_t level = getMaxDistToLeaf(i); // the level of a node is the dist to its furthest leaf.
        bool levelEnabled = level < 32 && (m_csvEnabledLevels & (uint32_t(1) << level)) != 0;
        if (!levelEnabled)
            continue;

        Float sum_importance_groundtruth = computeNodeImportanceGroundTruth(
            m_scene,
            this,
            static_cast<int>(i),
            its_xs,
            pRec_xe,
            N_SAMPLES,
            true
        );

        Float sum_importance_groundtruth_novisib = computeNodeImportanceGroundTruth(
            m_scene,
            this,
            static_cast<int>(i),
            its_xs,
            pRec_xe,
            N_SAMPLES,
            false
        );

        // Node Gaussian and VMF, as aggregated from the triangles
        Point gaussianMean;
        Matrix3x3 gaussianCov;
        m_nodeInfos[i].gaussianAggregate.getGaussian(m_nodeInfos[i].surfaceArea, gaussianMean, gaussianCov);
        Vector meanDir = m_nodeInfos[i].getVMFMeanDirection();
        Float kappa = m_nodeInfos[i].getVMFKappa();

        // Weight used by the traversal, and reference variants
        float importance_ours = computeNodeImportance(m_scene, this, static_cast<int>(i), its_xs, pRec_xe);
        float importance_mc_vis = computeNodeImportance_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, true); 
        float importance_mc_novis = computeNodeImportance_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, false); 
        float importance_lut_vis = computeNodeImportance_MC_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, true);
        float importance_lut_novis = computeNodeImportance_MC_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, false);
        float importance_unscented_mc_vis = computeNodeImportance_Unscented_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, true);
        float importance_unscented_mc_novis = computeNodeImportance_Unscented_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, false);
        float importance_unscented_lut_vis = computeNodeImportance_Unscented_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, true); // no samples for these last two
        float importance_unscented_lut_novis = computeNodeImportance_Unscented_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, false);


        out << i << "," << level << ","
            << m_nodes[i].bounds.min.x << "," << m_nodes[i].bounds.min.y << "," << m_nodes[i].bounds.min.z << ","
            << m_nodes[i].bounds.max.x << "," << m_nodes[i].bounds.max.y << "," << m_nodes[i].bounds.max.z << ","
            << m_nodes[i].nSubtreePrimitives << ","
            << m_nodes[i].nLeafPrimitives << ","
            << (m_nodes[i].isLeaf() ? "leaf" : "interior") << ","
            << m_nodeInfos[i].valid << ","
            << m_nodeInfos[i].surfaceArea << ","
            // Fitted Gaussian parameters
            << gaussianMean.x << "," << gaussianMean.y << "," << gaussianMean.z << ","
            << gaussianCov(0, 0) << "," // cov_xx
            << gaussianCov(0, 1) << "," // cov_xy
            << gaussianCov(0, 2) << "," // cov_xz
            << gaussianCov(1, 1) << "," // cov_yy
            << gaussianCov(1, 2) << "," // cov_yz
            << gaussianCov(2, 2) << "," // cov_zz
            // VMF:
            << meanDir.x << "," << meanDir.y << "," << meanDir.z << "," << kappa << ","
            // Ground truth importance values for debugging
            << sum_importance_groundtruth << ","
            << sum_importance_groundtruth_novisib << ","
            // Importance weights
            << importance_ours << ","
            << importance_mc_vis << ","
            << importance_mc_novis << ","
            << importance_lut_vis << ","
            << importance_lut_novis << ","
            << importance_unscented_mc_vis << ","
            << importance_unscented_mc_novis << ","
            << importance_unscented_lut_vis << ","
            << importance_unscented_lut_novis
            << "\n";

        // Validate all importance-related values are finite and non-negative
        auto checkNonNeg = [&](const char *name, Float v) {
            bool error = !std::isfinite(v) || v < 0.0f;
            if (error)
                Log(EWarn, "Node %i has an invalid importance value for %s: %f", (int) i, name, v);
            return error;

        };

        bool error = checkNonNeg("sum_importance_groundtruth", sum_importance_groundtruth);
        bool error2 = checkNonNeg("sum_importance_groundtruth_novisib", sum_importance_groundtruth_novisib);
        bool error3 = checkNonNeg("importance_ours", importance_ours);
        bool error4 = checkNonNeg("importance_mc_vis", importance_mc_vis);
        bool error5 = checkNonNeg("importance_mc_novis", importance_mc_novis);
        bool error6 = checkNonNeg("importance_lut_vis", importance_lut_vis);
        bool error7 = checkNonNeg("importance_lut_novis", importance_lut_novis);
        if (error || error2 || error3 || error4 || error5 || error6 || error7) {
            out.close();
            Log(EError, "Invalid node importance values, aborting the debug output");
        }
    }
    out.close();
}


MTS_NAMESPACE_END