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


Float safeErf(Float u) {
    // Clamp x to a reasonable range to prevent overflow in erfinv
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

    // Compute the pdf - unneeded for now since we want int_volume Gaussian(xv)
    // Float detSigma = m_covariance(0,0) * m_covariance(1,1) * m_covariance(2,2) + 
    //                  2 * m_covariance(0,1) * m_covariance(0,2) * m_covariance(1,2) - 
    //                  m_covariance(0,0) * m_covariance(1,2) * m_covariance(1,2) - 
    //                  m_covariance(1,1) * m_covariance(0,2) * m_covariance(0,2) - 
    //                  m_covariance(2,2) * m_covariance(0,1) * m_covariance(0,1);
    // detSigma = std::max(detSigma, 1e-12f); // Prevent
    // Float normFactor = 1.0f / std::pow(2.0f * M_PI, 1.5f) / std::sqrt(detSigma);
    // Float exponent = -0.5f * dot(offset, m_covariance.inverse() * offset);
    // pdf = normFactor * std::exp(exponent);

    // Compose the result and guard against non-finite / NaN values.
    Point3f result = m_mean + offset; // z was a standard normal, now a sample from N(mean, sigma)

    return result;
}


const std::string BVH_DEBUG_FILE = "bvh_debug.csv";

// --------------------------------------------------------------------
// Normals fitting into VMF:
const std::string BVH_NORMALS_FILE = "bvh_debug_normals.csv";
const int N_NORMALS_SAMPLES = 1000; // Number of random samples to draw for normal distribution debugging
const int node_idx_normals_debug = 0; // Index of the node for which we want to debug the normal distribution of sampled points (if we do per-node normal fitting)


void fitVMF(const std::vector<Normal>& normals, Vector3f& outMeanDir, Float& outKappa) {
    if (normals.empty()) {
        outMeanDir = Vector3f(0.0f, 0.0f, 1.0f);
        outKappa = 0.0f;
        return;
    }

    // 1. Sum all the normal vectors
    Vector3f sum(0.0f);
    for (const auto& n : normals) {
        sum += Vector3f(n); // Ensure we treat the normal as a vector for summation
    }

    // 2. Calculate the length of the sum
    Float sumLength = sum.length();
    
    // 3. The mean direction is the normalized sum
    if (sumLength > 1e-6f) {
        outMeanDir = sum / sumLength;
    } else {
        // If sumLength is ~0, samples are uniformly distributed/opposite
        outMeanDir = Vector3f(0.0f, 0.0f, 1.0f);
    }

    // 4. Calculate the mean resultant length (R-bar)
    // This represents how "clustered" the vectors are (0 to 1)
    Float rBar = sumLength / static_cast<Float>(normals.size());

    // 5. Use the Mitsuba interface to estimate kappa
    // This uses the Banerjee et al. approximation internally
    outKappa = VonMisesFisherDistr::forMeanLength(rBar);
}


Gaussian3D fitWeightedGaussian(const std::vector<Point3f>& points, const std::vector<Float>& weights) {
    SAssert(points.size() == weights.size());
    
    Float totalWeight = 0.0f;
    Vector3f weightedSum(0.0f);

    // 1. Calculate the Weighted Mean
    for (size_t i = 0; i < points.size(); ++i) {
        weightedSum += Vector3f(points[i]) * weights[i];
        totalWeight += weights[i];
    }

    if (totalWeight <= 1e-8f) {
        return { Point3f(0.0f), Matrix3x3(0.0f) }; 
    }

    Point3f mean = Point3f(weightedSum / totalWeight);

    // 2. Calculate the Weighted Covariance Matrix
    Matrix3x3 cov(0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
        Vector3f diff = points[i] - mean;
        
        // Compute outer product: diff * diff^T
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                cov(row, col) += weights[i] * diff[row] * diff[col];
            }
        }
    }

    cov /= totalWeight;

    return { mean, cov };
}

#ifndef USE_SGGX_AGGREGATES
// New print, after switching to VMF instead of SGGX in the BVH build process
void SamplingBVH::print() {
    const int N_SAMPLES = 2000; // Number of random samples to draw for each leaf node
    const bool RECOMPUTE_VMF_AND_GAUSSIAN = false; // if set to true, these are recomputed from random samples
                                                    // otherwise, the BVHNodeInfo data is printed 
    const bool DO_ONLY_LEAVES = false;

    std::ofstream out(BVH_DEBUG_FILE.c_str());
    if (!out) {
        SLog(EError, "Unable to open BVH_DEBUG_FILE for BVH debug output: %s", BVH_DEBUG_FILE.c_str());
        return;
    }
    ref<Sampler> sampler = m_scene->getSampler();

    // 1- Instance the reference xe and xs - emitter and floor samples for importance computation
    const ref_vector<Shape> &shapes = m_scene->getShapes();
    auto shape = shapes[2];
    const ref_vector<Emitter> &emitters = m_scene->getEmitters();
    auto emitter = emitters[0];
    std::cout << "There are " << shapes.size() << " shapes and " << emitters.size() << " emitters in the scene.\n";
    std::cout << "Using shape: " << shape->toString() << " and emitter: " << emitter->toString() << " for importance debugging.\n";

    Intersection its_xs;
    its_xs.p = Point(2.43008, -0.0490401, 0); // Dummy surface sample at origin facing up
    its_xs.shFrame = Frame(Normal(0, 0, 1)); // Normal facing up
    its_xs.geoFrame = Frame(Normal(0, 0, 1));
    its_xs.wi = Vector(0.f, 0.f, 1.f);       
    its_xs.shape = shape;
    
    std::cout << "Shape: " << shape->toString() << std::endl;

    PositionSamplingRecord pRec_xe;
    emitter->samplePosition(pRec_xe, sampler->next2D());
    pRec_xe.object = emitter.get();

    std::cout << "Sampled emitter position: " << pRec_xe.p.toString() << ", normal: " << pRec_xe.n.toString() << ", pdf: " << pRec_xe.pdf << std::endl;

    out << "node_index,level,aabb_min_x,aabb_min_y,aabb_min_z,aabb_max_x,aabb_max_y,aabb_max_z,"
           "n_subtree_primitives,n_leaf_primitives,node_type,valid,surface_area,"
           "gaussian_mu_x,gaussian_mu_y,gaussian_mu_z,gaussian_cov_xx,gaussian_cov_xy,gaussian_cov_xz,gaussian_cov_yy,gaussian_cov_yz,gaussian_cov_zz,"
           "vmf_mean_x,vmf_mean_y,vmf_mean_z,vmf_kappa,"
           "imp_gt_vis,imp_gt_novis,imp_ours,imp_ours_mc_vis,imp_ours_mc_novis,imp_lut_vis,imp_lut_novis,imp_unscented_mc_vis,imp_unscented_mc_novis,imp_unscented_lut_vis,imp_unscented_lut_novis\n";
    for (size_t i = 0; i < m_nodes.size(); ++i) {
        // m_csvEnabledLevels is a mask of 32 bits where each bit indicates whether to output CSV for that level. For example, if m_csvEnabledLevels = 0b111 (7 in decimal), then levels 0, 1 and 2 will be outputted, but not deeper levels.
        size_t level = getMaxDistToLeaf(i); // the level of a node is the dist to its furthest leaf.
        bool levelEnabled = level < 32 && (m_csvEnabledLevels & (uint32_t(1) << level)) != 0;
        if ((DO_ONLY_LEAVES && !m_nodes[i].isLeaf()) || !levelEnabled) {
            continue; // Only do this for leaf nodes for now
        }
        if (i%100 == 0) std::cout << "Processing node " << i << " / " << m_nodes.size() << std::endl;
        std::vector<Normal> sampledNormals(N_SAMPLES);
        std::vector<Point3f> sampledPoints(N_SAMPLES);
        std::vector<Float> sampleWeights(N_SAMPLES);
        
        // std::cout << "1 "  << std::endl;
        Float sum_importance_groundtruth = computeNodeImportanceGroundTruth(
            m_scene,
            this,
            static_cast<int>(i),
            its_xs,
            pRec_xe,
            N_SAMPLES,
            true
        );

        // std::cout << "2 "  << std::endl;
        Float sum_importance_groundtruth_novisib = computeNodeImportanceGroundTruth(
            m_scene,
            this,
            static_cast<int>(i),
            its_xs,
            pRec_xe,
            N_SAMPLES,
            false
        );

        // std::cout << "3 "  << std::endl;
        // Gaussian3D fittedGaussian;
        Point gaussianMean;
        Matrix3x3 gaussianCov;
        Vector3f meanDir;
        Float kappa;
        if (RECOMPUTE_VMF_AND_GAUSSIAN) {
            // Recompute the VMF and Gaussian parameters from the random samples for debugging
            fitVMF(sampledNormals, meanDir, kappa);
            Gaussian3D fittedGaussian = fitWeightedGaussian(sampledPoints, sampleWeights);
            gaussianMean = Point(fittedGaussian.m_mean);
            gaussianCov = fittedGaussian.m_covariance;
        }
        else {
            // Get these from the BVHNodeInfo. This means that the VMF and Gaussian are fitted from each of the triangles instead of random 
            // point samples. It should be more exact but similar!!!!!!!
            meanDir = m_nodeInfos[i].getVMFMeanDirection();
            kappa = m_nodeInfos[i].getVMFKappa();
            m_nodeInfos[i].gaussianAggregate.getGaussian(m_nodeInfos[i].surfaceArea, gaussianMean, gaussianCov);
        }

        // std::cout << "5 "  << std::endl;
        // Ensure our placeholder is non-negative to avoid failing the later checks
        float importance_ours = 0.f; // TODO: replace with real computation
        float importance_mc_vis = computeNodeImportance_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, true); 
        float importance_mc_novis = computeNodeImportance_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, false); 
        float importance_lut_vis = computeNodeImportance_MC_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, true);
        float importance_lut_novis = computeNodeImportance_MC_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, false);
        float importance_unscented_mc_vis = computeNodeImportance_Unscented_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, true);
        float importance_unscented_mc_novis = computeNodeImportance_Unscented_MC(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, N_SAMPLES, false);
        float importance_unscented_lut_vis = computeNodeImportance_Unscented_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, true); // no samples for these last two
        float importance_unscented_lut_novis = computeNodeImportance_Unscented_LUT(m_scene, this, static_cast<int>(i), its_xs, pRec_xe, false);

        // if (true || sum_importance_groundtruth > 0.f || sum_importance_groundtruth_novisib > 0.f) {
        //     std::cout << "Node " << i 
        //             << ": gt_vis = " << sum_importance_groundtruth << ", gt_novis = " << sum_importance_groundtruth_novisib 
        //             << ", ours = " << importance_ours 
        //             << ", mc_vis = " << importance_mc_vis << ", mc_novis = " << importance_mc_novis
        //             << ", lut_vis = " << importance_lut_vis << ", lut_novis = " << importance_lut_novis
        //             << ", unscented_mc_vis = " << importance_unscented_mc_vis << ", unscented_mc_novis = " << importance_unscented_mc_novis
        //             << ", unscented_lut_vis = " << importance_unscented_lut_vis << ", unscented_lut_novis = " << importance_unscented_lut_novis
        //             << std::endl;
        // }
        

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
            // Our importance value for this node (to be implemented)
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
                std::cerr << "Fatal: Node " << i << " has invalid importance value for " << name << " = " << v << std::endl;
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
            std::cerr << "Error: Node " << i << " has invalid importance values. Check the logs for details." << std::endl;
            out.close();
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "Finished processing all nodes. Output written to " << BVH_DEBUG_FILE << std::endl;
    out.close();
    

}
#else 
// This print is for testing the old importance function, w/sggx
void SamplingBVH::print() {
    
    std::ofstream out(BVH_DEBUG_FILE.c_str());
    if (!out) {
        SLog(EError, "Unable to open BVH_DEBUG_FILE for BVH debug output: %s", BVH_DEBUG_FILE.c_str());
        return;
    }

    const ref_vector<Shape> &shapes = m_scene->getShapes();
    auto shape = shapes[2];
    const ref_vector<Emitter> &emitters = m_scene->getEmitters();
    auto emitter = emitters[0];

    Intersection its_xs;
    its_xs.p = Point(2.43008, -0.0490401, 0); // Dummy surface sample at origin facing up
    its_xs.shFrame = Frame(Normal(0, 0, 1)); // Normal facing up
    its_xs.shape = shape;
    std::cout << "Shape: " << shape->toString() << std::endl;

    PositionSamplingRecord pRec_xe;
    pRec_xe.p = Point(-5, 0.725691, 1.99827); // Dummy emitter sample position
    pRec_xe.n = Normal(0.,0.,1.); // Normal facing in +X direction (towards the scene)
    pRec_xe.object = emitter; // Just a dummy emitter reference for the emitter sample

    // <translate x="-5" y="0" z="1"/>

    out << "node_index,aabb_min_x,aabb_min_y,aabb_min_z,aabb_max_x,aabb_max_y,aabb_max_z,"
           "n_subtree_primitives,n_leaf_primitives,node_type,valid,surface_area,center_of_mass_x,center_of_mass_y,center_of_mass_z,importance,importance_gt\n";
    for (size_t i = 0; i < m_nodes.size(); ++i) {
        float importance = computeNodeImportance(
            m_scene,
            this,
            i,
            its_xs,
            pRec_xe,
            m_nodeInfos[i],
            m_nodes[i].bounds,
            1., m_solidAngleScale, m_variancePenaltyScale, 1.f // pass importance config params
        );

        float importance_gt = computeNodeImportanceGroundTruth(
            m_scene,
            this,
            i,
            its_xs,
            pRec_xe,
            1000,
            false,
            m_nodeInfos[i]
        );
        std::cout << "Node " << i << ": importance = " << importance << " importance_gt = " << importance_gt << std::endl;
        
        const BVHNode &node = m_nodes[i];
        const BVHNodeInfo &info = m_nodeInfos[i];
        out << i << ","
            << node.bounds.min.x << "," << node.bounds.min.y << "," << node.bounds.min.z << ","
            << node.bounds.max.x << "," << node.bounds.max.y << "," << node.bounds.max.z << ","
            << node.nSubtreePrimitives << ","
            << node.nLeafPrimitives << ","
            << (node.isLeaf() ? "leaf" : "interior") << ","
            << info.valid << ","
            << info.surfaceArea << ","
            << info.centerOfMass.x << "," << info.centerOfMass.y << "," << info.centerOfMass.z << "," 
            << importance << ","
            << importance_gt << "\n";
    }

    // Printing some normals for debugging:
    std::ofstream outNormals(BVH_NORMALS_FILE.c_str());
    if (!outNormals) {
        SLog(EError, "Unable to open BVH_DEBUG_FILE for BVH normals debug output: %s", BVH_NORMALS_FILE.c_str());
        return;
    }
    // Here we sample random points instead of fitting each subnode
    outNormals << "px,py,pz,nx,ny,nz\n";
    ref<Sampler> sampler = m_scene->getSampler();

    std::vector<Normal> normals;
    std::vector<Point3f> points;
    std::vector<Float> weights;
    
    for (int i=0; i<N_NORMALS_SAMPLES; i++) {
        Intersection its_xp_temp;
        Float pdf_xp;
        bool ok = sampleNodePrimitive(m_scene, node_idx_normals_debug, sampler->next1D(), sampler->next1D(), sampler->next2D(), its_xs.p, its_xp_temp, pdf_xp); 

        { // Only draw some looking to the right (+Y) to validate the vmf fitting
            // Float dot = its.shFrame.n.dot(Vector(0, 1, 0));
            Float dot = its_xp_temp.shFrame.cosTheta(its_xp_temp.shFrame.toLocal(Vector(0, 1, 0)));
            ok &= (dot > 0.f); // Only keep samples with normals roughly facing in +Y direction (this is just to validate the VMF fitting, we should see a strong mean direction in +Y)
            // ok &= (dot > 0.8f); // Another test of a more pointy distribution
        }

        if (ok) {
            outNormals << its_xp_temp.p.x << "," << its_xp_temp.p.y << "," << its_xp_temp.p.z << ","
                       << its_xp_temp.shFrame.n.x << "," << its_xp_temp.shFrame.n.y << "," << its_xp_temp.shFrame.n.z << "\n";
            normals.push_back(its_xp_temp.shFrame.n);
            // weights.push_back(1.f/pdf);
            weights.push_back(1.f); //  Uniform weight?? 
            points.push_back(its_xp_temp.p);
        }
    }
    Vector3f meanDir; 
    Float kappa;
    fitVMF(normals, meanDir, kappa);

    Gaussian3D fittedGaussian = fitWeightedGaussian(points, weights);

    outNormals << "# Fitted VMF mean direction: " << meanDir.x << "," << meanDir.y << "," << meanDir.z << "\n";
    outNormals << "# Fitted VMF kappa: " << kappa << "\n";
    outNormals << "# Fitted Gaussian mean: " << fittedGaussian.mean.x << "," << fittedGaussian.mean.y << "," << fittedGaussian.mean.z << "\n";
    outNormals << "# Fitted Gaussian covariance: " 
               << fittedGaussian.covariance(0,0) << "," << fittedGaussian.covariance(0,1) << "," << fittedGaussian.covariance(0,2) << ";"
               << fittedGaussian.covariance(1,0) << "," << fittedGaussian.covariance(1,1) << "," << fittedGaussian.covariance(1,2) << ";"
               << fittedGaussian.covariance(2,0) << "," << fittedGaussian.covariance(2,1) << "," << fittedGaussian.covariance(2,2) << "\n";
    SLog(EInfo, "Fitted VMF to sampled normals: mean direction = (%.3f, %.3f, %.3f), kappa = %.3f", 
        meanDir.x, meanDir.y, meanDir.z, kappa);
    // exit(0); // We exit here to only do the normal fitting debug for now
}
#endif
// USE_SGGX_AGGREGATES block end


// Note: plugin export lives in src/geometrybvh/geometrybvh.cpp

MTS_NAMESPACE_END