#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <algorithm>
#include <fstream>
#include <mitsuba/core/matrix.h>


#include <mitsuba/core/vmf.h>

MTS_NAMESPACE_BEGIN

#define DUMP_DEBUG_TO_CSV 1

#if DUMP_DEBUG_TO_CSV == 1

const std::string file = "bvh_debug.csv";

// --------------------------------------------------------------------
// Normals fitting into VMF:
const std::string normalsfile = "bvh_debug_normals.csv";
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

// -------------------------------------------------------------------------------

struct Gaussian3D {
    Point3f mean;
    Matrix3x3 covariance;
};

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

void GeometryBVH::print() {
    
    std::ofstream out(file.c_str());
    if (!out) {
        SLog(EError, "Unable to open file for BVH debug output: %s", file.c_str());
        return;
    }

    const BSDF *bsdf = m_primitives[0].bsdf;
    // out << bsdf->toString() << std::endl;


    SurfaceSample xs (Point(2.43008, -0.0490401, 0), Normal(0, 0, 1), bsdf); // Dummy surface sample at origin facing up
    // <translate x="-5" y="0" z="1"/>
    EmitterSample xe (Point(-5, 0.725691, 1.99827), Normal(1.,0.,0.f)); 

    out << "node_index,aabb_min_x,aabb_min_y,aabb_min_z,aabb_max_x,aabb_max_y,aabb_max_z,"
           "n_subtree_primitives,n_leaf_primitives,node_type,valid,surface_area,center_of_mass_x,center_of_mass_y,center_of_mass_z,importance,importance_gt\n";
    for (size_t i = 0; i < m_nodes.size(); ++i) {
        float importance = computeNodeImportance(
            i,
            this,
            m_scene,
            xs,
            xe,
            m_nodeInfos[i],
            m_nodes[i].bounds,
            1., m_solidAngleScale, m_variancePenaltyScale, 1.f // pass importance config params
        );

        float importance_gt = computeNodeImportanceGroundTruth(
            i,
            this,
            m_scene,
            xs,
            xe,
            m_nodeInfos[i],
            2048,
            false // use ground-truth importance with many samples
        );

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
    std::ofstream outNormals(normalsfile.c_str());
    if (!outNormals) {
        SLog(EError, "Unable to open file for BVH normals debug output: %s", normalsfile.c_str());
        return;
    }
    // Here we sample random points instead of fitting each subnode
    outNormals << "px,py,pz,nx,ny,nz\n";
    ref<Sampler> sampler = m_scene->getSampler();

    std::vector<Normal> normals;
    std::vector<Point3f> points;
    std::vector<Float> weights;
    
    for (int i=0; i<N_NORMALS_SAMPLES; i++) {
        Intersection its;
        Float pdf;
        bool ok = sampleNodePrimitive(node_idx_normals_debug, m_scene, xs.p, sampler->next2D(), its, pdf); 

        { // Only draw some looking to the right (+Y) to validate the vmf fitting
            // Float dot = its.shFrame.n.dot(Vector(0, 1, 0));
            Float dot = its.shFrame.cosTheta(its.shFrame.toLocal(Vector(0, 1, 0)));
            ok &= (dot > 0.f); // Only keep samples with normals roughly facing in +Y direction (this is just to validate the VMF fitting, we should see a strong mean direction in +Y)
            // ok &= (dot > 0.8f); // Another test of a more pointy distribution
        }

        if (ok) {
            outNormals << its.p.x << "," << its.p.y << "," << its.p.z << ","
                       << its.shFrame.n.x << "," << its.shFrame.n.y << "," << its.shFrame.n.z << "\n";
            normals.push_back(its.shFrame.n);
            // weights.push_back(1.f/pdf);
            weights.push_back(1.f); //  Uniform weight?? 
            points.push_back(its.p);
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
}

#endif

// Note: plugin export lives in src/geometrybvh/geometrybvh.cpp

MTS_NAMESPACE_END