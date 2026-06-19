#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/core/pmf.h>

#include <mitsuba/core/properties.h>
#include <mitsuba/core/plugin.h>   // <-- needed for PluginManager definition
#include <algorithm>
#include <array>
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

MTS_NAMESPACE_BEGIN

struct WeightedReservoir {
    Float m_Wy; // weight of sample Y
    PositionSamplingRecord m_Y; // the sample Y itself

    Float m_w_sum, // sum of weights of all samples 
          m_C; // Confidence weight

    WeightedReservoir() : m_Wy(0.f), m_Y(), m_w_sum(0.f), m_C(0.f) { }
    
    void update(Sampler* sampler, PositionSamplingRecord &x_i, const Float &weight_i, const Float &confidence_i) {
        m_w_sum += weight_i;
        m_C += confidence_i; // accumulate confidence
        Float acceptanceProb = weight_i / m_w_sum;
        if (sampler->next1D() < acceptanceProb) {
            // m_Wy = weight_i;
            m_Y = x_i;
        }
    }

};

// forward declaration of the faster but less accurate version of the lut importance function
Float computeNodeImportance_Unscented_LUT_fast(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility=false
);

// forward declaration of the original (slowest) LUT importance function
Float computeNodeImportance_Unscented_LUT_original(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility=false
);

Float computeNodeImportance(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe
) {
    if (bvh->m_debug1 <= .0001f) return 1.f; // hack to have an equal importance mode for testing

    // debug2 > 0.5: use specular-aware importance (diffuse + specular lobes)
    // debug2 <= 0.5 (default): use diffuse-only importance
    if (bvh->m_debug2 > 0.5f)
        return computeNodeImportance_Unscented_LUT_Specular(scene, bvh, nodeIndex, its_xs, pRec_xe);

    // debug3 mode selector:
    //   < 0.5   : original slowest version (local gaussian sigma-point generation + per-point LUT eval)
    //   [0.5,1.5): precomputed gaussian sigma points + per-point LUT eval
    //   >= 1.5  : fastest (precomputed gaussian sigma points + single LUT eval at mean)
    if (bvh->m_debug3 < 0.5f)
        return computeNodeImportance_Unscented_LUT_original(scene, bvh, nodeIndex, its_xs, pRec_xe);

    if (bvh->m_debug3 >= 1.5f) {
        return computeNodeImportance_Unscented_LUT_fast(scene, bvh, nodeIndex, its_xs, pRec_xe);
    }
    return computeNodeImportance_Unscented_LUT(scene, bvh, nodeIndex, its_xs, pRec_xe);
}


Float computeNodeImportance_MC(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe, 
    const int numSamples,
    bool checkVisibility
) 
{
    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    // First, initialize the constant data for the full node

    // 1. Get the spatial distribution: 
    Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
    Sampler *sampler = scene->getSampler(); // make a local copy of the sampler to avoid modifying the global one
    // 2. Get the NDF:
    Vector vmf_mu = nodeInfo.getVMFMeanDirection();
    Float kappa   = nodeInfo.getVMFKappa();
    VonMisesFisherDistr ndf(kappa);   

    // 3. BSDF 
    // All of this dummyshape and diffuse BSDF nonesense is to be able to pass the diffuse albedo of the node to the length2 node
    // When we have proxy functions here we won't need this 
    Properties shapeProps("sphere");
    shapeProps.setFloat("radius", 1e-6f);
    ref<Shape> dummyshape = static_cast<Shape *>(
        PluginManager::getInstance()->createObject(shapeProps)
    );

    Properties props("diffuse");
    props.setSpectrum("reflectance", nodeInfo.getMeanDiffuseAlbedo());

    ref<BSDF> bsdf = static_cast<BSDF *>(
        PluginManager::getInstance()->createObject(props)
    );
    bsdf->configure();
    dummyshape->setBSDF(bsdf);

    Intersection its_xp; // we will fill this with the sampled point on the node
    its_xp.shape = dummyshape.get(); // set the shape to something non-null to avoid issues in evalLength2Contribution
    its_xp.instance = nullptr;
    its_xp.t = 1.f; // nonesense so that it is considered valid..
    
    Float luminance = 0.f;

    Frame vmfFrame = Frame(normalize(vmf_mu)); // construct a frame from the VMF mean direction to rotate sampled normals

    for (int i=0;i<numSamples;i++) {
        Point3f sample3 = Point3f(sampler->next1D(), sampler->next1D(), sampler->next1D());
        // Sample spatial and normal components:
        Point3f xp = nodeGaussian.sample(sample3); // no pdf since we want to have the product of the gaussian distr baked in 
        if (std::isnan(xp.x) || std::isnan(xp.y) || std::isnan(xp.z) || std::isinf(xp.x) || std::isinf(xp.y) || std::isinf(xp.z)) {
            std::cerr << "Warning: Sampled NaN or Inf position for node importance:\n" << xp.toString() <<" Input sample: " << sample3.toString() <<  "!!!!!\n";
            std::cerr << "Node Gaussian mean: " << nodeGaussian.m_mean.toString() << ", cov: " << nodeGaussian.m_covariance.toString() << std::endl;
            std::cerr << " Node: " << nodeInfo.toString() << std::endl;
            continue;
        }
        Normal n_vmf_local = normalize(ndf.sample(sampler->next2D()));
        Normal n_vmf = vmfFrame.toWorld(n_vmf_local);

        // Fill intersection
        its_xp.p = xp; // Point from gaussian
        if (std::isnan(its_xp.p.x) || std::isnan(its_xp.p.y) || std::isnan(its_xp.p.z)) {
            std::cerr << "Warning: Sampled NaN position for node importance. Skipping this sample.\n";
        }
        if (std::isnan(n_vmf.x) || std::isnan(n_vmf.y) || std::isnan(n_vmf.z)) {
            std::cerr << "Warning: Sampled NaN normal for node importance. Skipping this sample.\n";
            std::cerr << ndf.toString() << std::endl;
        }
        its_xp.geoFrame = Frame(n_vmf); // Normal from vMF
        its_xp.shFrame = its_xp.geoFrame;
        its_xp.wi = normalize(its_xs.p - xp); // from xp to xs 

        Spectrum contribution = evalLength2Contribution(scene, its_xs, its_xp, pRec_xe, checkVisibility);
        // Note about pdfs:
        // f = evalLength2(X, N) * p_gaussian(X) * p_vmf(N) * albedo
        // No division by pdfs here, bc we do want those distributions to weigh in the importance
        luminance += contribution.getLuminance();
    }

    if (luminance < 0.f || std::isnan(luminance)) {
        std::cerr << "Warning: Negative or NaN luminance in node importance. Setting to 0.\n";
    }

    // Multiply by area of the node:
    return nodeInfo.surfaceArea * luminance / static_cast<Float>(numSamples);
}


Float computeNodeImportance_MC_LUT(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe, 
    const int numSamples,
    bool checkVisibility
) 
{
    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    const VMFLUT &vmfLUT = bvh->getVMFLUT();
    // First, initialize the constant data for the full node

    // 1. Get the spatial distribution: 
    Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
    Sampler *sampler = scene->getSampler(); // make a local copy of the sampler to avoid modifying the global one
    // 2. Get the NDF:
    Vector vmf_mu = nodeInfo.getVMFMeanDirection();
    Float kappa   = nodeInfo.getVMFKappa();
    VonMisesFisherDistr ndf(kappa);   

    // 3. No BSDF thx to VMF lut !

    // Intersection its_xp; // - no longer needed !
    Float luminance = 0.f;

    Frame vmfFrame = Frame(normalize(vmf_mu)); // construct a frame from the VMF mean direction to rotate sampled normals

    for (int i=0;i<numSamples;i++) {
        Point3f sample3 = Point3f(sampler->next1D(), sampler->next1D(), sampler->next1D());
        // Sample spatial and normal components:
        Point xp = nodeGaussian.sample(sample3); // no pdf since we want to have the product of the gaussian distr baked in 

        // Normal n_vmf_local = normalize(ndf.sample(sampler->next2D()));
        // Normal n_vmf = vmfFrame.toWorld(n_vmf_local);

        // Fill intersection - No intersection needed with VMF Lut

        // -------------------------------------------------------------
        // Eval the contribution
        Spectrum contribution(0.0f);
        {
            // Directions, from xs (outside surface) to xp (gaussian point) 
            Vector d_sp = xp - its_xs.p;
            Float dist_sp = d_sp.length();
            d_sp /= dist_sp;
            
            Vector d_pe = pRec_xe.p - xp; // from xp (gaussian) to xe (emitter) 
            Float dist_pe = d_pe.length();
            d_pe /= dist_pe;
            // 1. Evaluate visibility if needed
            // 1.1. Visibility between xs and xp
            Ray shadowRay_sp(its_xs.p, d_sp, Epsilon, dist_sp * (1.0f - ShadowEpsilon), 0.f);
            if (checkVisibility && scene->rayIntersect(shadowRay_sp)) {
                contribution = Spectrum(0.0f);
                continue;
            }
            // 1.2. Visibility between xp and xe
            Ray shadowRay_pe(xp, d_pe, Epsilon, dist_pe * (1.0f - ShadowEpsilon), 0.f);
            if (checkVisibility && scene->rayIntersect(shadowRay_pe)) {
                contribution = Spectrum(0.0f);
                continue;
            }

            // 2. BSDF at xs (outside surface)
            // Evaluate the xs bounce: 
            Vector l_xs_to_xp = its_xs.toLocal(d_sp);
            BSDFSamplingRecord bRec_xs(its_xs, l_xs_to_xp);
            const BSDF *bsdf_xs = its_xs.getBSDF();
            if (!bsdf_xs) {
                std::cerr << "!!!! No BSDF at xs intersection, zero contribution.\n";
                contribution = Spectrum(0.0f);
                continue;
            }
            Spectrum f_xs = bsdf_xs->eval(bRec_xs);
            
            // 2. BSDF at xp (node with gaussian+vmf)
            // First get the local directions in the vmf frame to query the LUT:
            Vector3 w1_local = vmfFrame.toLocal(-d_sp); // direction from xp to xs in local frame
            Vector3 w2_local = vmfFrame.toLocal(d_pe);  // direction from xp to xe in local frame
            float vmf_convolution = vmfLUT.eval(w1_local, w2_local, kappa);  
            
            Spectrum f_xp = nodeInfo.getMeanDiffuseAlbedo() * (1.0f / M_PI) * vmf_convolution; 

            if (f_xs.isZero() || f_xp.isZero()) {
                contribution = Spectrum(0.0f);
                continue;
            }

            // 3. Evaluate emitter radiance at xe in the direction of xp
            const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
            DirectionSamplingRecord dRec_xe(-d_pe);
            dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
            Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);
            
            if (Le.isZero()) {
                // std::cout << "Zero contribution due to emitter evaluation.\n";
                contribution = Spectrum(0.0f);
                continue;
            }
            
            // 4. Geometry term
            // Float cos_theta_xs_out = std::abs(dot(its_xs.geoFrame.n, d_sp));
            // Float cos_theta_xp_in = std::abs(dot(its_xp.geoFrame.n, -d_sp));
            // Float cos_theta_xp_out = std::abs(dot(its_xp.geoFrame.n, d_pe));
            // Float cos_theta_xe_in = std::abs(dot(pRec_xe.n, -d_pe));


            Float G_sp = (1.f) / (dist_sp * dist_sp); // Cosines are in the LUT in this case (vmf_convolution), so geometry term is just the inverse squared distance
            Float G_pe = (1.f) / (dist_pe * dist_pe);
            
            contribution = (f_xs * G_sp * f_xp * G_pe * Le);
        }
        
        // ---------------------------------------------------------------
        // Note about pdfs:
        // f = evalLength2(X, N) * p_gaussian(X) * p_vmf(N) * albedo
        // No division by pdfs here, bc we do want those distributions to weigh in the importance
        luminance += contribution.getLuminance();
    }
    // Multiply by area of the node:
    return nodeInfo.surfaceArea * luminance / static_cast<Float>(numSamples);
}

Float computeNodeImportanceGroundTruth(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    int numSamples,
    bool checkVisibility
) {
    if (!scene || numSamples <= 0)
        return 0.0f;

    if (!bvh || !bvh->isBuilt())
        return 0.0f;

    Float contributionSum = 0.f;
    
    Intersection its_xp_temp;
    float pdf_xp_temp;
    int count_ok = 0;
    for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
    {
        // std::cout << "Sampling node " << nodeIndex << ", sample " << sampleIdx + 1 << " / " << numSamples << std::endl;
        bool ok = bvh->sampleNodePrimitive(
            scene,
            nodeIndex,
            scene->getSampler(),
            its_xs.p,
            its_xp_temp,
            pdf_xp_temp,
            false // return_solidangle_pdf, we want area pdf for ground truth contribution
        );

        if (!ok || pdf_xp_temp <= 0.f) {
            // std::cout << "Invalid sample for node " << bvh->getNode(nodeIndex).toString() << std::endl; 
            continue; // Sample failed, skip contribution
        }
        
        count_ok++;
        
        Spectrum contribution = evalLength2Contribution(scene, its_xs, its_xp_temp, pRec_xe, checkVisibility);
        contributionSum += contribution.getLuminance() / pdf_xp_temp;
    }
    if (count_ok == 0) {
        std::cout << "[computeNodeImportanceGroundTruth] Node " << nodeIndex << ": No valid samples found out of " << numSamples << ". Returning zero importance." << std::endl;
        return 0.f;
    }
    
    float luminanceContributionAvg = contributionSum / static_cast<float>(numSamples);
    
    return luminanceContributionAvg;
}

MTS_EXPORT_RENDER Float computeNodeImportance_Unscented_MC(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe, 
    const int numSamples,
    bool checkVisibility
) {

    if (!scene || !bvh || !bvh->isBuilt() || numSamples <= 0)
        return 0.0f;

    if (nodeIndex < 0 || (size_t) nodeIndex >= bvh->getNodeCount())
        return 0.0f;

    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon)
        return 0.0f;

    auto computeCholeskyLower = [](const Matrix3x3 &covariance) {
        Matrix3x3 L(0.0f);
        const Float eps = 1e-6f;

        L(0,0) = std::sqrt(std::max(covariance(0,0) + eps, (Float) 0.0f));
        if (L(0,0) <= eps)
            L(0,0) = eps;

        L(1,0) = covariance(1,0) / L(0,0);
        L(1,1) = std::sqrt(std::max(covariance(1,1) + eps - L(1,0)*L(1,0), (Float) 0.0f));
        if (L(1,1) <= eps)
            L(1,1) = eps;

        L(2,0) = covariance(2,0) / L(0,0);
        L(2,1) = (covariance(2,1) - L(2,0)*L(1,0)) / L(1,1);
        L(2,2) = std::sqrt(std::max(covariance(2,2) + eps - L(2,0)*L(2,0) - L(2,1)*L(2,1), (Float) 0.0f));
        if (L(2,2) <= eps)
            L(2,2) = eps;

        return L;
    };

    auto generateUnscentedPoints = [&](const Gaussian3D &gaussian) {
        struct WeightedPoint {
            Point p;
            Float w;
        };
        std::array<WeightedPoint, 7> points;

        const Float Ldim = 3.0f;
        const Float lambda = std::sqrt((Float) 3.0f);
        const Float sqrtScale = std::sqrt(Ldim + lambda);
        const Float w0 = lambda / (Ldim + lambda);
        const Float wi = 1.0f / (2.0f * (Ldim + lambda));

        Matrix3x3 S = computeCholeskyLower(gaussian.m_covariance);
        Point mean = Point(gaussian.m_mean);

        points[0] = { mean, w0 };
        for (int i = 0; i < 3; ++i) {
            Vector axisOffset = S.col(i) * sqrtScale;
            points[1 + 2*i]     = { mean + axisOffset, wi };
            points[1 + 2*i + 1] = { mean - axisOffset, wi };
        }

        return points;
    };

    Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
    const auto sigmaPoints = generateUnscentedPoints(nodeGaussian);

    Vector vmf_mu = nodeInfo.getVMFMeanDirection();
    if (vmf_mu.lengthSquared() <= Epsilon)
        vmf_mu = Vector(0.0f, 0.0f, 1.0f);
    vmf_mu = normalize(vmf_mu);

    Float kappa = std::max((Float) 0.0f, nodeInfo.getVMFKappa());
    VonMisesFisherDistr ndf(kappa);
    Frame vmfFrame(vmf_mu);

    Sampler *sampler = scene->getSampler();
    if (!sampler)
        return 0.0f;

    Properties shapeProps("sphere");
    shapeProps.setFloat("radius", 1e-6f);
    ref<Shape> dummyshape = static_cast<Shape *>(
        PluginManager::getInstance()->createObject(shapeProps)
    );

    Properties props("diffuse");
    props.setSpectrum("reflectance", nodeInfo.getMeanDiffuseAlbedo());

    ref<BSDF> bsdf = static_cast<BSDF *>(
        PluginManager::getInstance()->createObject(props)
    );
    bsdf->configure();
    dummyshape->setBSDF(bsdf);

    Intersection its_xp;
    its_xp.shape = dummyshape.get();
    its_xp.instance = nullptr;
    its_xp.t = 1.f;

    Float weightedLuminance = 0.0f;
    for (size_t pointIdx = 0; pointIdx < sigmaPoints.size(); ++pointIdx) {
        const Point &xp = sigmaPoints[pointIdx].p;
        const Float pointWeight = sigmaPoints[pointIdx].w;

        if (std::isnan(xp.x) || std::isnan(xp.y) || std::isnan(xp.z))
            continue;

        Float pointLuminance = 0.0f;
        for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx) {
            Normal n_vmf_local = normalize(ndf.sample(sampler->next2D()));
            Normal n_vmf = vmfFrame.toWorld(n_vmf_local);

            its_xp.p = xp;
            its_xp.geoFrame = Frame(n_vmf);
            its_xp.shFrame = its_xp.geoFrame;
            its_xp.wi = normalize(its_xs.p - xp);

            Spectrum contribution = evalLength2Contribution(scene, its_xs, its_xp, pRec_xe, checkVisibility);
            Float lum = contribution.getLuminance();
            if (std::isnan(lum) || lum < 0.0f)
                continue;

            pointLuminance += lum;
        }

        pointLuminance /= static_cast<Float>(numSamples);
        weightedLuminance += pointWeight * pointLuminance;
    }

    if (std::isnan(weightedLuminance) || weightedLuminance < 0.0f)
        return 0.0f;

    return nodeInfo.surfaceArea * weightedLuminance;
}

// TLDR: this "fast" version, taking some stuff out of the loop, is basically just as fast as the other one
Float computeNodeImportance_Unscented_LUT_fast(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility
) {
    // if (nodeIndex < 0 || (size_t) nodeIndex >= bvh->getNodeCount()) return 0.0f;
    // const BVHNodeInfo &nodeInfotest = bvh->getNodeInfo(nodeIndex);
    // return nodeInfotest.surfaceArea; 


    #ifdef DO_TIMING_IMPORTANCE
    auto &importanceTimer = bvh->getImportanceTimer();  
    float t = importanceTimer.timer->getSeconds();
    if (t > 2.f && t < 2.2f) {
        std::cout << "doing timings, this will likely be slow! Also, remember launching with -p 1 for a single thread (I will shut up after 5s)...\n";
    }
    uint64_t startTime = importanceTimer.getNanoSeconds();
    #endif 
    
    // --- 1. Early Exits & Hoisted Checks ---
    if (!scene || !bvh || !bvh->isBuilt()) return 0.0f;
    if (nodeIndex < 0 || (size_t) nodeIndex >= bvh->getNodeCount()) return 0.0f;

    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon) return 0.0f;

    // Hoist BSDF and Emitter fetch! If these fail, exit immediately.
    const BSDF *bsdf_xs = its_xs.getBSDF();
    if (!bsdf_xs) return 0.0f;

    const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
    if (!emitter) return 0.0f;

    const VMFLUT &vmfLUT = bvh->getVMFLUT();
    Float kappa = std::max((Float) 0.0f, nodeInfo.getVMFKappa());
    KappaQuery kappaQuery = vmfLUT.prepareKappa(kappa);

    Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
    Point mean = Point(nodeGaussian.m_mean);

    // --- 3. Spatial Clustering & Unscented Setup ---
    Float dist_xs_sq = (mean - its_xs.p).lengthSquared();
    Float dist_xe_sq = (mean - pRec_xe.p).lengthSquared();
    Float min_dist_sq = std::min(dist_xs_sq, dist_xe_sq);

    Float covTrace = nodeGaussian.m_covariance(0,0) + 
                     nodeGaussian.m_covariance(1,1) + 
                     nodeGaussian.m_covariance(2,2);

    bool isSpatiallyClustered = covTrace < (bvh->getGaussianSAThreshold() * min_dist_sq);

    const std::array<BVHNodeInfo::WeightedPoint, 7> &precomputedSigmaPoints = nodeInfo.getUnscentedPoints();
    std::array<BVHNodeInfo::WeightedPoint, 7> sigmaPoints;
    size_t numPointsToEval = 1;

    if (isSpatiallyClustered) {
        sigmaPoints[0] = { mean, 1.0f };
    } else {
        sigmaPoints = precomputedSigmaPoints;
        numPointsToEval = 7;
    }

    #ifdef DO_TIMING_IMPORTANCE
    if (importanceTimer.timer->getSeconds() > 2.f) {
        importanceTimer.addElapsed("init", startTime); 
        importanceTimer.addCounter();
    }
    #endif 

    // --- 4. Directional Decoupling (Evaluate Complex Math ONCE at the Mean) ---
    #ifdef DO_TIMING_IMPORTANCE
    startTime = importanceTimer.getNanoSeconds();
    #endif

    Vector d_sp_mean = mean - its_xs.p;
    Float dist_sp_mean = d_sp_mean.length();
    if (dist_sp_mean > Epsilon) d_sp_mean /= dist_sp_mean;

    Vector d_pe_mean = pRec_xe.p - mean;
    Float dist_pe_mean = d_pe_mean.length();
    if (dist_pe_mean > Epsilon) d_pe_mean /= dist_pe_mean;

    // Evaluate BSDF
    Vector l_xs_to_xp_mean = its_xs.toLocal(d_sp_mean);
    BSDFSamplingRecord bRec_xs(its_xs, l_xs_to_xp_mean);
    Spectrum f_xs = bsdf_xs->eval(bRec_xs);

    // Evaluate vMF LUT
    Vector vmf_mu = nodeInfo.getVMFMeanDirection();
    if (vmf_mu.lengthSquared() <= Epsilon) vmf_mu = Vector(0.0f, 0.0f, 1.0f);
    vmf_mu = normalize(vmf_mu);
    Frame vmfFrame(vmf_mu);

    Vector3 w1_local = vmfFrame.toLocal(-d_sp_mean);
    Vector3 w2_local = vmfFrame.toLocal(d_pe_mean);
    float vmf_convolution = vmfLUT.evalFast(w1_local, w2_local, kappaQuery);
    Spectrum f_xp = nodeInfo.getMeanDiffuseAlbedo() * (1.0f / M_PI) * vmf_convolution;

    // Evaluate Emitter
    DirectionSamplingRecord dRec_xe(-d_pe_mean);
    dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
    Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);

    // Compute Decoupled Base Luminance
    Spectrum base_contribution = f_xs * f_xp * Le;
    Float base_lum = base_contribution.getLuminance();

    #ifdef DO_TIMING_IMPORTANCE
    importanceTimer.addElapsed("directional_decoupled_eval", startTime);
    #endif

    // If the directional components amount to nothing, skip the UT entirely
    if (base_lum <= Epsilon || std::isnan(base_lum)) {
        return 0.0f;
    }

    // --- 5. Unscented Transform Loop (Only Geometry & Visibility vary per point) ---
    Float weightedLuminance = 0.0f;

    for (size_t pointIdx = 0; pointIdx < numPointsToEval; ++pointIdx) {

        #ifdef DO_TIMING_IMPORTANCE
        startTime = importanceTimer.getNanoSeconds();
        #endif 

        const Point &xp = sigmaPoints[pointIdx].p;
        const Float pointWeight = sigmaPoints[pointIdx].w;

        if (std::isnan(xp.x) || std::isnan(xp.y) || std::isnan(xp.z))
            continue;

        // Compute Distances
        Vector d_sp = xp - its_xs.p;
        Float dist_sp_sq = d_sp.lengthSquared();
        if (dist_sp_sq <= Epsilon) continue;

        Vector d_pe = pRec_xe.p - xp;
        Float dist_pe_sq = d_pe.lengthSquared();
        if (dist_pe_sq <= Epsilon) continue;

        // Visibility Checks 
        if (checkVisibility) {
            Float dist_sp = std::sqrt(dist_sp_sq);
            Ray shadowRay_sp(its_xs.p, d_sp / dist_sp, Epsilon, dist_sp * (1.0f - ShadowEpsilon), 0.f);
            if (scene->rayIntersect(shadowRay_sp)) continue;

            Float dist_pe = std::sqrt(dist_pe_sq);
            Ray shadowRay_pe(xp, d_pe / dist_pe, Epsilon, dist_pe * (1.0f - ShadowEpsilon), 0.f);
            if (scene->rayIntersect(shadowRay_pe)) continue;
        }

        #ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("visibility", startTime);
        #endif 

        // Apply only the geometric falloff to the decoupled base luminance
        Float G_total = 1.0f / (dist_sp_sq * dist_pe_sq);
        Float lum = base_lum * G_total;

        if (std::isnan(lum) || lum < 0.0f)
            continue;

        weightedLuminance += pointWeight * lum;
    }

    if (std::isnan(weightedLuminance) || weightedLuminance < 0.0f)
        return 0.0f;

    #ifdef DO_TIMING_IMPORTANCE
    if (importanceTimer.timer->getSeconds() > 10.f) {
        std::cout << "Importance computation timings:\n";
        importanceTimer.report();
        exit(0); 
    }
    #endif 

    return nodeInfo.surfaceArea * weightedLuminance;
}

// Original LUT version: local unscented sigma-point generation + per-point VMF LUT evaluation
Float computeNodeImportance_Unscented_LUT_original(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility
) {

    #ifdef DO_TIMING_IMPORTANCE
    auto &importanceTimer = bvh->getImportanceTimer();
    float t = importanceTimer.timer->getSeconds();
    if (t > 2.f && t < 2.2f) {
        std::cout << "doing timings, this will likely be slow! Also, remember launching with -p 1 for a single thread (I will shut up after 5s)...\n";
    }

    uint64_t startTime = importanceTimer.getNanoSeconds();
    #endif

    if (!scene || !bvh || !bvh->isBuilt())
        return 0.0f;

    if (nodeIndex < 0 || (size_t) nodeIndex >= bvh->getNodeCount())
        return 0.0f;

    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon)
        return 0.0f;

    auto computeCholeskyLower = [](const Matrix3x3 &covariance) {
        Matrix3x3 L(0.0f);
        const Float eps = 1e-6f;

        L(0,0) = std::sqrt(std::max(covariance(0,0) + eps, (Float) 0.0f));
        if (L(0,0) <= eps)
            L(0,0) = eps;

        L(1,0) = covariance(1,0) / L(0,0);
        L(1,1) = std::sqrt(std::max(covariance(1,1) + eps - L(1,0)*L(1,0), (Float) 0.0f));
        if (L(1,1) <= eps)
            L(1,1) = eps;

        L(2,0) = covariance(2,0) / L(0,0);
        L(2,1) = (covariance(2,1) - L(2,0)*L(1,0)) / L(1,1);
        L(2,2) = std::sqrt(std::max(covariance(2,2) + eps - L(2,0)*L(2,0) - L(2,1)*L(2,1), (Float) 0.0f));
        if (L(2,2) <= eps)
            L(2,2) = eps;

        return L;
    };

    struct WeightedPoint {
        Point p;
        Float w;
    };

    auto generateUnscentedPoints = [&](const Gaussian3D &gaussian) {
        std::array<WeightedPoint, 7> points;

        const Float Ldim = 3.0f;
        const Float lambda = std::sqrt((Float) 3.0f);
        const Float sqrtScale = std::sqrt(Ldim + lambda);
        const Float w0 = lambda / (Ldim + lambda);
        const Float wi = 1.0f / (2.0f * (Ldim + lambda));

        Matrix3x3 S = computeCholeskyLower(gaussian.m_covariance);
        Point mean = Point(gaussian.m_mean);

        points[0] = { mean, w0 };
        for (int i = 0; i < 3; ++i) {
            Vector axisOffset = S.col(i) * sqrtScale;
            points[1 + 2*i]     = { mean + axisOffset, wi };
            points[1 + 2*i + 1] = { mean - axisOffset, wi };
        }

        return points;
    };

    const VMFLUT &vmfLUT = bvh->getVMFLUT();

    Float kappa = std::max((Float) 0.0f, nodeInfo.getVMFKappa());
    KappaQuery kappaQuery = vmfLUT.prepareKappa(kappa);

    Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
    Point mean = Point(nodeGaussian.m_mean);

    Float dist_xs_sq = (mean - its_xs.p).lengthSquared();
    Float dist_xe_sq = (mean - pRec_xe.p).lengthSquared();
    Float min_dist_sq = std::min(dist_xs_sq, dist_xe_sq);

    Float covTrace = nodeGaussian.m_covariance(0,0) +
                     nodeGaussian.m_covariance(1,1) +
                     nodeGaussian.m_covariance(2,2);

    bool isSpatiallyClustered = covTrace < (bvh->getGaussianSAThreshold() * min_dist_sq);

    std::array<WeightedPoint, 7> sigmaPoints;
    size_t numPointsToEval = 1;

    if (isSpatiallyClustered) {
        sigmaPoints[0] = { mean, 1.0f };
    } else {
        sigmaPoints = generateUnscentedPoints(nodeGaussian);
        numPointsToEval = 7;
    }

    Vector vmf_mu = nodeInfo.getVMFMeanDirection();
    if (vmf_mu.lengthSquared() <= Epsilon)
        vmf_mu = Vector(0.0f, 0.0f, 1.0f);
    vmf_mu = normalize(vmf_mu);
    Frame vmfFrame(vmf_mu);

    #ifdef DO_TIMING_IMPORTANCE
    if (importanceTimer.timer->getSeconds() > 2.f) {
        importanceTimer.addElapsed("init", startTime);
        importanceTimer.addCounter();
    }
    #endif

    Float weightedLuminance = 0.0f;
    for (size_t pointIdx = 0; pointIdx < numPointsToEval; ++pointIdx) {

    #ifdef DO_TIMING_IMPORTANCE
        startTime = importanceTimer.getNanoSeconds();
    #endif

        const Point &xp = sigmaPoints[pointIdx].p;
        const Float pointWeight = sigmaPoints[pointIdx].w;

        if (std::isnan(xp.x) || std::isnan(xp.y) || std::isnan(xp.z))
            continue;

        Spectrum contribution(0.0f);

        Vector d_sp = xp - its_xs.p;
        Float dist_sp = d_sp.length();
        if (dist_sp <= Epsilon)
            continue;
        d_sp /= dist_sp;

        Vector d_pe = pRec_xe.p - xp;
        Float dist_pe = d_pe.length();
        if (dist_pe <= Epsilon)
            continue;
        d_pe /= dist_pe;

        Ray shadowRay_sp(its_xs.p, d_sp, Epsilon, dist_sp * (1.0f - ShadowEpsilon), 0.f);
        if (checkVisibility && scene->rayIntersect(shadowRay_sp))
            continue;

        Ray shadowRay_pe(xp, d_pe, Epsilon, dist_pe * (1.0f - ShadowEpsilon), 0.f);
        if (checkVisibility && scene->rayIntersect(shadowRay_pe))
            continue;

    #ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("visibility", startTime);
        startTime = importanceTimer.getNanoSeconds();
    #endif

        Vector l_xs_to_xp = its_xs.toLocal(d_sp);
        BSDFSamplingRecord bRec_xs(its_xs, l_xs_to_xp);
        const BSDF *bsdf_xs = its_xs.getBSDF();
        if (!bsdf_xs)
            continue;

        Spectrum f_xs = bsdf_xs->eval(bRec_xs);

    #ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("bsdf", startTime);
        startTime = importanceTimer.getNanoSeconds();
    #endif
        Vector3 w1_local = vmfFrame.toLocal(-d_sp);
        Vector3 w2_local = vmfFrame.toLocal(d_pe);
        float vmf_convolution = vmfLUT.evalFast(w1_local, w2_local, kappaQuery);

    #ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("vmfeval", startTime);
        startTime = importanceTimer.getNanoSeconds();
    #endif

        Spectrum f_xp = nodeInfo.getMeanDiffuseAlbedo() * (1.0f / M_PI) * vmf_convolution;

        if (f_xs.isZero() || f_xp.isZero())
            continue;

        const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
        if (!emitter)
            continue;

        DirectionSamplingRecord dRec_xe(-d_pe);
        dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
        Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);
        if (Le.isZero())
            continue;

        Float G_sp = 1.0f / (dist_sp * dist_sp);
        Float G_pe = 1.0f / (dist_pe * dist_pe);
        contribution = f_xs * G_sp * f_xp * G_pe * Le;

        Float lum = contribution.getLuminance();
        if (std::isnan(lum) || lum < 0.0f)
            continue;

    #ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("emitter", startTime);
        startTime = importanceTimer.getNanoSeconds();
    #endif
        weightedLuminance += pointWeight * lum;
    }

    if (std::isnan(weightedLuminance) || weightedLuminance < 0.0f)
        return 0.0f;

    #ifdef DO_TIMING_IMPORTANCE
    if (importanceTimer.timer->getSeconds() > 10.f) {
        std::cout << "Importance computation timings:\n";
        importanceTimer.report();
        exit(0);
    }
    #endif

    return nodeInfo.surfaceArea * weightedLuminance;
}

MTS_EXPORT_RENDER Float computeNodeImportance_Unscented_LUT(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility
) {
    
    #ifdef DO_TIMING_IMPORTANCE
    auto &importanceTimer = bvh->getImportanceTimer();  
    float t = importanceTimer.timer->getSeconds();
    if (t>2.f && t < 2.2f) {
        std::cout << "doing timings, this will likely be slow! Also, remember launching with -p 1 for a single thread (I will shut up after 5s)...\n";

        // Example usage:
        // importanceTimer.addTime("thing1", 10.f);
        // importanceTimer.addTime("another one", 20.f);
        // importanceTimer.report();

    }

    // define timers:
    uint64_t startTime = importanceTimer.getNanoSeconds();

    #endif 
    
    if (!scene || !bvh || !bvh->isBuilt())
        return 0.0f;

    if (nodeIndex < 0 || (size_t) nodeIndex >= bvh->getNodeCount())
        return 0.0f;

    const uint32_t ablationMask = bvh->m_ablationMask;
    const bool ablateVMF        = (ablationMask & GeometryBVH::AblateVMF) != 0;
    const bool ablateGaussian   = (ablationMask & GeometryBVH::AblateGaussian) != 0;
    const bool ablateCosineXs   = (ablationMask & GeometryBVH::AblateCosineXs) != 0;
    const bool ablateCosineXe   = (ablationMask & GeometryBVH::AblateCosineXe) != 0;
    const bool ablateDistanceXs = (ablationMask & GeometryBVH::AblateDistanceXs) != 0;
    const bool ablateDistanceXe = (ablationMask & GeometryBVH::AblateDistanceXe) != 0;
    const bool ablateAlbedo     = (ablationMask & GeometryBVH::AblateAlbedo) != 0;

    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon)
        return 0.0f;

    const VMFLUT &vmfLUT = bvh->getVMFLUT();

    // precompute kappa here, compare w/ the unoptimized version both output and time
    Float kappa = std::max((Float) 0.0f, nodeInfo.getVMFKappa());
    KappaQuery kappaQuery = vmfLUT.prepareKappa(kappa);

    
    const BVHNode &node = bvh->getNode(nodeIndex);
    Point mean;
    if (ablateGaussian) {
        // Gaussian ablation: ignore node Gaussian entirely and use AABB center as spatial proxy.
        mean = node.bounds.getCenter();
    } else {
        Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
        mean = Point(nodeGaussian.m_mean);
    }

    // --- 3. Spatial Clustering & Unscented Setup ---
    Float dist_xs_sq = (mean - its_xs.p).lengthSquared();
    Float dist_xe_sq = (mean - pRec_xe.p).lengthSquared();
    Float min_dist_sq = std::min(dist_xs_sq, dist_xe_sq);

    Float covTrace = 0.0f;
    if (!ablateGaussian) {
        Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
        covTrace = nodeGaussian.m_covariance(0,0) + 
                   nodeGaussian.m_covariance(1,1) + 
                   nodeGaussian.m_covariance(2,2);
    }

    bool isSpatiallyClustered = ablateGaussian || (covTrace < (bvh->getGaussianSAThreshold() * min_dist_sq));

    const std::array<BVHNodeInfo::WeightedPoint, 7> &precomputedSigmaPoints = nodeInfo.getUnscentedPoints();
    std::array<BVHNodeInfo::WeightedPoint, 7> sigmaPoints;
    size_t numPointsToEval = 1;

    if (isSpatiallyClustered) {
        sigmaPoints[0] = { mean, 1.0f };
    } else {
        sigmaPoints = precomputedSigmaPoints;
        numPointsToEval = 7;
    }

    Vector vmf_mu = nodeInfo.getVMFMeanDirection();
    if (vmf_mu.lengthSquared() <= Epsilon)
        vmf_mu = Vector(0.0f, 0.0f, 1.0f);
    vmf_mu = normalize(vmf_mu);
    Frame vmfFrame(vmf_mu);

    

    // --- Optimization 2: Isotropic vMF Check ---
    // const Float kappaThreshold = 10.f; // TODO: VALUE ? 
    // bool isIsotropic = kappa < kappaThreshold;
    // float cached_vmf_convolution = -1.0f;
    // if (isIsotropic) { // TODO - im focusing on the gaussian optimization first
    //     // For isotropic distributions, the convolution is constant and can be computed once
    //     Vector3 w1_local = vmfFrame.toLocal(-normalize(mean - its_xs.p));
    //     Vector3 w2_local = vmfFrame.toLocal(normalize(pRec_xe.p - mean));
    //     cached_vmf_convolution = vmfLUT.eval(w1_local, w2_local, kappa);
    // }

    #ifdef DO_TIMING_IMPORTANCE
    if (importanceTimer.timer->getSeconds() > 2.f) {
        // importanceTimer.addTime("init", importanceTimer.getNanoSeconds() - startTime);
        importanceTimer.addElapsed("init", startTime); // ^ same
        importanceTimer.addCounter();
    }
    #endif 

    Float weightedLuminance = 0.0f;
    for (size_t pointIdx = 0; pointIdx < numPointsToEval; ++pointIdx) {

    #ifdef DO_TIMING_IMPORTANCE
        startTime = importanceTimer.getNanoSeconds();
    #endif 

        const Point &xp = sigmaPoints[pointIdx].p;
        const Float pointWeight = sigmaPoints[pointIdx].w;

        if (std::isnan(xp.x) || std::isnan(xp.y) || std::isnan(xp.z))
            continue;

        Spectrum contribution(0.0f);

        Vector d_sp = xp - its_xs.p;
        Float dist_sp = d_sp.length();
        if (dist_sp <= Epsilon)
            continue;
        d_sp /= dist_sp;

        Vector d_pe = pRec_xe.p - xp;
        Float dist_pe = d_pe.length();
        if (dist_pe <= Epsilon)
            continue;
        d_pe /= dist_pe;

        Ray shadowRay_sp(its_xs.p, d_sp, Epsilon, dist_sp * (1.0f - ShadowEpsilon), 0.f);
        if (checkVisibility && scene->rayIntersect(shadowRay_sp))
            continue;

        Ray shadowRay_pe(xp, d_pe, Epsilon, dist_pe * (1.0f - ShadowEpsilon), 0.f);
        if (checkVisibility && scene->rayIntersect(shadowRay_pe))
            continue;

        
#ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("visibility", startTime);
        startTime = importanceTimer.getNanoSeconds();
#endif 

        Spectrum f_xs(1.0f);
        if (!ablateCosineXs) {
            Vector l_xs_to_xp = its_xs.toLocal(d_sp);
            BSDFSamplingRecord bRec_xs(its_xs, l_xs_to_xp);
            const BSDF *bsdf_xs = its_xs.getBSDF();
            if (!bsdf_xs)
                continue;
            f_xs = bsdf_xs->eval(bRec_xs);
        }

#ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("bsdf", startTime); 
        startTime = importanceTimer.getNanoSeconds();
#endif 
        Vector3 w1_local = vmfFrame.toLocal(-d_sp);
        Vector3 w2_local = vmfFrame.toLocal(d_pe);
        float vmf_convolution = ablateVMF ? 1.0f : vmfLUT.evalFast(w1_local, w2_local, kappaQuery);
        // Old, slower version: vmfLUT.eval(w1_local, w2_local, kappa);

#ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("vmfeval", startTime); 
        startTime = importanceTimer.getNanoSeconds();
#endif 
        // std::cout << "VMF convolution: " << vmf_convolution << ", fast version: " << vmf_convolution_fast << std::endl;
        // Uncomment to check correctness. Fast returns identical values to original

        Spectrum nodeAlbedo = ablateAlbedo ? Spectrum(1.0f) : nodeInfo.getMeanDiffuseAlbedo();
        Spectrum f_xp = nodeAlbedo * (1.0f / M_PI) * vmf_convolution;

        
        if (f_xs.isZero() || f_xp.isZero())
            continue;

        const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
        if (!emitter)
            continue;

        Spectrum Le = emitter->evalPosition(pRec_xe);
        if (!ablateCosineXe) {
            DirectionSamplingRecord dRec_xe(-d_pe);
            dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
            Le *= emitter->evalDirection(dRec_xe, pRec_xe);
        }
        if (Le.isZero())
            continue;

        Float G_sp = ablateDistanceXs ? 1.0f : (1.0f / (dist_sp * dist_sp));
        Float G_pe = ablateDistanceXe ? 1.0f : (1.0f / (dist_pe * dist_pe));
        contribution = f_xs * G_sp * f_xp * G_pe * Le;

        Float lum = contribution.getLuminance();
        if (std::isnan(lum) || lum < 0.0f)
            continue;

        
#ifdef DO_TIMING_IMPORTANCE
        importanceTimer.addElapsed("emitter", startTime); 
        startTime = importanceTimer.getNanoSeconds();
#endif 
        weightedLuminance += pointWeight * lum;
    }

    if (std::isnan(weightedLuminance) || weightedLuminance < 0.0f)
        return 0.0f;


    
#ifdef DO_TIMING_IMPORTANCE
    if (importanceTimer.timer->getSeconds() > 10.f) {
        std::cout << "Importance computation timings:\n";
        importanceTimer.report();
        exit(0); 
    }
#endif 

    return nodeInfo.surfaceArea * weightedLuminance;
}

MTS_EXPORT_RENDER Float computeNodeImportance_Unscented_LUT_Specular(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility
) {
    if (!scene || !bvh || !bvh->isBuilt())
        return 0.0f;

    if (nodeIndex < 0 || (size_t) nodeIndex >= bvh->getNodeCount())
        return 0.0f;

    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon)
        return 0.0f;

    auto computeCholeskyLower = [](const Matrix3x3 &covariance) {
        Matrix3x3 L(0.0f);
        const Float eps = 1e-6f;

        L(0,0) = std::sqrt(std::max(covariance(0,0) + eps, (Float) 0.0f));
        if (L(0,0) <= eps) L(0,0) = eps;

        L(1,0) = covariance(1,0) / L(0,0);
        L(1,1) = std::sqrt(std::max(covariance(1,1) + eps - L(1,0)*L(1,0), (Float) 0.0f));
        if (L(1,1) <= eps) L(1,1) = eps;

        L(2,0) = covariance(2,0) / L(0,0);
        L(2,1) = (covariance(2,1) - L(2,0)*L(1,0)) / L(1,1);
        L(2,2) = std::sqrt(std::max(covariance(2,2) + eps - L(2,0)*L(2,0) - L(2,1)*L(2,1), (Float) 0.0f));
        if (L(2,2) <= eps) L(2,2) = eps;

        return L;
    };

    struct WeightedPoint {
        Point p;
        Float w;
    };

    auto generateUnscentedPoints = [&](const Gaussian3D &gaussian) {
        std::array<WeightedPoint, 7> points;

        const Float Ldim = 3.0f;
        const Float lambda = std::sqrt((Float) 3.0f);
        const Float sqrtScale = std::sqrt(Ldim + lambda);
        const Float w0 = lambda / (Ldim + lambda);
        const Float wi = 1.0f / (2.0f * (Ldim + lambda));

        Matrix3x3 S = computeCholeskyLower(gaussian.m_covariance);
        Point mean = Point(gaussian.m_mean);

        points[0] = { mean, w0 };
        for (int i = 0; i < 3; ++i) {
            Vector axisOffset = S.col(i) * sqrtScale;
            points[1 + 2*i]     = { mean + axisOffset, wi };
            points[1 + 2*i + 1] = { mean - axisOffset, wi };
        }

        return points;
    };

    const VMFLUT &vmfLUT = bvh->getVMFLUT();

    // Use precomputed kappas (set by finalize() during BVH build)
    KappaQuery kappaQuery_n    = vmfLUT.prepareKappa(nodeInfo.kappa_n);
    KappaQuery kappaQuery_conv = vmfLUT.prepareKappa(nodeInfo.kappa_conv);

    // Effective F0 as Spectrum: E[β^m*β^c + β^s - β^m*β^s]
    Spectrum f0_eff = nodeInfo.getMeanF0();

    Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
    Point mean = Point(nodeGaussian.m_mean);

    Float dist_xs_sq = (mean - its_xs.p).lengthSquared();
    Float dist_xe_sq = (mean - pRec_xe.p).lengthSquared();
    Float min_dist_sq = std::min(dist_xs_sq, dist_xe_sq);

    Float covTrace = nodeGaussian.m_covariance(0,0) +
                     nodeGaussian.m_covariance(1,1) +
                     nodeGaussian.m_covariance(2,2);

    bool isSpatiallyClustered = covTrace < (bvh->getGaussianSAThreshold() * min_dist_sq);

    std::array<WeightedPoint, 7> sigmaPoints;
    size_t numPointsToEval = 1;

    if (isSpatiallyClustered) {
        sigmaPoints[0] = { mean, 1.0f };
    } else {
        sigmaPoints = generateUnscentedPoints(nodeGaussian);
        numPointsToEval = 7;
    }

    // Use precomputed mean direction (set by finalize() during BVH build)
    Frame vmfFrame(nodeInfo.mu_n);

    Float weightedLuminance = 0.0f;
    for (size_t pointIdx = 0; pointIdx < numPointsToEval; ++pointIdx) {
        const Point &xp = sigmaPoints[pointIdx].p;
        const Float pointWeight = sigmaPoints[pointIdx].w;

        if (std::isnan(xp.x) || std::isnan(xp.y) || std::isnan(xp.z))
            continue;

        Spectrum contribution(0.0f);

        Vector d_sp = xp - its_xs.p;
        Float dist_sp = d_sp.length();
        if (dist_sp <= Epsilon)
            continue;
        d_sp /= dist_sp;

        Vector d_pe = pRec_xe.p - xp;
        Float dist_pe = d_pe.length();
        if (dist_pe <= Epsilon)
            continue;
        d_pe /= dist_pe;

        Ray shadowRay_sp(its_xs.p, d_sp, Epsilon, dist_sp * (1.0f - ShadowEpsilon), 0.f);
        if (checkVisibility && scene->rayIntersect(shadowRay_sp))
            continue;

        Ray shadowRay_pe(xp, d_pe, Epsilon, dist_pe * (1.0f - ShadowEpsilon), 0.f);
        if (checkVisibility && scene->rayIntersect(shadowRay_pe))
            continue;

        Vector l_xs_to_xp = its_xs.toLocal(d_sp);
        BSDFSamplingRecord bRec_xs(its_xs, l_xs_to_xp);
        const BSDF *bsdf_xs = its_xs.getBSDF();
        if (!bsdf_xs)
            continue;

        Spectrum f_xs = bsdf_xs->eval(bRec_xs);

        Vector3 w1_local = vmfFrame.toLocal(-d_sp);
        Vector3 w2_local = vmfFrame.toLocal(d_pe);

        // Evaluate LUT for diffuse (kappa_n) and specular (kappa_conv) lobes
        float vmf_diffuse  = vmfLUT.evalFast(w1_local, w2_local, kappaQuery_n);
    #if 0
        // oldspec
        float vmf_specular = vmfLUT.evalFast(w1_local, w2_local, kappaQuery_conv);
    #else
        // newspec
        // Instead of using the LUT, which stores the convolution of the cosines with the VMF, evaluate this VMF directly:
        VonMisesFisherDistr vmfSpec(nodeInfo.kappa_conv);
        Vector3 whalf = w1_local + w2_local;
        float lengthHalf = whalf.length();
        float vmf_specular;
        if (lengthHalf <= Epsilon)
            vmf_specular = 0.0f;
        else {
            whalf /= lengthHalf;
            // z is the cosine, because we are in local coordinates:
            vmf_specular = vmfSpec.eval(whalf.z);
        }

    #endif
        // Schlick F_c from the path directions at xp (independent of the surface normal).
        // At xp, -d_sp points toward xs and d_pe points toward xe.
        Float Fc = schlickFc(-d_sp, d_pe);
        Spectrum fresnel_xp = f0_eff + Fc * (Spectrum(1.0f) - f0_eff);

        // GGX Smith masking-shadowing G2 = G1(wo) * G1(wi) for the specular term at xp.
        // w1_local (outgoing, xp->xs) and w2_local (incoming, xp->xe) are in the vmfFrame
        // whose Z-axis is mu_n, so .z == cosine with the mean normal.
        Float alpha_xp = std::sqrt(std::max(nodeInfo.getMeanAlphaSq(), (Float)1e-8f));
        alpha_xp = std::max(alpha_xp, (Float)1e-4f);
        auto ggxG1 = [&](Float cosTheta) -> Float {
            if (cosTheta <= 0.0f) return 0.0f;
            Float tanTheta2 = (1.0f - cosTheta * cosTheta) / (cosTheta * cosTheta);
            return 2.0f / (1.0f + std::sqrt(1.0f + alpha_xp * alpha_xp * tanTheta2));
        };
        Float G_xp = ggxG1(w1_local.z) * ggxG1(w2_local.z);

        // f_xp = diffuse term + specular term
        // The convolved kappa accounts for both geometric normal spread and microfacet NDF spread.
        // The specular term now uses the full Schlick Fresnel F = F0 + Fc*(1-F0) instead of just F0,
        // and is modulated by the GGX Smith masking-shadowing term G_xp.

        Spectrum albedo = nodeInfo.getMeanDiffuseAlbedo();

        //TEMP, bypass albedo:
        // albedo = Spectrum(1.0f);

        Spectrum f_xp = albedo * (1.0f / M_PI) * vmf_diffuse
                      + fresnel_xp                       * G_xp           * vmf_specular;

        if (f_xs.isZero() || f_xp.isZero())
            continue;

        const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
        if (!emitter)
            continue;

        DirectionSamplingRecord dRec_xe(-d_pe);
        dRec_xe.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
        Spectrum Le = emitter->evalPosition(pRec_xe) * emitter->evalDirection(dRec_xe, pRec_xe);
        if (Le.isZero())
            continue;

        Float G_sp = 1.0f / (dist_sp * dist_sp);
        Float G_pe = 1.0f / (dist_pe * dist_pe);
        contribution = f_xs * G_sp * f_xp * G_pe * Le;

        Float lum = contribution.getLuminance();
        if (std::isnan(lum) || lum < 0.0f)
            continue;

        weightedLuminance += pointWeight * lum;
    }

    if (std::isnan(weightedLuminance) || weightedLuminance < 0.0f)
        return 0.0f;

    return nodeInfo.surfaceArea * weightedLuminance;
}

// ---------------------------------------------------------------------------
// computeNodeImportanceGlint
//
// Estimates the integral over a BVH node of:
//   W_e(x_n→x_o) · G(x_n,x_o) · f_r(x_n) · G(x_n,x_e) · L_e
//
// for the glint path x_o → x_n → x_e (pinhole camera, no BSDF at x_o).
//
// Key differences from computeNodeImportance_Unscented_LUT:
//   • No BSDF at x_o — W_e (sensor importance) replaces f_xs.
//   • sensor->sampleDirect is used for FoV early-exit AND to obtain W_e(x_n→x_o).
//     For a perspective camera, sampleDirect returns
//       importance(ω) / dist_op²  =  m_normalization / (cos³(θ_xo) · dist_op²)
//     which is W_e_dir / dist_op².  The cos(θ_xn→xo) factor at x_n is absorbed
//     into the VMF LUT evaluation (both incoming and outgoing cosines at x_n are
//     passed to the LUT, giving ∫ cos(θ→xo)·cos(θ→xe)·VMF dn).
//   • FoV checks happen before the expensive BSDF/LUT evaluation:
//       1. Node mean outside FoV → return 0 immediately.
//       2. Emitter faces away from node mean → return 0 immediately.
//       3. Per-sigma-point: FoV check via sampleDirect before VMF evaluation.
//       4. Per-sigma-point: emitter evalDirection check before VMF evaluation.
// ---------------------------------------------------------------------------
MTS_EXPORT_RENDER Float computeNodeImportanceGlint(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Point &x_o,
    const Sensor *sensor,
    const PositionSamplingRecord &pRec_xe
) {
    // ---- 1. Validity --------------------------------------------------------
    if (!scene || !bvh || !sensor || !bvh->isBuilt()) return 0.0f;
    if (nodeIndex < 0 || (size_t)nodeIndex >= bvh->getNodeCount()) return 0.0f;

    const BVHNodeInfo &nodeInfo = bvh->getNodeInfo(nodeIndex);
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon) return 0.0f;

    const Emitter *emitter = static_cast<const Emitter *>(pRec_xe.object);
    if (!emitter) return 0.0f;

    // ---- 2. Spatial distribution (node Gaussian) ----------------------------
    Gaussian3D nodeGaussian = nodeInfo.getGaussian3D();
    Point mean = Point(nodeGaussian.m_mean);

    // ---- 3. FoV early exit at node mean ------------------------------------
    // sampleDirect also gives us W_e_mean = importance / dist_op_mean²
    // We use this to build a base luminance and short-circuit out-of-FoV nodes.
    {
        // Use a dummy 2D sample; perspective camera does not use it (pinhole).
        DirectSamplingRecord dRec_test(mean, /* time */ 0.f);
        Spectrum W_e_test = sensor->sampleDirect(dRec_test, Point2(0.5f, 0.5f));
        if (W_e_test.isZero()) return 0.0f;   // Node mean is outside camera FoV
    }

    // ---- 4. Emitter direction early exit at node mean ----------------------
    // If the emitter is facing away from the node center, the whole node likely
    // contributes zero → skip before doing any heavier work.
    Vector d_mean_to_xe = pRec_xe.p - mean;
    Float dist_pe_mean = d_mean_to_xe.length();
    if (dist_pe_mean <= Epsilon) return 0.0f;
    d_mean_to_xe /= dist_pe_mean;   // normalised direction toward emitter

    // evalPosition does not depend on direction; hoist outside the loop.
    Spectrum Le_pos = emitter->evalPosition(pRec_xe);
    if (Le_pos.isZero()) return 0.0f;

    {
        DirectionSamplingRecord dRec_xe_test(-d_mean_to_xe);   // x_e → x_n
        dRec_xe_test.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
        Spectrum Le_dir = emitter->evalDirection(dRec_xe_test, pRec_xe);
        if (Le_dir.isZero()) return 0.0f;   // Emitter faces away from node center
    }
    const bool needsDirSample = emitter->needsDirectionSample();

    // ---- 5. VMF setup (shared across all sigma points) ----------------------
    const VMFLUT &vmfLUT  = bvh->getVMFLUT();
    Float kappa           = std::max((Float) 0.0f, nodeInfo.getVMFKappa());
    KappaQuery kappaQuery = vmfLUT.prepareKappa(kappa);

    Vector vmf_mu = nodeInfo.getVMFMeanDirection();
    if (vmf_mu.lengthSquared() <= Epsilon)
        vmf_mu = Vector(0.0f, 0.0f, 1.0f);
    vmf_mu = normalize(vmf_mu);
    Frame vmfFrame(vmf_mu);

    Spectrum meanAlbedo = nodeInfo.getMeanDiffuseAlbedo();
    if (meanAlbedo.isZero()) return 0.0f;   // No diffuse reflectance on node

    // ---- 6. Spatial clustering check ----------------------------------------
    // If the node subtends a very small solid angle from both x_o and x_e, use
    // only the mean point (bypass Cholesky + 6 extra sigma points).
    Float dist_op_sq = (mean - x_o).lengthSquared();
    Float dist_pe_sq = (mean - pRec_xe.p).lengthSquared();
    Float min_dist_sq = std::min(dist_op_sq, dist_pe_sq);
    Float covTrace = nodeGaussian.m_covariance(0,0)
                   + nodeGaussian.m_covariance(1,1)
                   + nodeGaussian.m_covariance(2,2);
    const bool isSpatiallyClustered =
        covTrace < (bvh->getGaussianSAThreshold() * min_dist_sq);

    // ---- 7. Sigma point generation (Unscented Transform) --------------------
    // Cholesky decomposition helper (same as existing importance functions).
    auto computeCholeskyLower = [](const Matrix3x3 &cov) {
        Matrix3x3 L(0.0f);
        const Float eps = 1e-6f;
        L(0,0) = std::sqrt(std::max(cov(0,0) + eps, (Float) 0.0f));
        if (L(0,0) <= eps) L(0,0) = eps;
        L(1,0) = cov(1,0) / L(0,0);
        L(1,1) = std::sqrt(std::max(cov(1,1) + eps - L(1,0)*L(1,0), (Float) 0.0f));
        if (L(1,1) <= eps) L(1,1) = eps;
        L(2,0) = cov(2,0) / L(0,0);
        L(2,1) = (cov(2,1) - L(2,0)*L(1,0)) / L(1,1);
        L(2,2) = std::sqrt(std::max(cov(2,2) + eps
                                    - L(2,0)*L(2,0) - L(2,1)*L(2,1), (Float) 0.0f));
        if (L(2,2) <= eps) L(2,2) = eps;
        return L;
    };

    struct WeightedPoint { Point p; Float w; };

    auto generateUnscentedPoints = [&](const Gaussian3D &g) {
        std::array<WeightedPoint, 7> pts;
        const Float Ldim      = 3.0f;
        const Float lambda    = std::sqrt((Float) 3.0f);
        const Float sqrtScale = std::sqrt(Ldim + lambda);
        const Float w0 = lambda / (Ldim + lambda);
        const Float wi = 1.0f / (2.0f * (Ldim + lambda));
        Matrix3x3 S  = computeCholeskyLower(g.m_covariance);
        Point     mu = Point(g.m_mean);
        pts[0] = { mu, w0 };
        for (int i = 0; i < 3; ++i) {
            Vector ax = S.col(i) * sqrtScale;
            pts[1 + 2*i]     = { mu + ax, wi };
            pts[1 + 2*i + 1] = { mu - ax, wi };
        }
        return pts;
    };

    std::array<WeightedPoint, 7> sigmaPoints;
    size_t numPoints = 1;
    if (isSpatiallyClustered) {
        sigmaPoints[0] = { mean, 1.0f };
    } else {
        sigmaPoints = generateUnscentedPoints(nodeGaussian);
        numPoints = 7;
    }

    // ---- 8. Unscented Transform loop ----------------------------------------
    // Ordering within the loop: cheapest checks first, VMF LUT last.
    //
    // Per sigma point:
    //   a) FoV check + W_e  via sensor->sampleDirect   (cheap: affine transform + bounds)
    //   b) Emitter Le       via evalDirection           (cheap: 1 dot + compare)
    //   c) VMF convolution  via vmfLUT.evalFast         (moderate: LUT lookup)
    //   d) Geometry terms   1/dist_pe²                 (trivial)
    //   e) Accumulate       W_e * f_xp * G_pe * Le     (trivial)
    //
    // Note: sampleDirect for a pinhole camera does NOT use the 2D sample parameter.
    const Point2 dummySample(0.5f, 0.5f);

    Float weightedLuminance = 0.0f;
    for (size_t idx = 0; idx < numPoints; ++idx) {
        const Point &xp      = sigmaPoints[idx].p;
        const Float  ptWeight = sigmaPoints[idx].w;

        if (std::isnan(xp.x) || std::isnan(xp.y) || std::isnan(xp.z))
            continue;

        // --- (a) FoV check + W_e ------------------------------------------
        // sampleDirect(dRec, sample) fills dRec.d (x_n → x_o direction) and
        // returns W_e = importance(ω) / dist_op²  (or zero if out of FoV).
        DirectSamplingRecord dRec_cam(xp, /* time */ 0.f);
        Spectrum W_e = sensor->sampleDirect(dRec_cam, dummySample);
        if (W_e.isZero()) continue;   // Sigma point outside camera FoV → skip

        // dist from sigma point to emitter
        Vector d_xp_to_xe = pRec_xe.p - xp;
        Float  dist_pe    = d_xp_to_xe.length();
        if (dist_pe <= Epsilon) continue;
        d_xp_to_xe /= dist_pe;

        // --- (b) Emitter direction check ------------------------------------
        DirectionSamplingRecord dRec_xe(-d_xp_to_xe);  // direction x_e → x_n
        dRec_xe.measure = needsDirSample ? ESolidAngle : EDiscrete;
        Spectrum Le_dir = emitter->evalDirection(dRec_xe, pRec_xe);
        if (Le_dir.isZero()) continue;  // Emitter faces away from sigma point → skip

        // --- (c) VMF convolution (BSDF at x_n approximation) ---------------
        // dRec_cam.d is the direction from sigma point TOWARD the camera (x_n → x_o).
        // The LUT evaluates ∫ (n·w1)(n·w2) VMF(n;mu,kappa) dn, which integrates
        //   cos(θ_toward_camera) · cos(θ_toward_emitter)
        // over the node's normal distribution.  This captures both cosine factors
        // at x_n: the geometry term G(x_n,x_o) cosine and the BSDF cos(θ_toward_xe).
        Vector3 w1_local = vmfFrame.toLocal(dRec_cam.d);   // toward camera (x_n → x_o)
        Vector3 w2_local = vmfFrame.toLocal(d_xp_to_xe);   // toward emitter (x_n → x_e)
        float   vmf_conv = vmfLUT.evalFast(w1_local, w2_local, kappaQuery);
        if (vmf_conv <= 0.0f) continue;

        Spectrum f_xp = meanAlbedo * (1.0f / (float) M_PI) * vmf_conv;

        // --- (d) Geometry term ----------------------------------------------
        // W_e already encodes 1/dist_op² (from sampleDirect).
        // G_pe = 1/dist_pe².
        Float G_pe = 1.0f / (dist_pe * dist_pe);

        // --- (e) Accumulate -------------------------------------------------
        Spectrum contribution = W_e * f_xp * G_pe * (Le_pos * Le_dir);
        Float lum = contribution.getLuminance();
        if (std::isnan(lum) || lum < 0.0f) continue;

        weightedLuminance += ptWeight * lum;
    }

    if (std::isnan(weightedLuminance) || weightedLuminance < 0.0f)
        return 0.0f;

    return nodeInfo.surfaceArea * weightedLuminance;
}

MTS_NAMESPACE_END
