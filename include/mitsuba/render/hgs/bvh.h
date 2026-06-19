/*
BVH with Geometric Aggregates for Importance Sampling
*/

#pragma once
#ifndef __MITSUBA_RENDER_BVH_BVH_H_
#define __MITSUBA_RENDER_BVH_BVH_H_

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/aabb.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/render/common.h>
#include <mitsuba/core/cobject.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/vmf.h>

#include <mitsuba/render/hgs/spherical_aabb.h>  
#include <mitsuba/render/hgs/vmf_lut.h> 

#include <array>

// #define DO_TIMING_IMPORTANCE dock
// ^ Enable the timer for importance computation
// #define DO_TIMING_SAMPLING 
// ^ Enable the timer for BVH sampling

#if defined(DO_TIMING_IMPORTANCE) || defined(DO_TIMING_SAMPLING)
#include <mitsuba/core/timer.h>
#include <mitsuba/core/logger.h>

struct MultiTimer {
    // NON-Thread safe timer for keeping track of multiple named timers
    // usage:
    //  MultiTimer timer;
    //  uint64_t t0 = timer.getNanoSeconds();
    //  ... code to time ...
    //  timer.addElapsed("MyCode", t0);
    // timer.report(); // print report at the end
    std::map<std::string, uint64_t> timings_ns;
    mitsuba::ref<mitsuba::Timer> timer;
    uint64_t counter = 0; // counts the number of times we've added time, useful for averaging

    std::map<std::string, uint32_t> counters; // optional additional counters for specific events, e.g. number of times we sampled a node, number of times we did a visibility check, etc.  

    MultiTimer(bool start_now=true) : timer(new mitsuba::Timer(start_now)) { }
    
    void addTime(const std::string& name, uint64_t time) {
        if (timings_ns.find(name) == timings_ns.end()) {
            timings_ns[name] = 0;
        }
        timings_ns[name] += time;
    }

    void addCounter(uint64_t count=1) {
        counter += count;
    }

    void addNamedCounter(const std::string& name, uint32_t count=1) {
        if (counters.find(name) == counters.end()) {
            counters[name] = 0;
        }
        counters[name] += count;
    }

    uint64_t getNanoSeconds() const {
        return timer->getNanoseconds();
    } 
    
    void addElapsed(const std::string& name, uint64_t start_time_ns) {
        uint64_t current_time_ns = timer->getNanoseconds();
        uint64_t elapsed_time = (uint64_t)(current_time_ns - start_time_ns);
        addTime(name, elapsed_time);
    }

    void reset() {
        timer->reset();
        timings_ns.clear();
    }
    void report() const {
        SLog(mitsuba::EInfo, "Importance Computation Timing Report (calls=%llu)",
            (unsigned long long) counter);

        uint64_t totalTimeNs = 0;
        for (const auto& entry : timings_ns)
            totalTimeNs += entry.second;

        const double calls = (counter > 0) ? static_cast<double>(counter) : 1.0;

        SLog(mitsuba::EInfo, "  %-24s | %14s | %14s | %12s | %12s | %9s",
            "Name", "Raw (ns)", "Avg Raw(ns)", "Time (ms)", "Avg ms", "Percent");
        SLog(mitsuba::EInfo, "  %s",
            "-------------------------+----------------+----------------+--------------+--------------+----------");

        for (const auto& entry : timings_ns) {
            const double rawNs    = static_cast<double>(entry.second);
            const double avgRawNs = rawNs / calls;
            const double ms       = rawNs * 1e-6;
            const double avgMs    = avgRawNs * 1e-6;
            const double pct      = (totalTimeNs > 0)
                ? (rawNs / static_cast<double>(totalTimeNs)) * 100.0
                : 0.0;

            SLog(mitsuba::EInfo, "  %-24s | %14.0f | %14.2f | %12.3f | %12.3f | %8.2f%%",
                entry.first.c_str(), rawNs, avgRawNs, ms, avgMs, pct);
        }

        const double totalNsD = static_cast<double>(totalTimeNs);
        SLog(mitsuba::EInfo, "  %s",
            "-------------------------+----------------+----------------+--------------+--------------+----------");
        SLog(mitsuba::EInfo, "  %-24s | %14.0f | %14.2f | %12.3f | %12.3f | %8.2f%%",
            "Total", totalNsD, totalNsD / calls, totalNsD * 1e-6, (totalNsD / calls) * 1e-6, 100.0);

        if (!counters.empty()) {
            SLog(mitsuba::EInfo, "Additional Counters:");
            for (const auto& entry : counters) {
                SLog(mitsuba::EInfo, "  %-24s | %10u", entry.first.c_str(), entry.second);
            }
        }

    }
};

#endif

MTS_NAMESPACE_BEGIN

/// Computes the F_c term of the Schlick Fresnel approximation:
///   F_c(cos) = (1 - |cos|)^5,  where cos = dot(omega_h, omega_o)
///              and omega_h = normalize(omega_i + omega_o).
///
/// Under Schlick's approximation the full Fresnel response is
///   F(omega_h, omega_o; r0) = r0*(1 - F_c) + F_c.
///
/// F_c depends only on omega_i and omega_o through their half-vector —
/// it does NOT depend on the surface normal. This makes it possible to
/// evaluate it independently of the node's normal distribution and
/// incorporate it into the node aggregation contribution.
inline Float schlickFc(Float cosHalfAngle) {
    Float c = std::max((Float)0.0f, std::min((Float)1.0f, std::abs(cosHalfAngle)));
    Float oneMinus = 1.0f - c;
    return oneMinus * oneMinus * oneMinus * oneMinus * oneMinus;
}

/// Convenience overload: computes F_c from omega_i and omega_o directly,
/// computing the half-vector as normalize(omega_i + omega_o) internally.
/// Returns 1 (grazing-angle limit) when omega_i + omega_o ≈ 0.
inline Float schlickFc(const Vector &wi, const Vector &wo) {
    Vector H = wi + wo;
    Float lenSq = H.lengthSquared();
    if (lenSq <= 0.0f)
        return 1.0f;
    H /= std::sqrt(lenSq);
    return schlickFc(dot(H, wo));
}

// Sampling mode enumeration for leaf node sampling
enum class GeometryBVHSamplingMode {
    Primitive,           // Sample leaf primitives uniformly (wrt. area)
    SphericalAABB,       // Sample using spherical AABB, then ray-cast to geometry
    Primitive_3_retries, // retry up to 3 times to sample a point in the surface that has both cosines positive
    Primitive_5_retries,  // retry up to 5 times to sample a point in the surface that has both cosines positive
    SphericalAABB_anyNode, // Sample using spherical AABB for any node, not just leaves (only different from SphericalAABB if pruning/lightcuts is enabled. )
                           // in that case: SphericalAABB first goes down to a leaf by uniform area, then samples its SphericalAABB; SphericalAABB_anyNode directly samples the SphericalAABB of the interior node
    // SphericalAABB_anyNode_3_retries, // Same as ^, with 3 retries for a valid point
};

struct Gaussian3D {
    Point3f m_mean;       // Mean of the Gaussian
    Matrix3x3 m_covariance; // Covariance matrix of the Gaussian

    Gaussian3D() : m_mean(0.0f), m_covariance(0.0f) { }
    Gaussian3D(const Point3f &mean, const Matrix3x3 &cov) : m_mean(mean), m_covariance(cov) { }
    Point3f sample(const Point3f &sample3) const;
};

struct GaussianAggregate {
    Vector3d sumWeightedCentroids; // Stores the running sum of (Area * Centroid)
    Matrix3x3d sumSecondMoment;   // Stores the running sum of the M_i matrices

    // Initialize to zero
    GaussianAggregate() :
        sumWeightedCentroids(0.0f), 
        sumSecondMoment(0.0f) { }

    // Helper: Computes the outer product of a vector with itself (v * v^T)
    static Matrix3x3d outerProduct(const Vector3d& v) {
        Matrix3x3d M;
        M(0, 0) = v.x * v.x; M(0, 1) = v.x * v.y; M(0, 2) = v.x * v.z;
        M(1, 0) = v.y * v.x; M(1, 1) = v.y * v.y; M(1, 2) = v.y * v.z;
        M(2, 0) = v.z * v.x; M(2, 1) = v.z * v.y; M(2, 2) = v.z * v.z;
        return M;
    }

    // Add a single triangle to the aggregate
    inline void setFromTriangle(const Point& v1, const Point& v2, const Point& v3, double area) {
        if (area <= 0.0f) return; // Ignore degenerate triangles
        Vector3d p1(v1), p2(v2), p3(v3);
        Vector3d centroid = (p1 + p2 + p3) / 3.0f;

        // 3. Compute the uncentered second moment matrix for this triangle
        // M_i = (Area / 12) * (v1*v1^T + v2*v2^T + v3*v3^T + 9 * c*c^T)
        Matrix3x3d M_i = (outerProduct(p1) + 
                        outerProduct(p2) + 
                        outerProduct(p3) + 
                        outerProduct(centroid) * 9.0f) * (area / 12.0f);

        // 4. Accumulate
        sumWeightedCentroids += centroid * area;
        sumSecondMoment += M_i;
    }

    // Combine another aggregate into this one (associative and commutative)
    inline void combine(const GaussianAggregate& other) {
        sumWeightedCentroids += other.sumWeightedCentroids;
        sumSecondMoment += other.sumSecondMoment;
    }
    
    // Extract the final 3D Gaussian parameters (Removed Float surfaceArea parameter)
    inline void getGaussian(double surfaceArea, Point& outMean, Matrix3x3& outCovariance) const {
        if (surfaceArea <= 0.0) {
            outMean = Point(0.0f);
            outCovariance = Matrix3x3(0.0f);
            return;
        }

        Vector3d meanVec = sumWeightedCentroids / surfaceArea;
        Matrix3x3d meanOuter = outerProduct(meanVec);
        Matrix3x3d covDouble = (sumSecondMoment / surfaceArea) - meanOuter;

        // Downcast to Float for the final output
        outMean = Point((Float)meanVec.x, (Float)meanVec.y, (Float)meanVec.z);
        
        const Float epsilon = 1e-6f; // Minimum variance to ensure matrix is positive definite

        for(int i = 0; i < 3; ++i) {
            for(int j = 0; j < 3; ++j) {
                outCovariance(i,j) = (Float)covDouble(i,j);
            }
            // Clamp diagonal entries to prevent negative variance, and regularize
            outCovariance(i,i) = std::max(outCovariance(i,i), 0.0f) + epsilon;
        }
    }
};


struct NormalDistributionVMF {
    // The VMF can be represented by just the sum of area-weighted normals and the total area, since we can compute mean direction and kappa from those.
    // surfaceArea is in the BVHNodeInfo
    Vector s_normals; // *Unnormalized* sum of weighted normals:  sum ( area_i * n_i ) for triangles in this node

    NormalDistributionVMF() : s_normals(0.0f) { }

    NormalDistributionVMF(const Vector &normal, Float area=1.f) : s_normals(normal * area) { }

    // Returns the mean direction (mu) of the fitted VMF distribution
    inline Vector getMu() const {    
        Float len = s_normals.length();
        if (len > 1e-20f)
            return s_normals / len;

        // std::cout << "WARNING: VMF with zero mean direction ?? s_normals=" << s_normals.toString() << std::endl;
        return Vector(0.0f, 0.0f, 1.0f); // in case they are isotropic
    }

    // surfaceArea must be > 0
    inline Float getKappa(Float surfaceArea) const {
        Float rBar = s_normals.length() / surfaceArea;
        rBar = std::max((Float) 0.0f, std::min((Float) (1.0f - 1e-6f), rBar));
        return VonMisesFisherDistr::forMeanLength(rBar);
    }
};

/// BVH tree node
/// For interior nodes: nLeafPrimitives = 0, secondChildOffset points to right child
/// For leaf nodes: nLeafPrimitives > 0, primitivesOffset points into primitive array
struct BVHNode {
    AABB bounds;                // Bounding box of this node
    
    union {
        uint32_t primitivesOffset;  // For leaf: offset into primitive array
        uint32_t secondChildOffset; // For interior: offset to second child
    };
    uint32_t nSubtreePrimitives; // Total primitives in this subtree
    uint32_t nLeafPrimitives;    // >0 = leaf with N primitives, 0 = interior node (upgraded to uint32_t to support shallow depth)
    uint8_t splitAxis;           // Axis used for split (0=x, 1=y, 2=z)
    uint16_t depth;              // depth of this node in the tree (root=0), (note: up to 65535 should be mooooore than enough)
    uint8_t pad;                 // Padding for alignment
    
    BVHNode() : nSubtreePrimitives(0), nLeafPrimitives(0), splitAxis(0) {
        primitivesOffset = 0;
        pad = 0;
    }
    
    bool isLeaf() const { return nLeafPrimitives > 0; }
    bool isInterior() const { return nLeafPrimitives == 0; }

    std::string toString() const {
        std::ostringstream oss;
        oss << "BVHNode(bounds=" << bounds.toString()
            << ", nSubtreePrimitives=" << nSubtreePrimitives
            << ", nLeafPrimitives=" << nLeafPrimitives
            << ", splitAxis=" << (int)splitAxis
            << ", primitivesOffset=" << primitivesOffset
            << ", secondChildOffset=" << secondChildOffset
            << ")";
        return oss.str();
    }
};

/// Aggregate information for a BVH node (or a single triangle)
/// Used both for leaf-level triangle data and accumulated interior node data
struct BVHNodeInfo {
    struct WeightedPoint {
        Point p;
        Float w;
    };

    // ---- Spatial information for the node ----
    double surfaceArea;                    // Σ Area_i
    GaussianAggregate gaussianAggregate;  // Stores the raw sums needed to compute the Gaussian parameters (mean and covariance) in a single pass over the triangles

    // ---- Geometric normal distribution ----
    NormalDistributionVMF normalDistribution;  // Stores the raw sums needed to compute the VMF parameters (mean direction and kappa) in a single pass over the triangles

    // ---- Material/reflectance information for the node ----
    // Raw area-weighted sums. Needs to be divided by surfaceArea for per-area mean values.
    // Stored as sums so accumulate() is just addition
    Spectrum s_diffuse_albedo;     // Σ Area_i * (1-metallic_i) * reflectance_i          (for E[f_d] ∝ E[(1-β^m)*β^c])
    Spectrum s_metallic_albedo;    // Σ Area_i * metallic_i * reflectance_i              (for E[β^m * β^c])
    Float    s_specular;           // Σ Area_i * specularF0_i                            (for E[β^s])
    Float    s_metallic_specular;  // Σ Area_i * metallic_i * specularF0_i               (for E[β^m * β^s])
    Float    s_alpha_sq;           // Σ Area_i * α_i²                                    (for E[α²])

    // ---- Derived values (valid after finalize() is called) ----
    // These are cached from the raw sums for fast access in importance functions.
    Float  kappa_n    = 0.0f;                      // VMF concentration from geometric normals
    Vector mu_n       = Vector(0.0f, 0.0f, 1.0f);  // VMF mean direction (unit)
    Float  kappa_m    = 2.0f;                       // VMF approx of GGX NDF: 2/E[α²] (default: α=1)
    Float  kappa_conv = 0.0f;                       // Effective lobe: κ_n*κ_m/(κ_n+κ_m)
    std::array<WeightedPoint, 7> unscentedPoints;   // Query-independent sigma points for node Gaussian

    bool valid;                     // Whether this node has valid data

    BVHNodeInfo()
        : surfaceArea(0.0),
          s_diffuse_albedo(0.0f),
          s_metallic_albedo(0.0f),
          s_specular(0.0f),
          s_metallic_specular(0.0f),
          s_alpha_sq(0.0f),
          valid(false) {}

    void setFromTriangle(const Point& v1, const Point& v2, const Point& v3, const Shape* shape);

    /// Compute and cache all derived values from the raw sums.
    /// Must be called once after all accumulate() calls for a node are complete.
    void finalize() {
        float surfaceAreaFloat = (float) surfaceArea;
        kappa_n = normalDistribution.getKappa(surfaceAreaFloat);
        mu_n    = normalDistribution.getMu();
        Float a2 = (surfaceAreaFloat > 0.f) ? s_alpha_sq / surfaceAreaFloat : 1.0f;
        kappa_m  = (a2 > 1e-8f) ? std::min(2.0f / a2, (Float) 1e6f) : 1e6f;
        Float d  = kappa_n + kappa_m;
        kappa_conv = (d > 0.f) ? (kappa_n * kappa_m) / d : 0.0f;

        auto computeCholeskyLower = [](const Matrix3x3 &covariance) {
            Matrix3x3 L(0.0f);
            const Float eps = 1e-6f;

            L(0,0) = std::sqrt(std::max(covariance(0,0) + eps, (Float) 0.0f));
            if (L(0,0) <= eps)
                L(0,0) = eps;

            L(1,0) = covariance(1,0) / L(0,0);
            L(1,1) = std::sqrt(std::max(covariance(1,1) + eps - L(1,0) * L(1,0), (Float) 0.0f));
            if (L(1,1) <= eps)
                L(1,1) = eps;

            L(2,0) = covariance(2,0) / L(0,0);
            L(2,1) = (covariance(2,1) - L(2,0) * L(1,0)) / L(1,1);
            L(2,2) = std::sqrt(std::max(covariance(2,2) + eps - L(2,0) * L(2,0) - L(2,1) * L(2,1), (Float) 0.0f));
            if (L(2,2) <= eps)
                L(2,2) = eps;

            return L;
        };

        Gaussian3D nodeGaussian = getGaussian3D();
        const Float Ldim = 3.0f;
        const Float lambda = std::sqrt((Float) 3.0f);
        const Float sqrtScale = std::sqrt(Ldim + lambda);
        const Float w0 = lambda / (Ldim + lambda);
        const Float wi = 1.0f / (2.0f * (Ldim + lambda));

        Matrix3x3 S = computeCholeskyLower(nodeGaussian.m_covariance);
        Point mean = Point(nodeGaussian.m_mean);

        unscentedPoints[0] = { mean, w0 };
        for (int i = 0; i < 3; ++i) {
            Vector axisOffset = S.col(i) * sqrtScale;
            unscentedPoints[1 + 2*i]     = { mean + axisOffset, wi };
            unscentedPoints[1 + 2*i + 1] = { mean - axisOffset, wi };
        }
    }

    inline void accumulate(const BVHNodeInfo &other);

    /// VMF kappa from geometric normals (precomputed by finalize())
    Float getVMFKappa() const { return kappa_n; }
    /// VMF mean direction (precomputed by finalize())
    Vector getVMFMeanDirection() const { return mu_n; }
    /// Microfacet VMF kappa = 2/E[α²] (precomputed by finalize())
    Float getMicrofacetKappa() const { return kappa_m; }
    /// Convolved kappa = κ_n*κ_m/(κ_n+κ_m) (precomputed by finalize())
    Float getConvolvedKappa() const { return kappa_conv; }
    /// Area-averaged roughness squared E[α²] (computed on-the-fly from s_alpha_sq)
    Float getMeanAlphaSq() const {
        return (surfaceArea > 0.f) ? s_alpha_sq / surfaceArea : 1.0f;
    }
    Gaussian3D getGaussian3D() const {
        Point mean;
        Matrix3x3 cov;
        gaussianAggregate.getGaussian(surfaceArea, mean, cov);
        return Gaussian3D(Point3f(mean), cov);
    }
    const std::array<WeightedPoint, 7> &getUnscentedPoints() const {
        return unscentedPoints;
    }
    /// Area-averaged diffuse albedo: E[(1-β^m)*β^c] as Spectrum
    Spectrum getMeanDiffuseAlbedo() const {
        return (surfaceArea > 0.f) ? s_diffuse_albedo / surfaceArea : Spectrum(0.0f);
    }
    /// Luminance of the area-averaged diffuse albedo (convenience Float accessor)
    Float getMeanDiffuseAlbedoLum() const {
        return getMeanDiffuseAlbedo().getLuminance();
    }
    /// Area-averaged effective Fresnel at normal incidence:
    ///   E[β^m*β^c + β^s - β^m*β^s]  =  metallic*F_metal + (1-metallic)*F_dielectric at cos=1
    Spectrum getMeanF0() const {
        if (surfaceArea <= 0.f) return Spectrum(0.0f);
        return (s_metallic_albedo + Spectrum(s_specular) - Spectrum(s_metallic_specular)) / surfaceArea;
    }

    // Describe the node info (for debugging)
    inline std::string toString() const {
        std::ostringstream oss;
        oss << "BVHNodeInfo[" << std::endl;
        oss << "  surfaceArea = " << surfaceArea << "," << std::endl;
        oss << "  s_diffuse_albedo (sum) = " << s_diffuse_albedo.toString()
            << " mean = " << getMeanDiffuseAlbedo().toString() << "," << std::endl;
        oss << "  s_metallic_albedo (sum) = " << s_metallic_albedo.toString() << "," << std::endl;
        oss << "  s_specular = " << s_specular << "," << std::endl;
        oss << "  s_metallic_specular = " << s_metallic_specular << "," << std::endl;
        oss << "  s_alpha_sq = " << s_alpha_sq << " (getMeanAlphaSq=" << getMeanAlphaSq() << ")," << std::endl;
        oss << "  getMeanF0 = " << getMeanF0().toString()
            << ", getMicrofacetKappa = " << getMicrofacetKappa()
            << ", getConvolvedKappa = " << getConvolvedKappa() << "," << std::endl;
        #ifdef USE_SGGX_AGGREGATES
        oss << "  normalDistribution.m1 = " << normalDistribution.m1.toString() << "," << std::endl;
        oss << "  normalDistribution.m2 = " << normalDistribution.m2.toString() << "," << std::endl;
        #else 
        oss << "  normalDistribution.s_normals = " << normalDistribution.s_normals.toString() << "," << std::endl;
        oss << "  normalDistribution.kappa = " << getVMFKappa() << "," << std::endl;
        Point mean;
        Matrix3x3 cov;
        gaussianAggregate.getGaussian(surfaceArea, mean, cov);
        oss << "  gaussianAggregate.mean = " << mean.toString() << "," << std::endl;
        oss << "  gaussianAggregate.covariance = " << cov.toString()  << "," << std::endl;
        #endif
        oss << "  valid = " << (valid ? "true" : "false") << std::endl;
        oss << "]";
        return oss.str();
    }
};

/// Build-time primitive with cached data used for SAH and partitioning.
struct BVHBuildPrimitive {
    uint32_t meshIndex;         // Index of the TriMesh in scene->getMeshes()
    uint32_t triangleIndex;     // Index of the triangle within the mesh
    AABB bounds;                // Cached bounding box
    Point centroid;             // Cached centroid for sorting
    Float surfaceArea;          // Cached triangle area (for angular SAH)
    Vector areaWeightedNormal;  // Cached area-weighted geometric normal (for angular SAH)

    BVHBuildPrimitive()
        : meshIndex(0), triangleIndex(0), surfaceArea(0.0f), areaWeightedNormal(0.0f) {}

    BVHBuildPrimitive(uint32_t mIdx, uint32_t tIdx, const AABB &b, const Point &c,
                      Float area, const Vector &weightedNormal)
        : meshIndex(mIdx), triangleIndex(tIdx), bounds(b), centroid(c),
          surfaceArea(area), areaWeightedNormal(weightedNormal) {}
};

/// Compact runtime primitive reference (indices only).
struct BVHPrimitive {
    uint32_t meshIndex;     // Index of the TriMesh in scene->getMeshes()
    uint32_t triangleIndex; // Index of the triangle within the mesh

    BVHPrimitive() : meshIndex(0), triangleIndex(0) {}
    BVHPrimitive(uint32_t mIdx, uint32_t tIdx) : meshIndex(mIdx), triangleIndex(tIdx) {}
};

/// Main BVH class for geometry sampling
/// Builds a triangle-level BVH with geometric aggregates for importance-driven traversal
class MTS_EXPORT_RENDER GeometryBVH : public ConfigurableObject {
public:
    MTS_DECLARE_CLASS()
    
    /// Constructor with configurable depth and sampling mode
    /// \param maxDepth Maximum depth of the tree (default: 16)
    /// \param samplingMode Sampling strategy for leaves (default: Primitive)
    GeometryBVH(uint32_t maxDepth = 16, GeometryBVHSamplingMode samplingMode = GeometryBVHSamplingMode::Primitive);

    /// Constructor from properties
    /// @param props Properties object containing configuration parameters
    GeometryBVH(const Properties &props);

    /// Unserialize a geometrybvh instance from a binary data stream
    GeometryBVH(Stream *stream, InstanceManager *manager);
    
    /// Destructor
    ~GeometryBVH();
    
    /// Build BVH from scene geometry
    /// Collects all triangles from scene->getMeshes() and builds object-partition BVH
    /// \param scene The scene containing triangle meshes
    void buildBVH(Scene *scene);
    
    /// Build geometric aggregates for all nodes (call after build())
    /// Computes albedo, area, normal distribution for each node bottom-up
    /// \param scene The scene (needed to access mesh data for triangle aggregates)
    void buildAggregates(const Scene *scene);

    /// Approximate in-memory footprint of BVH-owned storage in bytes.
    size_t getMemoryUsageBytes() const;

    /// Approximate in-memory footprint of BVH-owned storage in MiB.
    Float getMemoryUsageMB() const {
        return (Float) getMemoryUsageBytes() / (1024.0f * 1024.0f);
    }
    
    // ========== Accessors ==========
    
    /// Get total number of nodes in the tree
    size_t getNodeCount() const { return m_nodes.size(); }
    
    /// Get total number of primitives (triangles)
    size_t getPrimitiveCount() const { return m_primitives.size(); }
    
    /// Access a node by index
    const BVHNode& getNode(size_t index) const { return m_nodes[index]; }
    
    /// Access a primitive by index
    const BVHPrimitive& getPrimitive(size_t index) const { return m_primitives[index]; }
    
    /// Access aggregate info for a node
    const BVHNodeInfo& getNodeInfo(size_t index) const { return m_nodeInfos[index]; }
    
    /// Get the configured maximum tree depth
    uint32_t getMaxDepth() const { return m_maxDepth; }
    
    /// Get the sampling mode
    GeometryBVHSamplingMode getSamplingMode() const { return m_samplingMode; }
    
    /// Get left child index (always nodeIndex + 1 for interior nodes)
    size_t getLeftChild(size_t nodeIndex) const { return nodeIndex + 1; }
    
    /// Get right child index (from secondChildOffset for interior nodes)
    size_t getRightChild(size_t nodeIndex) const { 
        return m_nodes[nodeIndex].secondChildOffset; 
    }

    /// Returns the maximum distance from the given node to any leaf in its subtree (in terms of number of edges)
    /// A leaf is distance 0, parent distance 1, ... 
    size_t getMaxDistToLeaf(size_t nodeIndex) const {
        if (m_nodes[nodeIndex].isLeaf()) return 0.f; // end recursion at leaf
        int left = getMaxDistToLeaf(getLeftChild(nodeIndex));
        int right = getMaxDistToLeaf(getRightChild(nodeIndex));
        return 1 + std::max(left, right);
    }   

    /// Returns the minimum distance from the given node to any leaf in its subtree (in terms of number of edges)
    /// A leaf is distance 0, parent distance 1, ... 
    /// ((I don't think this will be useful but just in case.))
    size_t getMinDistToLeaf(size_t nodeIndex) const {
        if (m_nodes[nodeIndex].isLeaf()) return 0.f;
        int left = getMinDistToLeaf(getLeftChild(nodeIndex));
        int right = getMinDistToLeaf(getRightChild(nodeIndex));
        return 1 + std::min(left, right);
    }   

    size_t getLevel(size_t nodeIndex) const { return getMinDistToLeaf(nodeIndex); } // Level is the same as min distance to leaf
    
    
    /// Check if BVH has been built
    bool isBuilt() const { return !m_nodes.empty(); }

    /// Sample a point on geometry using importance-driven BVH traversal
    /// \param scene Scene for accessing mesh data
    /// \param its_xs Current shading point for importance evaluation
    /// \param pRec_xe Emitter sample for importance evaluation
    /// \param sample 2D random sample (used for traversal and triangle sampling)
    /// \param its_xp Output: intersection record at sampled position
    /// \param pdf Output: probability density of the sample
    /// \return true if sampling succeeded, false otherwise
    bool sampleGeometry(
        const Scene *scene,
        Sampler *sampler,
        const Intersection &its_xs,
        const PositionSamplingRecord &pRec_xe,
        Intersection &its_xp,
        Float &pdf
    ) const;
    
    /// Importance-driven geometry sampling from the camera origin (glints use case).
    /// Uses computeNodeImportanceGlint for traversal, which incorporates camera FoV,
    /// emitter visibility, and BSDF (via VMF LUT) into the node importance weights.
    /// \param scene Scene for ray-intersection queries
    /// \param sampler Random sampler
    /// \param x_o Camera origin in world space (pinhole position)
    /// \param sensor Sensor for FoV checks and W_e evaluation inside importance function
    /// \param pRec_xe Emitter sample record (position, pdf, object)
    /// \param its_xn Output: intersection record of the sampled intermediate vertex
    /// \param pdf Output: solid-angle PDF of the sample
    /// \return true if sampling succeeded
    bool sampleGeometryCamera(
        const Scene *scene,
        Sampler *sampler,
        const Point &x_o,
        const Sensor *sensor,
        const PositionSamplingRecord &pRec_xe,
        Intersection &its_xn,
        Float &pdf
    );

    /// Compute the PDF for sampling an intersection using the BVH sampling strategy
    /// \param scene Scene for accessing mesh data
    /// \param its_xs Current shading point
    /// \param pRec_xe Emitter sample
    /// \param its_xp Intersection record of the point being queried
    /// \return The probability density
    Float pdfGeometry(
        const Scene *scene,
        const Intersection &its_xs,
        const PositionSamplingRecord &pRec_xe,
        const Intersection &its_xp
    ) const;

    /// Compute the PDF for sampling a specific triangle
    /// \param scene Scene for accessing mesh data
    /// \param its_xs Current shading point
    /// \param pRec_xe Emitter sample
    /// \param meshIndex Index of the mesh being queried
    /// \param triangleIndex Index of the triangle within the mesh
    /// \param position Position on the triangle (used for area/solid angle PDF)
    /// \return The probability density
    Float pdfGeometry(
        const Scene *scene,
        const Intersection &its_xs,
        const PositionSamplingRecord &pRec_xe,
        const Intersection &its_xp,
        uint32_t meshIndex,
        uint32_t triangleIndex,
        const Point &position
    ) const;

    /// Compute the PDF for sampling x_n via sampleGeometryCamera (glint technique).
    /// Mirrors pdfGeometry but uses computeNodeImportanceGlint and the camera origin
    /// as the "from" point, matching the distribution used during sampling.
    /// \param scene   Scene for accessing mesh data
    /// \param x_o     Camera pinhole position (world space)
    /// \param sensor  Perspective sensor — used inside computeNodeImportanceGlint
    /// \param pRec_xe Emitter sample record
    /// \param its_xn  Intersection being queried
    /// \return The solid-angle probability density
    Float pdfGeometryCamera(
        const Scene *scene,
        const Point &x_o,
        const Sensor *sensor,
        const PositionSamplingRecord &pRec_xe,
        const Intersection &its_xn
    ) const;

    /// Full version of pdfGeometryCamera with pre-resolved mesh/triangle indices.
    /// \param scene          Scene for accessing mesh data
    /// \param x_o            Camera pinhole position (world space)
    /// \param sensor         Perspective sensor
    /// \param pRec_xe        Emitter sample record
    /// \param its_xn         Intersection being queried
    /// \param meshIndex      Index of the mesh in scene->getMeshes()
    /// \param triangleIndex  Index of the triangle within the mesh
    /// \param position       World-space position on the triangle
    /// \return The solid-angle probability density
    Float pdfGeometryCamera(
        const Scene *scene,
        const Point &x_o,
        const Sensor *sensor,
        const PositionSamplingRecord &pRec_xe,
        const Intersection &its_xn,
        uint32_t meshIndex,
        uint32_t triangleIndex,
        const Point &position
    ) const;

    /// Check whether an intersection belongs to a given leaf node
    /// \param scene Scene for resolving mesh indices
    /// \param its Intersection record to validate
    /// \param leafNodeIndex Index of the leaf node being tested
    /// \return true if the intersection's primitive is within the leaf
    bool intersectionInLeafNode(
        const Scene *scene,
        const Intersection &its,
        size_t leafNodeIndex) const;

    /// Check whether an intersection belongs to a given inner node
    /// \param scene Scene for resolving mesh indices
    /// \param its Intersection record to validate
    /// \param nodeIndex Index of the node being tested
    /// \return true if the intersection's primitive is within the leaf
    bool intersectionInNode(
        const Scene *scene,
        const Intersection &its,
        size_t nodeIndex) const;

        

    void print();

    const VMFLUT& getVMFLUT() const { return m_vmfLUT; }
    
    bool sampleLeafPrimitive(
        const Scene *scene,
        const BVHNode &leafNode,
        size_t leafNodeIndex,
        const Float &sample1,
        const Point2 &sample2,
        const Point &xs,
        Intersection &its_xp,
        Float &pdf_xp, 
        bool return_solidangle_pdf=true
    ) const;
    
    /// Sample leaf using spherical AABB approach
    /// Samples a direction toward the leaf AABB using solid angle, then ray-casts to find geometry
    /// \param leafNode The leaf node to sample
    /// \param leafNodeIndex Index of the leaf node
    /// \param scene Scene for accessing mesh data and ray tracing
    /// \param xs Current shading point
    /// \param sample2 2D random sample for leaf-level sampling
    /// \param its Output: intersection record at sampled position
    /// \param pdf Output: combined PDF (AABB solid angle * area PDF)
    /// \return true if sampling succeeded, false otherwise
    bool sampleLeafSphericalAABB(
        const Scene *scene,
        const BVHNode &leafNode,
        size_t leafNodeIndex,
        const Point2 &sample2,
        const Point &xs,
        Intersection &its_xp,
        Float &pdf_xp
    ) const;

    // Same as the sampleLeafPrimitive, but for non-nodes. Internally, samples a leaf uniformly, then calls sampleLeafPrimitive
    // Returns the pdf in solid angle
    bool sampleNodePrimitive(
        const Scene *scene,
        size_t nodeIdx, 
        Sampler* sampler,
        const Point &xs,
        Intersection &its_xp,
        Float &pdf_xp,
        bool return_solidangle_pdf=true
    ) const;

    float getGaussianSAThreshold() const { return m_gaussianSAThreshold; }
    float getKappaThreshold() const { return m_kappaThreshold; }
    
protected:
    Scene *m_scene; // Non-owning pointer to the scene (for accessing mesh data during sampling and aggregate computation
    
    VMFLUT m_vmfLUT; // Precomputed VMF intensity lookup table for importance evaluation

    std::vector<BVHNode> m_nodes;           // Tree nodes (depth-first order)
    std::vector<BVHPrimitive> m_primitives; // Leaf-ordered primitives (compact indices only)
    std::vector<BVHNodeInfo> m_nodeInfos;   // Parallel array of aggregates

    // Per-leaf alias tables for O(1) area-proportional primitive sampling.
    // For leaf node i: table starts at m_leafAliasTableOffsets[i] and has nLeafPrimitives entries.
    std::vector<uint32_t> m_leafAliasTableOffsets;
    std::vector<Float> m_leafAliasProb;
    std::vector<uint32_t> m_leafAliasAlias;

    std::vector<uint32_t> m_primitiveLevels; // Level of each primitive in the tree (for debugging)
     
    // Triangle to global primitive index mapping
    std::vector<uint32_t> m_meshOffsets;    // Offset for each mesh in m_triangleToPrim
    std::vector<uint32_t> m_triangleToPrim; // Flat map of global primitive indices
    
    uint32_t m_maxDepth;                    // Maximum depth of the tree
    GeometryBVHSamplingMode m_samplingMode; // Current sampling mode

    // Node importance configuration (0=ignore, 1=full importance):
public:
    float m_debug1, m_debug2, m_debug3, m_debug4, m_debug5; // to quickly hack different things 

    enum Ablation {
        AblateNone = 0,
        AblateVMF = 1, 
        AblateGaussian = 2,
        AblateCosineXs = 4, 
        AblateCosineXe = 8, 
        AblateDistanceXs = 16,
        AblateDistanceXe = 32,
        AblateAlbedo = 64
    };

    uint32_t m_ablationMask; // Bitmask to disable certain importance components for ablation studies

    
#if defined(DO_TIMING_IMPORTANCE) || defined(DO_TIMING_SAMPLING)
    mutable MultiTimer importanceTimer;

    MultiTimer& getImportanceTimer() const { return importanceTimer; }
#endif

protected:
    float m_probmult; // Multiplier for importance to control its influence on sampling
    float m_defensive_pdf; // Small constant added to importance to avoid zero probabilities
    float m_rrStrength; // Russian roulette strength for BVH traversal (0 disables RR)

    bool m_enableLightcuts; // Whether to enable lightcut-like behavior by treating low-importance nodes as leaves during sampling
    float m_lightcutThreshold; // threshold for lightcuts

    float m_gaussianSAThreshold; // Solid angle threshold for treating a gaussian as a point during importance evaluation (to avoid expensive integrals for small/far nodes)
                                 // value should be 0.f for NEVER treating them as points, >=4PI for ALWAYS treating them as points. 
                                 // .01*PI or .001*PI might be reasonable
    float m_kappaThreshold; // Concentration threshold for treating a VMF as a spherical distribution. Currently does nothing!
 
    bool m_outputDebugCSV; // Whether to output debug info for each node to CSV (for visualization/debugging)
    uint32_t m_csvEnabledLevels; // Only output debug CSV for nodes up to this level (to avoid huge files)

    bool m_enableSAH = true; // Whether to use SAH for BVH construction (instead of median split)
    float m_angularSAHWeight = 0.0f;      // Extra cost weight for angular dispersion term (0 disables angular SAH)
    float m_angularSAHKappaScale = 10.0f; // Scale to map VMF kappa to [0,1] coherence in angular SAH
    bool m_logAngularSAHStats = true;      // Emit build-time summary of angular SAH penalties

    float m_buildAngularPenaltySum = 0.0;         // Sum of unweighted angular penalties for selected splits
    float m_buildWeightedAngularPenaltySum = 0.0; // Sum of weighted angular additions for selected splits
    uint32_t m_buildAngularSplitCount = 0;         // Number of selected splits contributing to angular stats
    
    /// Recursive build function using object median split
    /// \param buildPrims Working array of primitives (will be partially sorted)
    /// \param start Start index in buildPrims
    /// \param end End index in buildPrims (exclusive)
    /// \param orderedPrims Output array for leaf-ordered primitives
    /// \param depth Current depth of the recursion
    /// \return Index of the created node
    size_t buildBVHRecursive(
        std::vector<BVHBuildPrimitive> &buildPrims,
        size_t start, size_t end,
        std::vector<BVHPrimitive> &orderedPrims,
        std::vector<Float> &orderedAreas,
        uint32_t depth
    );

        
    /// Recursive build function using Surface Area Heuristic (SAH) for splitting
    /// \param buildPrims Working array of primitives (will be partially sorted)
    /// \param start Start index in buildPrims
    /// \param end End index in buildPrims (exclusive)
    /// \param orderedPrims Output array for leaf-ordered primitives
    /// \param depth Current depth of the recursion
    /// \return Index of the created node
    size_t buildBVHRecursiveSAH(
        std::vector<BVHBuildPrimitive> &buildPrims,
        size_t start, size_t end,
        std::vector<BVHPrimitive> &orderedPrims,
        std::vector<Float> &orderedAreas,
        uint32_t depth
    );
    
    /// Recursive aggregate computation (post-order traversal)
    /// \param nodeIndex Current node index
    /// \param scene Scene for accessing mesh data
    void buildAggregatesRecursive(size_t nodeIndex, const Scene *scene);

    /// Build per-leaf alias tables for O(1) area-proportional primitive sampling.
    void buildLeafSamplingTables(const Scene *scene, const std::vector<Float> &primitiveAreas);

};

// Export functions for external linkage
/// Compute aggregate information for a single triangle
/// \param shape The shape (must be a TriMesh)
/// \param triangleIndex Index of the triangle within the mesh
/// \param scene Optional scene for context (can be NULL)
/// \return BVHNodeInfo with area, albedo, normal, centerOfMass; valid=true if successful
MTS_EXPORT_RENDER BVHNodeInfo computeTriangleInfo(
    const Shape *shape,
    size_t triangleIndex,
    const Scene *scene
);

// Evaluate the lenght-2 contribution of subpath xs -> xp -> xe,
// where xp is a point sampled on the geometry, xs is a point on the path and xe is the emitter sample.
// Prerequisites:
// - its_xs.shape must be a valid shape with a BSDF
// - same for its_xp.shape
// - pRec_xe.object must point to a valid emitter
MTS_EXPORT_RENDER Spectrum evalLength2Contribution(
    const Scene *scene,
    const Intersection &its_xs,
    const Intersection &its_xp,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility = true
);

// Evaluate the lenght-2 contribution of subpath xs -> xp -> xe,
// where xp is a point sampled on the geometry, xs is a point on the path and xe is the emitter sample.
// Prerequisites:
// - its_xs.shape must be a valid shape with a BSDF
// - same for its_xp.shape
// - pRec_xe.object must point to a valid emitter
MTS_EXPORT_RENDER Spectrum evalLength2Contribution(
    const Scene *scene,
    const Intersection &its_xs,
    const Point &xp,
    const Normal &np, 
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility = true
);

// Evaluate the glint contribution of subpath x_o -> x_n -> x_e,
// where x_n is a point on geometry sampled from the camera origin x_o and x_e is the emitter.
// Unlike evalLength2Contribution there is no BSDF at x_o (pinhole camera).
// The sensor importance W_e and the x_o → x_n visibility are handled by
// sampleAttenuatedSensorDirect in the integrator before calling this function.
// Prerequisites:
// - its_xn.shape must be a valid shape with a BSDF
// - pRec_xe.object must point to a valid emitter
// - d_n_to_o is the direction from x_n toward the camera (= dRec.d from sampleAttenuatedSensorDirect)
// Returns: f_xn * G_ne * Le  (does NOT divide by pRec_xe.pdf — caller does)
MTS_EXPORT_RENDER Spectrum evalGlintContribution(
    const Scene *scene,
    const Intersection &its_xn,
    const PositionSamplingRecord &pRec_xe,
    const Vector &d_n_to_o,
    bool checkVisibility = true
);

// Compute the importance of a BVH node for glint sampling (x_o → x_n → x_e).
// Unlike computeNodeImportance this has no BSDF at x_o (pinhole camera).
// Uses sensor->sampleDirect for FoV early-exit + W_e evaluation, and the VMF LUT
// for the BSDF at x_n (diffuse approximation).  Unscented transform over the node's
// 3-D Gaussian for spatial integration.
// \param x_o    Camera pinhole position (world space)
// \param sensor Perspective sensor — used for FoV check and sensor importance W_e
// \param pRec_xe Emitter sample (position, pdf, object)
MTS_EXPORT_RENDER Float computeNodeImportanceGlint(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Point &x_o,
    const Sensor *sensor,
    const PositionSamplingRecord &pRec_xe
);

// Compute the importance of a BVH node for sampling between xs and xe
// This function can be used both for importance-driven sampling and for debugging/visualization of node importance.
MTS_EXPORT_RENDER Float computeNodeImportance(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe
);

// Compute the importance of a BVH node for sampling between xs and xe
// This version uses MC to solve the triple integral over the node's geometry, the normals, and the albedo
MTS_EXPORT_RENDER Float computeNodeImportance_MC(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    const int numSamples = 10000,
    bool checkVisibility = false
);

Float computeNodeImportance_MC_LUT(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe, 
    const int numSamples,
    bool checkVisibility=false
);

MTS_EXPORT_RENDER Float computeNodeImportance_Unscented_MC(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe, 
    const int numSamples,
    bool checkVisibility=false
);

MTS_EXPORT_RENDER Float computeNodeImportance_Unscented_LUT(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility=false
);

/// Specular-aware variant of computeNodeImportance_Unscented_LUT.
/// Uses both diffuse and specular BVH aggregates (s_alpha_sq, s_metallic_albedo,
/// s_specular, s_metallic_specular) to estimate node importance.
/// The specular lobe is evaluated with a convolved VMF kappa that accounts for
/// both geometric normal spread (kappa_n) and microfacet NDF spread (kappa_m).
MTS_EXPORT_RENDER Float computeNodeImportance_Unscented_LUT_Specular(
    const Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    bool checkVisibility=false
);

// Ground truth importance function: evaluates how important a node is for connecting xs to xe 
// (based on actual path contribution by sampling random points on the node's geometry and evaluating their contribution)
MTS_EXPORT_RENDER Float computeNodeImportanceGroundTruth(
    Scene *scene,
    const GeometryBVH *bvh,
    int nodeIndex,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    int numSamples,
    bool checkVisibility
);

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_BVH_BVH_H_ */