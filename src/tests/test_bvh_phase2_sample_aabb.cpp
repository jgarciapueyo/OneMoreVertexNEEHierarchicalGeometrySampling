#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/render/bvh/spherical_aabb.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/core/sched.h>
#include <iostream>
#include <cmath>
#include <iomanip>

MTS_NAMESPACE_BEGIN

// ============================================================================
// Helper functions
// ============================================================================

/// Create a simple triangle mesh with N triangles arranged in a grid
TriMesh* createTestMesh(int numTriangles, Float albedo = 0.8f) {
    // Create a strip of triangles
    int numVerts = numTriangles + 2;
    
    TriMesh *mesh = new TriMesh("test_mesh", numTriangles, numVerts, 
                                 true, false, false, false, false);
    
    // Set up vertices in a strip
    Point *vertices = mesh->getVertexPositions();
    Normal *normals = mesh->getVertexNormals();
    
    for (int i = 0; i < numVerts; ++i) {
        Float x = static_cast<Float>(i);
        Float y = (i % 2 == 0) ? 0.0f : 1.0f;
        vertices[i] = Point(x, y, 0.0f);
        normals[i] = Normal(0.0f, 0.0f, 1.0f);
    }
    
    // Set up triangles
    Triangle *triangles = mesh->getTriangles();
    for (int i = 0; i < numTriangles; ++i) {
        triangles[i].idx[0] = i;
        triangles[i].idx[1] = i + 1;
        triangles[i].idx[2] = i + 2;
    }
    
    // Create diffuse BSDF
    Properties bsdfProps("diffuse");
    bsdfProps.setSpectrum("reflectance", Spectrum(albedo));
    BSDF *bsdf = static_cast<BSDF *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(BSDF), bsdfProps));
    bsdf->configure();
    mesh->addChild(bsdf);
    
    mesh->configure();
    return mesh;
}

/// Create a simple scene with a single mesh
Scene* createTestScene(TriMesh *mesh) {
    Properties sceneProps("scene");
    Scene *scene = new Scene(sceneProps);
    scene->addChild(mesh);
    scene->configure();
    scene->initialize();
    return scene;
}

// Helper to check if value is finite
static inline bool isFinite(Float v) {
    return std::isfinite(v) && v > 0.0f;
}

// ============================================================================
// Test functions
// ============================================================================

bool testSphericalAABBSamplingModeCreation() {
    std::cout << "Test: SphericalAABB mode BVH creation... ";
    
    TriMesh *mesh = createTestMesh(20);
    Scene *scene = createTestScene(mesh);
    
    // Create BVH with SphericalAABB mode
    GeometryBVH bvh(4, GeometryBVHSamplingMode::SphericalAABB);
    
    if (bvh.getSamplingMode() != GeometryBVHSamplingMode::SphericalAABB) {
        std::cout << "FAILED: Sampling mode not set correctly" << std::endl;
        return false;
    }
    
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);
    
    if (!bvh.isBuilt()) {
        std::cout << "FAILED: BVH not built" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testSphericalAABBLeafSampling() {
    std::cout << "Test: SphericalAABB leaf sampling... ";
    
    TriMesh *mesh = createTestMesh(20);
    Scene *scene = createTestScene(mesh);
    
    // Create BVH with SphericalAABB mode
    GeometryBVH bvh(4, GeometryBVHSamplingMode::SphericalAABB);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);
    
    // Sample from above the geometry
    SurfaceSample xs(Point(5.0f, 0.5f, 5.0f), Normal(0.0f, 0.0f, -1.0f));
    EmitterSample xe(Point(5.0f, 0.5f, 0.0f), Normal(0.0f, 0.0f, 1.0f));
    
    Intersection its;
    Float pdf;
    
    Point2 sample1(0.5f, 0.5f);
    Point2 sample2(0.3f, 0.7f);
    
    bool success = bvh.sampleGeometry(scene, xs, xe, sample1, sample2, its, pdf);
    Point position = its.p;
    Normal normal = its.shFrame.n;
    
    if (!success) {
        std::cout << "FAILED: Sampling failed" << std::endl;
        return false;
    }
    
    if (!isFinite(pdf)) {
        std::cout << "FAILED: PDF not finite or zero, pdf=" << pdf << std::endl;
        return false;
    }
    
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
        std::cout << "FAILED: Sampled position contains NaN" << std::endl;
        return false;
    }
    
    if (!std::isfinite(normal.x) || !std::isfinite(normal.y) || !std::isfinite(normal.z)) {
        std::cout << "FAILED: Sampled normal contains NaN" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testPrimitiveSamplingModeStillWorks() {
    std::cout << "Test: Primitive mode still works... ";
    
    TriMesh *mesh = createTestMesh(20);
    Scene *scene = createTestScene(mesh);
    
    // Create BVH with Primitive mode (default)
    GeometryBVH bvh(4, GeometryBVHSamplingMode::Primitive);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);
    
    SurfaceSample xs(Point(5.0f, 0.5f, 5.0f), Normal(0.0f, 0.0f, -1.0f));
    EmitterSample xe(Point(5.0f, 0.5f, 0.0f), Normal(0.0f, 0.0f, 1.0f));
    
    Intersection its;
    Float pdf;
    
    Point2 sample1(0.5f, 0.5f);
    Point2 sample2(0.3f, 0.7f);
    
    bool success = bvh.sampleGeometry(scene, xs, xe, sample1, sample2, its, pdf);
    
    if (!success) {
        std::cout << "FAILED: Primitive sampling failed" << std::endl;
        return false;
    }
    
    if (!isFinite(pdf)) {
        std::cout << "FAILED: PDF not finite or zero, pdf=" << pdf << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testSphericalAABBMultipleSamples() {
    std::cout << "Test: SphericalAABB multiple samples (robustness)... ";
    
    TriMesh *mesh = createTestMesh(30);
    Scene *scene = createTestScene(mesh);
    
    GeometryBVH bvh(4, GeometryBVHSamplingMode::SphericalAABB);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);
    
    SurfaceSample xs(Point(10.0f, 0.5f, 8.0f), Normal(0.0f, 0.0f, -1.0f));
    EmitterSample xe(Point(10.0f, 0.5f, 0.0f), Normal(0.0f, 0.0f, 1.0f));
    
    int successful_samples = 0;
    int failed_samples = 0;
    Float min_pdf = std::numeric_limits<Float>::infinity(), max_pdf = 0.0f;
    
    // Try 100 random samples
    ref<Random> rng = new Random(12345ULL);
    for (int i = 0; i < 100; ++i) {
        Point2 sample1(rng->nextFloat(), rng->nextFloat());
        Point2 sample2(rng->nextFloat(), rng->nextFloat());
        
        Intersection its;
        Float pdf;
        
        if (bvh.sampleGeometry(scene, xs, xe, sample1, sample2, its, pdf)) {
            if (isFinite(pdf)) {
                successful_samples++;
                min_pdf = std::min(min_pdf, pdf);
                max_pdf = std::max(max_pdf, pdf);
            } else {
                failed_samples++;
            }
        } else {
            failed_samples++;
        }
    }
    
    // We expect most samples to succeed
    if (successful_samples < 50) {
        std::cout << "FAILED: Only " << successful_samples 
                  << "/100 samples succeeded" << std::endl;
        return false;
    }
    
    if (min_pdf == std::numeric_limits<Float>::infinity() || max_pdf == 0.0f) {
        std::cout << "FAILED: Invalid PDF range" << std::endl;
        return false;
    }
    
    std::cout << "PASSED (" << successful_samples << "/100 succeeded, "
              << "PDF range: [" << std::scientific << std::setprecision(2)
              << min_pdf << ", " << max_pdf << "])" << std::endl;
    return true;
}

bool testComparisonPrimitiveVsSphericalAABB() {
    std::cout << "Test: Comparison Primitive vs SphericalAABB sampling... ";
    
    TriMesh *mesh = createTestMesh(40);
    Scene *scene = createTestScene(mesh);
    
    GeometryBVH bvh_primitive(4, GeometryBVHSamplingMode::Primitive);
    bvh_primitive.buildBVH(scene);
    bvh_primitive.buildAggregates(scene);
    
    GeometryBVH bvh_spherical(4, GeometryBVHSamplingMode::SphericalAABB);
    bvh_spherical.buildBVH(scene);
    bvh_spherical.buildAggregates(scene);
    
    SurfaceSample xs(Point(15.0f, 0.5f, 10.0f), Normal(0.0f, 0.0f, -1.0f));
    EmitterSample xe(Point(15.0f, 0.5f, 0.0f), Normal(0.0f, 0.0f, 1.0f));
    
    int primitive_success = 0, spherical_success = 0;
    Float primitive_avg_pdf = 0.0f, spherical_avg_pdf = 0.0f;
    
    ref<Random> rng = new Random(54321ULL);
    for (int i = 0; i < 50; ++i) {
        Point2 sample1(rng->nextFloat(), rng->nextFloat());
        Point2 sample2(rng->nextFloat(), rng->nextFloat());
        
        Intersection its1, its2;
        Float pdf1, pdf2;
        
        std::cout << "Sample " << i+1 << " geometry..." << std::flush;
        if (bvh_primitive.sampleGeometry(scene, xs, xe, sample1, sample2, its1, pdf1)) {
            if (isFinite(pdf1)) {
                primitive_success++;
                primitive_avg_pdf += pdf1;
            }
        }
        
        std::cout << "spherical..." << std::endl;
        
        if (bvh_spherical.sampleGeometry(scene, xs, xe, sample1, sample2, its2, pdf2)) {
            if (isFinite(pdf2)) {
                spherical_success++;
                spherical_avg_pdf += pdf2;
            }
        }
    }
    
    if (primitive_success > 0) {
        primitive_avg_pdf /= primitive_success;
    }
    if (spherical_success > 0) {
        spherical_avg_pdf /= spherical_success;
    }
    
    std::cout << "PASSED\n  Primitive: " << primitive_success << "/50 succeeded, "
              << "avg PDF=" << std::scientific << std::setprecision(2) 
              << primitive_avg_pdf << "\n  SphericalAABB: " << spherical_success 
              << "/50 succeeded, avg PDF=" << spherical_avg_pdf << std::endl;
    return true;
}

bool testDefaultConstructorUsesDefaultMode() {
    std::cout << "Test: Default constructor uses Primitive mode... ";
    
    // Create BVH with default parameters
    GeometryBVH bvh(4);
    
    if (bvh.getSamplingMode() != GeometryBVHSamplingMode::Primitive) {
        std::cout << "FAILED: Default mode is not Primitive" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// ============================================================================
// Main test runner
// ============================================================================

int main(int argc, char **argv) {
    Class::staticInitialization();
    Object::staticInitialization();
    PluginManager::staticInitialization();
    Thread::staticInitialization();
    FileStream::staticInitialization();
    Spectrum::staticInitialization();
    Logger::staticInitialization();
    Thread::getThread()->getLogger()->setLogLevel(EWarn);
    Scheduler::staticInitialization();
    {
        Scheduler *scheduler = Scheduler::getInstance();
        scheduler->registerWorker(new LocalWorker(-1, "wrk0"));
        scheduler->start();
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "BVH Phase 2.3 - Spherical AABB Integration Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int passed = 0, failed = 0;
    
    if (testDefaultConstructorUsesDefaultMode()) passed++; else failed++;
    if (testSphericalAABBSamplingModeCreation()) passed++; else failed++;
    if (testPrimitiveSamplingModeStillWorks()) passed++; else failed++;
    if (testSphericalAABBLeafSampling()) passed++; else failed++;
    if (testSphericalAABBMultipleSamples()) passed++; else failed++;
    if (testComparisonPrimitiveVsSphericalAABB()) passed++; else failed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    Scheduler::staticShutdown();
    
    return failed;
}

MTS_NAMESPACE_END

int main(int argc, char **argv) {
    return mitsuba::main(argc, argv);
}
