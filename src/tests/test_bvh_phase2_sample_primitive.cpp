#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <iostream>
#include <cmath>

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

// ============================================================================
// Test functions
// ============================================================================

bool testBasicConstruction() {
    std::cout << "Test: Basic BVH Construction... ";
    
    // Create mesh with 20 triangles
    TriMesh *mesh = createTestMesh(20);
    Scene *scene = createTestScene(mesh);
    
    // Build BVH
    SamplingBVH bvh(4); // maxLeafSize = 4
    bvh.buildBVH(scene);
    
    // Verify primitives collected
    if (bvh.getPrimitiveCount() != 20) {
        std::cout << "FAILED: Expected 20 primitives, got " 
                  << bvh.getPrimitiveCount() << std::endl;
        return false;
    }
    
    // Verify nodes created
    if (bvh.getNodeCount() == 0) {
        std::cout << "FAILED: No nodes created" << std::endl;
        return false;
    }
    
    // Verify BVH is marked as built
    if (!bvh.isBuilt()) {
        std::cout << "FAILED: BVH not marked as built" << std::endl;
        return false;
    }
    
    std::cout << "PASSED (nodes=" << bvh.getNodeCount() 
              << ", prims=" << bvh.getPrimitiveCount() << ")" << std::endl;
    return true;
}

// ============================================================================
// PDF consistency test (forward vs. recomputed reverse PDF)
// ============================================================================

/// Check whether point P lies on triangle (A,B,C) within tolerance
static bool pointOnTriangle(const Point &P, const Point &A, const Point &B, const Point &C, Float eps = 1e-4f) {
    Vector v0 = B - A;
    Vector v1 = C - A;
    Vector v2 = P - A;

    const Float dot00 = dot(v0, v0);
    const Float dot01 = dot(v0, v1);
    const Float dot11 = dot(v1, v1);
    const Float dot02 = dot(v0, v2);
    const Float dot12 = dot(v1, v2);

    const Float denom = dot00 * dot11 - dot01 * dot01;
    if (std::abs(denom) < Epsilon) return false;

    const Float u = (dot11 * dot02 - dot01 * dot12) / denom;
    const Float v = (dot00 * dot12 - dot01 * dot02) / denom;

    if (u < -eps || v < -eps || (u + v) > 1.0f + eps)
        return false;

    // Plane distance
    Vector n = cross(v0, v1);
    const Float nlen = n.length();
    if (nlen < Epsilon) return false;
    n /= nlen;
    const Float dist = std::abs(dot(Vector(P - A), n));
    return dist <= eps;
}

/// Find the primitive index (in BVH primitive array) whose triangle contains P.
/// Returns true and sets outPrimIdx if found.
static bool findPrimitiveContainingPoint(const SamplingBVH &bvh, const Scene *scene, const Point &P, uint32_t &outPrimIdx) {
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    const size_t primCount = bvh.getPrimitiveCount();
    for (size_t i = 0; i < primCount; ++i) {
        const BVHPrimitive &prim = bvh.getPrimitive(i);
        const TriMesh *mesh = meshes[prim.meshIndex];
        const Triangle *tris = mesh->getTriangles();
        const Triangle &tri = tris[prim.triangleIndex];

        const Point *verts = mesh->getVertexPositions();
        const Point &A = verts[tri.idx[0]];
        const Point &B = verts[tri.idx[1]];
        const Point &C = verts[tri.idx[2]];

        if (pointOnTriangle(P, A, B, C)) {
            outPrimIdx = static_cast<uint32_t>(i);
            return true;
        }
    }
    return false;
}

/// Check whether the primitive index is contained in the subtree rooted at nodeIndex
static bool primitiveInSubtree(const SamplingBVH &bvh, size_t nodeIndex, uint32_t primIdx) {
    const BVHNode &node = bvh.getNode(nodeIndex);
    if (node.isLeaf()) {
        for (uint32_t i = 0; i < node.nLeafPrimitives; ++i) {
            if (bvh.getPrimitive(node.primitivesOffset + i).meshIndex == bvh.getPrimitive(primIdx).meshIndex &&
                bvh.getPrimitive(node.primitivesOffset + i).triangleIndex == bvh.getPrimitive(primIdx).triangleIndex) {
                return true;
            }
        }
        return false;
    }
    // interior: check both children
    size_t left = bvh.getLeftChild(nodeIndex);
    size_t right = bvh.getRightChild(nodeIndex);
    return primitiveInSubtree(bvh, left, primIdx) || primitiveInSubtree(bvh, right, primIdx);
}

bool testSamplePDFConsistency() {
    std::cout << "Test: sampleGeometry PDF consistency... ";

    TriMesh *mesh = createTestMesh(20);
    Scene *scene = createTestScene(mesh);

    SamplingBVH bvh(4);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    // Create context samples
    SurfaceSample xs(Point(0.0f, 0.5f, -1.0f), Normal(0.0f, 0.0f, 1.0f));
    EmitterSample xe(Point(0.0f, 0.5f, 2.0f), Normal(0.0f, 0.0f, -1.0f));

    Point2 samp(0.42f, 0.77f);
    Point2 samp2(0.33f, 0.66f);
    Intersection its; Float pdf_fwd = 0.0f;

    bool ok = bvh.sampleGeometry(scene, xs, xe, samp, samp2, its, pdf_fwd);
    Point pos = its.p;
    if (!ok) {
        std::cout << "FAILED: sampleGeometry returned false" << std::endl;
        return false;
    }

    if (!std::isfinite((double) pdf_fwd) || pdf_fwd <= 0.0f) {
        std::cout << "FAILED: sampleGeometry returned non-positive pdf: " << pdf_fwd << std::endl;
        return false;
    }

    // Find primitive containing pos
    uint32_t primIdx = (uint32_t) -1;
    if (!findPrimitiveContainingPoint(bvh, scene, pos, primIdx)) {
        std::cout << "FAILED: Could not locate primitive for sampled position" << std::endl;
        return false;
    }

    const BVHPrimitive &prim = bvh.getPrimitive(primIdx);

    // Recompute PDF using the new API
    Float pdf_rev = bvh.pdfGeometry(scene, xs, xe, prim.meshIndex, prim.triangleIndex, pos);

    // Relative error
    const Float denom = std::max(std::max(pdf_fwd, pdf_rev), (Float)1e-12f);
    const Float rel_err = std::abs(pdf_fwd - pdf_rev) / denom;

    const Float tol = 1e-6f;
    if (rel_err > tol) {
        std::cout << "FAILED: PDF mismatch (fwd=" << pdf_fwd << ", rev=" << pdf_rev << ", rel_err=" << rel_err << ")" << std::endl;
        return false;
    }

    std::cout << "PASSED (pdf=" << pdf_fwd << ", rel_err=" << rel_err << ")" << std::endl;
    return true;
}

// ===========================================================================
// Main test entry point
// ============================================================================

int runBVHPhase2SampleTests() {
    int failures = 0;

    if (!testBasicConstruction())
        failures++;

    if (!testSamplePDFConsistency())
        failures++;

    return failures;
}

int main(int argc, char **argv) {

    // Initialize Mitsuba
    Class::staticInitialization();
    Object::staticInitialization();
    PluginManager::staticInitialization();
    Thread::staticInitialization();
    FileStream::staticInitialization();
    Spectrum::staticInitialization();
    Logger::staticInitialization();
    Scheduler::staticInitialization();
    
    // Set up plugin path
    FileResolver *resolver = Thread::getThread()->getFileResolver();
    
    // Try common plugin locations
    fs::path pluginPath;
    if (argc > 1) {
        pluginPath = fs::path(argv[1]);
    } else {
        // Try to find plugins relative to executable
        pluginPath = fs::path("plugins");
        if (!fs::exists(pluginPath)) {
            pluginPath = fs::path("../plugins");
        }
        if (!fs::exists(pluginPath)) {
            pluginPath = fs::path("build/release");
        }
    }
    
    if (fs::exists(pluginPath)) {
        resolver->appendPath(pluginPath);
    }
    
    int result = runBVHPhase2SampleTests();
    
    // Cleanup
    Scheduler::staticShutdown();
    Logger::staticShutdown();
    Spectrum::staticShutdown();
    FileStream::staticShutdown();
    Thread::staticShutdown();
    PluginManager::staticShutdown();
    Object::staticShutdown();
    Class::staticShutdown();
    
    return result;
}

MTS_NAMESPACE_END

int main(int argc, char **argv) {
    return mitsuba::main(argc, argv);
}