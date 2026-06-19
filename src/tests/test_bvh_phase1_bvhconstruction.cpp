/*
    Test suite for BVH construction (Phase 2)
    
    Tests:
    1. Basic BVH construction from single mesh
    2. Tree structure validation (interior/leaf nodes)
    3. Bounds correctness
    4. Primitive ordering
    5. Aggregate propagation
    6. Edge cases (empty, single triangle, degenerate)
*/

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

/// Check if a point is inside an AABB (with epsilon tolerance)
bool pointInAABB(const Point &p, const AABB &box, Float eps = 1e-5f) {
    return p.x >= box.min.x - eps && p.x <= box.max.x + eps &&
           p.y >= box.min.y - eps && p.y <= box.max.y + eps &&
           p.z >= box.min.z - eps && p.z <= box.max.z + eps;
}

/// Check if one AABB contains another
bool aabbContains(const AABB &outer, const AABB &inner, Float eps = 1e-5f) {
    return outer.min.x <= inner.min.x + eps &&
           outer.min.y <= inner.min.y + eps &&
           outer.min.z <= inner.min.z + eps &&
           outer.max.x >= inner.max.x - eps &&
           outer.max.y >= inner.max.y - eps &&
           outer.max.z >= inner.max.z - eps;
}

AABB primitiveBounds(const Scene *scene, const BVHPrimitive &prim) {
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    const TriMesh *mesh = meshes[prim.meshIndex];
    const Triangle &tri = mesh->getTriangles()[prim.triangleIndex];
    return tri.getAABB(mesh->getVertexPositions());
}

// ============================================================================
// Test functions
// ============================================================================

bool testBasicConstruction() {
    std::cout << "Test: Basic BVH Construction... ";
    
    // Create mesh with 8 triangles
    TriMesh *mesh = createTestMesh(8);
    Scene *scene = createTestScene(mesh);
    
    // Build BVH
    SamplingBVH bvh(4); // maxLeafSize = 4
    bvh.buildBVH(scene);
    
    // Verify primitives collected
    if (bvh.getPrimitiveCount() != 8) {
        std::cout << "FAILED: Expected 8 primitives, got " 
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

bool testTreeStructure() {
    std::cout << "Test: Tree Structure Validation... ";
    
    TriMesh *mesh = createTestMesh(16);
    Scene *scene = createTestScene(mesh);
    
    SamplingBVH bvh(4);
    bvh.buildBVH(scene);
    
    // Count leaf and interior nodes
    size_t leafCount = 0;
    size_t interiorCount = 0;
    size_t totalPrimsInLeaves = 0;
    
    for (size_t i = 0; i < bvh.getNodeCount(); ++i) {
        const BVHNode &node = bvh.getNode(i);
        if (node.isLeaf()) {
            leafCount++;
            totalPrimsInLeaves += node.nLeafPrimitives;
        } else {
            interiorCount++;
            
            // Verify left child is at i+1
            size_t leftChild = bvh.getLeftChild(i);
            if (leftChild != i + 1) {
                std::cout << "FAILED: Left child not at expected position" << std::endl;
                return false;
            }
            
            // Verify right child index is valid
            size_t rightChild = bvh.getRightChild(i);
            if (rightChild >= bvh.getNodeCount()) {
                std::cout << "FAILED: Right child index out of bounds" << std::endl;
                return false;
            }
        }
    }
    
    // All primitives should be in leaves
    if (totalPrimsInLeaves != bvh.getPrimitiveCount()) {
        std::cout << "FAILED: Primitive count mismatch (leaves=" 
                  << totalPrimsInLeaves << ", total=" << bvh.getPrimitiveCount() << ")" << std::endl;
        return false;
    }
    
    std::cout << "PASSED (interior=" << interiorCount 
              << ", leaves=" << leafCount << ")" << std::endl;
    return true;
}

bool testBoundsCorrectness() {
    std::cout << "Test: Bounds Correctness... ";
    
    TriMesh *mesh = createTestMesh(8);
    Scene *scene = createTestScene(mesh);
    
    SamplingBVH bvh(2);
    bvh.buildBVH(scene);
    
    // Check root bounds contain all primitives
    const AABB &rootBounds = bvh.getNode(0).bounds;
    
    for (size_t i = 0; i < bvh.getPrimitiveCount(); ++i) {
        const BVHPrimitive &prim = bvh.getPrimitive(i);
        const AABB primBounds = primitiveBounds(scene, prim);
        if (!aabbContains(rootBounds, primBounds)) {
            std::cout << "FAILED: Root bounds don't contain primitive " << i << std::endl;
            return false;
        }
    }
    
    // Check each interior node's bounds contain its children
    for (size_t i = 0; i < bvh.getNodeCount(); ++i) {
        const BVHNode &node = bvh.getNode(i);
        if (node.isInterior()) {
            const AABB &leftBounds = bvh.getNode(bvh.getLeftChild(i)).bounds;
            const AABB &rightBounds = bvh.getNode(bvh.getRightChild(i)).bounds;
            
            if (!aabbContains(node.bounds, leftBounds)) {
                std::cout << "FAILED: Node " << i << " bounds don't contain left child" << std::endl;
                return false;
            }
            if (!aabbContains(node.bounds, rightBounds)) {
                std::cout << "FAILED: Node " << i << " bounds don't contain right child" << std::endl;
                return false;
            }
        }
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testPrimitiveOrdering() {
    std::cout << "Test: Primitive Ordering in Leaves... ";
    
    TriMesh *mesh = createTestMesh(12);
    Scene *scene = createTestScene(mesh);
    
    SamplingBVH bvh(3);
    bvh.buildBVH(scene);
    
    // Verify each leaf's primitives are within the leaf's bounds
    for (size_t i = 0; i < bvh.getNodeCount(); ++i) {
        const BVHNode &node = bvh.getNode(i);
        if (node.isLeaf()) {
            for (uint32_t j = 0; j < node.nLeafPrimitives; ++j) {
                const BVHPrimitive &prim = bvh.getPrimitive(node.primitivesOffset + j);
                const AABB primBounds = primitiveBounds(scene, prim);
                if (!aabbContains(node.bounds, primBounds)) {
                    std::cout << "FAILED: Primitive not contained in leaf bounds" << std::endl;
                    return false;
                }
            }
        }
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testAggregateComputation() {
    std::cout << "Test: Aggregate Computation... ";
    
    TriMesh *mesh = createTestMesh(8, 0.5f);
    Scene *scene = createTestScene(mesh);
    
    SamplingBVH bvh(4);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);
    
    // Check root aggregate
    const BVHNodeInfo &rootInfo = bvh.getNodeInfo(0);
    
    if (!rootInfo.valid) {
        std::cout << "FAILED: Root aggregate not valid" << std::endl;
        return false;
    }
    
    if (rootInfo.surfaceArea <= 0.0f) {
        std::cout << "FAILED: Root surface area should be positive" << std::endl;
        return false;
    }
    
    // Albedo should be close to 0.5 (what we set)
    if (std::abs(rootInfo.getMeanDiffuseAlbedoLum() - 0.5f) > 0.1f) {
        std::cout << "FAILED: Root albedo unexpected: " << rootInfo.getMeanDiffuseAlbedoLum() << std::endl;
        return false;
    }
    
    // For interior nodes, check that aggregate is combination of children
    for (size_t i = 0; i < bvh.getNodeCount(); ++i) {
        const BVHNode &node = bvh.getNode(i);
        if (node.isInterior()) {
            const BVHNodeInfo &info = bvh.getNodeInfo(i);
            const BVHNodeInfo &leftInfo = bvh.getNodeInfo(bvh.getLeftChild(i));
            const BVHNodeInfo &rightInfo = bvh.getNodeInfo(bvh.getRightChild(i));
            
            // Surface area should be sum of children
            Float expectedArea = leftInfo.surfaceArea + rightInfo.surfaceArea;
            if (std::abs(info.surfaceArea - expectedArea) > 1e-4f) {
                std::cout << "FAILED: Interior node area mismatch" << std::endl;
                return false;
            }
        }
    }
    
    std::cout << "PASSED (rootArea=" << rootInfo.surfaceArea 
              << ", rootAlbedo=" << rootInfo.getMeanDiffuseAlbedoLum() << ")" << std::endl;
    return true;
}

bool testSingleTriangle() {
    std::cout << "Test: Single Triangle Edge Case... ";
    
    TriMesh *mesh = createTestMesh(1);
    Scene *scene = createTestScene(mesh);
    
    SamplingBVH bvh(4);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);
    
    // Should have exactly 1 node (leaf) with 1 primitive
    if (bvh.getNodeCount() != 1) {
        std::cout << "FAILED: Expected 1 node, got " << bvh.getNodeCount() << std::endl;
        return false;
    }
    
    if (bvh.getPrimitiveCount() != 1) {
        std::cout << "FAILED: Expected 1 primitive, got " << bvh.getPrimitiveCount() << std::endl;
        return false;
    }
    
    const BVHNode &root = bvh.getNode(0);
    if (!root.isLeaf()) {
        std::cout << "FAILED: Single triangle should create leaf node" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testLargeLeafSize() {
    std::cout << "Test: Large Leaf Size (all in one leaf)... ";
    
    TriMesh *mesh = createTestMesh(10);
    Scene *scene = createTestScene(mesh);
    
    // maxLeafSize larger than primitive count
    SamplingBVH bvh(100);
    bvh.buildBVH(scene);
    
    // Should have exactly 1 node
    if (bvh.getNodeCount() != 1) {
        std::cout << "FAILED: Expected 1 node, got " << bvh.getNodeCount() << std::endl;
        return false;
    }
    
    const BVHNode &root = bvh.getNode(0);
    if (!root.isLeaf()) {
        std::cout << "FAILED: Should be single leaf" << std::endl;
        return false;
    }
    
    if (root.nLeafPrimitives != 10) {
        std::cout << "FAILED: Leaf should contain all 10 primitives" << std::endl;
        return false;
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

bool testMaxLeafSizeParameter() {
    std::cout << "Test: MaxLeafSize Parameter Enforcement... ";
    
    TriMesh *mesh = createTestMesh(32);
    Scene *scene = createTestScene(mesh);
    
    SamplingBVH bvh(4);
    bvh.buildBVH(scene);
    
    // Check that no leaf has more than maxLeafSize primitives
    // (except degenerate cases which we don't have here)
    for (size_t i = 0; i < bvh.getNodeCount(); ++i) {
        const BVHNode &node = bvh.getNode(i);
        if (node.isLeaf() && node.nLeafPrimitives > bvh.getMaxLeafSize()) {
            std::cout << "FAILED: Leaf has " << node.nLeafPrimitives 
                      << " primitives, max is " << bvh.getMaxLeafSize() << std::endl;
            return false;
        }
    }
    
    std::cout << "PASSED" << std::endl;
    return true;
}

// ============================================================================
// Main test runner
// ============================================================================

int runBVHPhase1Tests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "BVH Phase 1 Tests: Tree Construction" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // Run tests
    if (testBasicConstruction()) passed++; else failed++;
    if (testTreeStructure()) passed++; else failed++;
    if (testBoundsCorrectness()) passed++; else failed++;
    if (testPrimitiveOrdering()) passed++; else failed++;
    if (testAggregateComputation()) passed++; else failed++;
    if (testSingleTriangle()) passed++; else failed++;
    if (testLargeLeafSize()) passed++; else failed++;
    if (testMaxLeafSizeParameter()) passed++; else failed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return failed;
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
    
    int result = runBVHPhase1Tests();
    
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