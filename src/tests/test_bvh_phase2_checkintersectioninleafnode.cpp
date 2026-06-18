#include <mitsuba/render/bvh/bvh.h>
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
#include <vector>
#include <limits>
#include <cmath>
#include <iomanip>

MTS_NAMESPACE_BEGIN

struct TriangleDef {
    Point p0;
    Point p1;
    Point p2;
};

static TriMesh* createMeshFromTriangles(const std::vector<TriangleDef> &tris, Float albedo = 0.8f) {
    const size_t numTriangles = tris.size();
    const size_t numVerts = numTriangles * 3;

    TriMesh *mesh = new TriMesh("test_mesh", numTriangles, numVerts,
        true, false, false, false, false);

    Point *vertices = mesh->getVertexPositions();
    Normal *normals = mesh->getVertexNormals();
    Triangle *triangles = mesh->getTriangles();

    for (size_t i = 0; i < numTriangles; ++i) {
        const size_t base = i * 3;
        vertices[base + 0] = tris[i].p0;
        vertices[base + 1] = tris[i].p1;
        vertices[base + 2] = tris[i].p2;
        normals[base + 0] = Normal(0.0f, 0.0f, 1.0f);
        normals[base + 1] = Normal(0.0f, 0.0f, 1.0f);
        normals[base + 2] = Normal(0.0f, 0.0f, 1.0f);

        triangles[i].idx[0] = static_cast<uint32_t>(base + 0);
        triangles[i].idx[1] = static_cast<uint32_t>(base + 1);
        triangles[i].idx[2] = static_cast<uint32_t>(base + 2);
    }

    Properties bsdfProps("diffuse");
    bsdfProps.setSpectrum("reflectance", Spectrum(albedo));
    BSDF *bsdf = static_cast<BSDF *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(BSDF), bsdfProps));
    bsdf->configure();
    mesh->addChild(bsdf);

    mesh->configure();
    return mesh;
}

static Scene* createTestScene(TriMesh *mesh) {
    Properties sceneProps("scene");
    Scene *scene = new Scene(sceneProps);
    scene->addChild(mesh);
    scene->configure();
    scene->initialize();
    return scene;
}

static bool isFinite(Float v) {
    return std::isfinite(v) && v > 0.0f;
}

static uint32_t findMeshIndex(const Scene *scene, const TriMesh *mesh) {
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    for (uint32_t i = 0; i < meshes.size(); ++i) {
        if (meshes[i] == mesh) {
            return i;
        }
    }
    return 0xFFFFFFFF;
}

static uint32_t findPrimitiveIndex(const GeometryBVH &bvh, uint32_t meshIndex, uint32_t triangleIndex) {
    const size_t primCount = bvh.getPrimitiveCount();
    for (size_t i = 0; i < primCount; ++i) {
        const BVHPrimitive &prim = bvh.getPrimitive(i);
        if (prim.meshIndex == meshIndex && prim.triangleIndex == triangleIndex) {
            return static_cast<uint32_t>(i);
        }
    }
    return 0xFFFFFFFF;
}

static size_t findLeafForPrimitive(const GeometryBVH &bvh, uint32_t primIdx) {
    const size_t nodeCount = bvh.getNodeCount();
    for (size_t i = 0; i < nodeCount; ++i) {
        const BVHNode &node = bvh.getNode(i);
        if (!node.isLeaf()) {
            continue;
        }
        const uint32_t start = node.primitivesOffset;
        const uint32_t end = start + node.nLeafPrimitives;
        if (primIdx >= start && primIdx < end) {
            return i;
        }
    }
    return std::numeric_limits<size_t>::max();
}

static std::vector<size_t> getLeafNodes(const GeometryBVH &bvh) {
    std::vector<size_t> leaves;
    const size_t nodeCount = bvh.getNodeCount();
    for (size_t i = 0; i < nodeCount; ++i) {
        const BVHNode &node = bvh.getNode(i);
        if (node.isLeaf()) {
            leaves.push_back(i);
        }
    }
    return leaves;
}

static bool aabbOverlaps(const AABB &a, const AABB &b) {
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
           (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
           (a.min.z <= b.max.z && a.max.z >= b.min.z);
}

static Intersection makeIntersection(const TriMesh *mesh, uint32_t triangleIndex) {
    Intersection its;
    its.shape = mesh;
    its.primIndex = triangleIndex;
    its.instance = NULL;
    return its;
}

// ============================================================================
// Test cases
// ============================================================================

static bool testNoOverlapSinglePrimitiveLeaf() {
    std::cout << "Test: No overlap, leaf size=1... ";

    std::vector<TriangleDef> tris = {
        { Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0) },
        { Point(10, 0, 0), Point(11, 0, 0), Point(10, 1, 0) }
    };

    TriMesh *mesh = createMeshFromTriangles(tris);
    Scene *scene = createTestScene(mesh);

    GeometryBVH bvh(1, GeometryBVHSamplingMode::SphericalAABB);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    const std::vector<size_t> leaves = getLeafNodes(bvh);
    if (leaves.size() != 2) {
        std::cout << "FAILED: Expected 2 leaves, got " << leaves.size() << std::endl;
        return false;
    }

    const AABB &leafA = bvh.getNode(leaves[0]).bounds;
    const AABB &leafB = bvh.getNode(leaves[1]).bounds;
    if (aabbOverlaps(leafA, leafB)) {
        std::cout << "FAILED: Expected no overlap between leaf bounds" << std::endl;
        return false;
    }

    uint32_t meshIndex = findMeshIndex(scene, mesh);
    if (meshIndex == 0xFFFFFFFF) {
        std::cout << "FAILED: Mesh not found in scene" << std::endl;
        return false;
    }

    for (uint32_t triIndex = 0; triIndex < 2; ++triIndex) {
        uint32_t primIdx = findPrimitiveIndex(bvh, meshIndex, triIndex);
        if (primIdx == 0xFFFFFFFF) {
            std::cout << "FAILED: Primitive not found for triangle " << triIndex << std::endl;
            return false;
        }
        size_t leafIdx = findLeafForPrimitive(bvh, primIdx);
        if (leafIdx == std::numeric_limits<size_t>::max()) {
            std::cout << "FAILED: Leaf not found for primitive " << primIdx << std::endl;
            return false;
        }

        Intersection its = makeIntersection(mesh, triIndex);
        if (!bvh.intersectionInLeafNode(scene, its, leafIdx)) {
            std::cout << "FAILED: Intersection not recognized in correct leaf" << std::endl;
            return false;
        }

        size_t otherLeaf = (leafIdx == leaves[0]) ? leaves[1] : leaves[0];
        if (bvh.intersectionInLeafNode(scene, its, otherLeaf)) {
            std::cout << "FAILED: Intersection incorrectly accepted in other leaf" << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

static bool testNoOverlapMultiPrimitiveLeaf() {
    std::cout << "Test: No overlap, leaf size>1... ";

    std::vector<TriangleDef> tris = {
        { Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0) },
        { Point(2, 0, 0), Point(3, 0, 0), Point(2, 1, 0) },
        { Point(10, 0, 0), Point(11, 0, 0), Point(10, 1, 0) },
        { Point(12, 0, 0), Point(13, 0, 0), Point(12, 1, 0) }
    };

    TriMesh *mesh = createMeshFromTriangles(tris);
    Scene *scene = createTestScene(mesh);

    GeometryBVH bvh(2, GeometryBVHSamplingMode::SphericalAABB);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    const std::vector<size_t> leaves = getLeafNodes(bvh);
    if (leaves.size() != 2) {
        std::cout << "FAILED: Expected 2 leaves, got " << leaves.size() << std::endl;
        return false;
    }

    const AABB &leafA = bvh.getNode(leaves[0]).bounds;
    const AABB &leafB = bvh.getNode(leaves[1]).bounds;
    if (aabbOverlaps(leafA, leafB)) {
        std::cout << "FAILED: Expected no overlap between leaf bounds" << std::endl;
        return false;
    }

    uint32_t meshIndex = findMeshIndex(scene, mesh);
    if (meshIndex == 0xFFFFFFFF) {
        std::cout << "FAILED: Mesh not found in scene" << std::endl;
        return false;
    }

    for (uint32_t triIndex = 0; triIndex < 4; ++triIndex) {
        uint32_t primIdx = findPrimitiveIndex(bvh, meshIndex, triIndex);
        if (primIdx == 0xFFFFFFFF) {
            std::cout << "FAILED: Primitive not found for triangle " << triIndex << std::endl;
            return false;
        }
        size_t leafIdx = findLeafForPrimitive(bvh, primIdx);
        if (leafIdx == std::numeric_limits<size_t>::max()) {
            std::cout << "FAILED: Leaf not found for primitive " << primIdx << std::endl;
            return false;
        }

        Intersection its = makeIntersection(mesh, triIndex);
        if (!bvh.intersectionInLeafNode(scene, its, leafIdx)) {
            std::cout << "FAILED: Intersection not recognized in correct leaf" << std::endl;
            return false;
        }

        size_t otherLeaf = (leafIdx == leaves[0]) ? leaves[1] : leaves[0];
        if (bvh.intersectionInLeafNode(scene, its, otherLeaf)) {
            std::cout << "FAILED: Intersection incorrectly accepted in other leaf" << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

static bool testOverlapSinglePrimitiveLeaf() {
    std::cout << "Test: Overlap, leaf size=1... ";

    std::vector<TriangleDef> tris = {
        { Point(0, 0, 0), Point(2, 0, 0), Point(0, 1, 0) },
        { Point(1, 0, 0), Point(3, 0, 0), Point(1, 1, 0) }
    };

    TriMesh *mesh = createMeshFromTriangles(tris);
    Scene *scene = createTestScene(mesh);

    GeometryBVH bvh(1, GeometryBVHSamplingMode::SphericalAABB);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    const std::vector<size_t> leaves = getLeafNodes(bvh);
    if (leaves.size() != 2) {
        std::cout << "FAILED: Expected 2 leaves, got " << leaves.size() << std::endl;
        return false;
    }

    const AABB &leafA = bvh.getNode(leaves[0]).bounds;
    const AABB &leafB = bvh.getNode(leaves[1]).bounds;
    if (!aabbOverlaps(leafA, leafB)) {
        std::cout << "FAILED: Expected overlap between leaf bounds" << std::endl;
        return false;
    }

    uint32_t meshIndex = findMeshIndex(scene, mesh);
    if (meshIndex == 0xFFFFFFFF) {
        std::cout << "FAILED: Mesh not found in scene" << std::endl;
        return false;
    }

    for (uint32_t triIndex = 0; triIndex < 2; ++triIndex) {
        uint32_t primIdx = findPrimitiveIndex(bvh, meshIndex, triIndex);
        if (primIdx == 0xFFFFFFFF) {
            std::cout << "FAILED: Primitive not found for triangle " << triIndex << std::endl;
            return false;
        }
        size_t leafIdx = findLeafForPrimitive(bvh, primIdx);
        if (leafIdx == std::numeric_limits<size_t>::max()) {
            std::cout << "FAILED: Leaf not found for primitive " << primIdx << std::endl;
            return false;
        }

        Intersection its = makeIntersection(mesh, triIndex);
        if (!bvh.intersectionInLeafNode(scene, its, leafIdx)) {
            std::cout << "FAILED: Intersection not recognized in correct leaf" << std::endl;
            return false;
        }

        size_t otherLeaf = (leafIdx == leaves[0]) ? leaves[1] : leaves[0];
        if (bvh.intersectionInLeafNode(scene, its, otherLeaf)) {
            std::cout << "FAILED: Intersection incorrectly accepted in other leaf" << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

static bool testOverlapMultiPrimitiveLeaf() {
    std::cout << "Test: Overlap, leaf size>1... ";

    std::vector<TriangleDef> tris;
    for (int i = 0; i < 4; ++i) {
        Float x = static_cast<Float>(i);
        tris.push_back({ Point(x, 0, 0), Point(x + 2, 0, 0), Point(x, 1, 0) });
    }

    TriMesh *mesh = createMeshFromTriangles(tris);
    Scene *scene = createTestScene(mesh);

    GeometryBVH bvh(2, GeometryBVHSamplingMode::SphericalAABB);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    const std::vector<size_t> leaves = getLeafNodes(bvh);
    if (leaves.size() != 2) {
        std::cout << "FAILED: Expected 2 leaves, got " << leaves.size() << std::endl;
        return false;
    }

    const AABB &leafA = bvh.getNode(leaves[0]).bounds;
    const AABB &leafB = bvh.getNode(leaves[1]).bounds;
    if (!aabbOverlaps(leafA, leafB)) {
        std::cout << "FAILED: Expected overlap between leaf bounds" << std::endl;
        return false;
    }

    uint32_t meshIndex = findMeshIndex(scene, mesh);
    if (meshIndex == 0xFFFFFFFF) {
        std::cout << "FAILED: Mesh not found in scene" << std::endl;
        return false;
    }

    for (uint32_t triIndex = 0; triIndex < 4; ++triIndex) {
        uint32_t primIdx = findPrimitiveIndex(bvh, meshIndex, triIndex);
        if (primIdx == 0xFFFFFFFF) {
            std::cout << "FAILED: Primitive not found for triangle " << triIndex << std::endl;
            return false;
        }
        size_t leafIdx = findLeafForPrimitive(bvh, primIdx);
        if (leafIdx == std::numeric_limits<size_t>::max()) {
            std::cout << "FAILED: Leaf not found for primitive " << primIdx << std::endl;
            return false;
        }

        Intersection its = makeIntersection(mesh, triIndex);
        if (!bvh.intersectionInLeafNode(scene, its, leafIdx)) {
            std::cout << "FAILED: Intersection not recognized in correct leaf" << std::endl;
            return false;
        }

        size_t otherLeaf = (leafIdx == leaves[0]) ? leaves[1] : leaves[0];
        if (bvh.intersectionInLeafNode(scene, its, otherLeaf)) {
            std::cout << "FAILED: Intersection incorrectly accepted in other leaf" << std::endl;
            return false;
        }
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

static bool testCompareSphericalAABBVsPrimitiveSampling() {
    std::cout << "Test: Compare SphericalAABB vs Primitive sampling... ";

    std::vector<TriangleDef> tris;
    for (int i = 0; i < 20; ++i) {
        Float x = static_cast<Float>(i);
        tris.push_back({ Point(x, 0, 0), Point(x + 1, 0, 0), Point(x, 1, 0) });
    }

    TriMesh *mesh = createMeshFromTriangles(tris);
    Scene *scene = createTestScene(mesh);

    GeometryBVH bvhPrimitive(4, GeometryBVHSamplingMode::Primitive);
    bvhPrimitive.buildBVH(scene);
    bvhPrimitive.buildAggregates(scene);

    GeometryBVH bvhSpherical(4, GeometryBVHSamplingMode::SphericalAABB);
    bvhSpherical.buildBVH(scene);
    bvhSpherical.buildAggregates(scene);

    SurfaceSample xs(Point(10.0f, 0.5f, 5.0f), Normal(0.0f, 0.0f, -1.0f));
    EmitterSample xe(Point(10.0f, 0.5f, 0.0f), Normal(0.0f, 0.0f, 1.0f));

    ref<Random> rng = new Random(1234ULL);

    int primitiveSuccess = 0;
    int sphericalSuccess = 0;
    Float primitiveAvgPdf = 0.0f;
    Float sphericalAvgPdf = 0.0f;

    for (int i = 0; i < 50; ++i) {
        Point2 sample1(rng->nextFloat(), rng->nextFloat());
        Point2 sample2(rng->nextFloat(), rng->nextFloat());

        Intersection its1, its2;
        Float pdf1 = 0.0f, pdf2 = 0.0f;

        if (bvhPrimitive.sampleGeometry(scene, xs, xe, sample1, sample2, its1, pdf1)) {
            if (isFinite(pdf1)) {
                primitiveSuccess++;
                primitiveAvgPdf += pdf1;
            }
        }

        if (bvhSpherical.sampleGeometry(scene, xs, xe, sample1, sample2, its2, pdf2)) {
            if (isFinite(pdf2)) {
                sphericalSuccess++;
                sphericalAvgPdf += pdf2;
            }
        }
    }

    if (primitiveSuccess > 0) {
        primitiveAvgPdf /= primitiveSuccess;
    }
    if (sphericalSuccess > 0) {
        sphericalAvgPdf /= sphericalSuccess;
    }

    if (primitiveSuccess < 20 || sphericalSuccess < 20) {
        std::cout << "FAILED: Too few successful samples (Primitive=" << primitiveSuccess
                  << ", Spherical=" << sphericalSuccess << ")" << std::endl;
        return false;
    }

    std::cout << "PASSED (Primitive avg PDF=" << std::scientific << std::setprecision(2)
              << primitiveAvgPdf << ", Spherical avg PDF=" << sphericalAvgPdf << ")" << std::endl;
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
    std::cout << "BVH Phase 2 - Intersection-in-Leaf Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int passed = 0, failed = 0;

    if (testNoOverlapSinglePrimitiveLeaf()) passed++; else failed++;
    if (testNoOverlapMultiPrimitiveLeaf()) passed++; else failed++;
    if (testOverlapSinglePrimitiveLeaf()) passed++; else failed++;
    if (testOverlapMultiPrimitiveLeaf()) passed++; else failed++;
    if (testCompareSphericalAABBVsPrimitiveSampling()) passed++; else failed++;

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
