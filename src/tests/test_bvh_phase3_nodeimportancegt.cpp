#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>

#include <iostream>
#include <vector>
#include <random>
#include <limits>
#include <cmath>

MTS_NAMESPACE_BEGIN

static TriMesh *createTestMesh(int numTriangles, Float albedo = 0.8f) {
    int numVerts = numTriangles + 2;
    TriMesh *mesh = new TriMesh("test_mesh", numTriangles, numVerts,
        true, false, false, false, false);

    Point *vertices = mesh->getVertexPositions();
    Normal *normals = mesh->getVertexNormals();

    for (int i = 0; i < numVerts; ++i) {
        Float x = static_cast<Float>(i) * 0.2f;
        Float y = (i % 2 == 0) ? 0.0f : 0.6f;
        vertices[i] = Point(x, y, 0.0f);
        normals[i] = Normal(0.0f, 0.0f, 1.0f);
    }

    Triangle *triangles = mesh->getTriangles();
    for (int i = 0; i < numTriangles; ++i) {
        triangles[i].idx[0] = i;
        triangles[i].idx[1] = i + 1;
        triangles[i].idx[2] = i + 2;
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

static Scene *createTestScene(TriMesh *mesh) {
    Properties sceneProps("scene");
    Scene *scene = new Scene(sceneProps);
    scene->addChild(mesh);
    scene->configure();
    scene->initialize();
    return scene;
}

static TriMesh *createOccluderQuad(Float z = 0.0f, Float extent = 1.5f, Float albedo = 0.5f) {
    TriMesh *mesh = new TriMesh("occluder_quad", 2, 4,
        true, false, false, false, false);

    Point *vertices = mesh->getVertexPositions();
    Normal *normals = mesh->getVertexNormals();
    Triangle *triangles = mesh->getTriangles();

    vertices[0] = Point(-extent, -extent, z);
    vertices[1] = Point( extent, -extent, z);
    vertices[2] = Point( extent,  extent, z);
    vertices[3] = Point(-extent,  extent, z);

    normals[0] = Normal(0.0f, 0.0f, 1.0f);
    normals[1] = Normal(0.0f, 0.0f, 1.0f);
    normals[2] = Normal(0.0f, 0.0f, 1.0f);
    normals[3] = Normal(0.0f, 0.0f, 1.0f);

    triangles[0].idx[0] = 0; triangles[0].idx[1] = 1; triangles[0].idx[2] = 2;
    triangles[1].idx[0] = 0; triangles[1].idx[1] = 2; triangles[1].idx[2] = 3;

    Properties bsdfProps("diffuse");
    bsdfProps.setSpectrum("reflectance", Spectrum(albedo));
    BSDF *bsdf = static_cast<BSDF *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(BSDF), bsdfProps));
    bsdf->configure();
    mesh->addChild(bsdf);

    mesh->configure();
    return mesh;
}

static Scene *createTestSceneWithOccluder(TriMesh *targetMesh, TriMesh *occluderMesh) {
    Properties sceneProps("scene");
    Scene *scene = new Scene(sceneProps);
    scene->addChild(targetMesh);
    scene->addChild(occluderMesh);
    scene->configure();
    scene->initialize();
    return scene;
}

static void collectSubtreePrims(const GeometryBVH *bvh, int nodeIndex, std::vector<uint32_t> &primitiveIndices)
{
    const BVHNode &node = bvh->getNode(nodeIndex);
    if (node.isLeaf())
    {
        const uint32_t start = node.primitivesOffset;
        const uint32_t end = start + node.nLeafPrimitives;
        for (uint32_t prim = start; prim < end; ++prim)
            primitiveIndices.push_back(prim);
        return;
    }

    const size_t leftChild = bvh->getLeftChild(nodeIndex);
    const size_t rightChild = bvh->getRightChild(nodeIndex);
    collectSubtreePrims(bvh, leftChild, primitiveIndices);
    collectSubtreePrims(bvh, rightChild, primitiveIndices);
}

static Float evalContributionProxy(
    const Point &p,
    const Normal &n,
    const SurfaceSample &xs,
    const EmitterSample &xe,
    bool checkVisibility,
    const Scene *scene)
{
    // Direction and distance: xs -> p
    Vector xsToP = p - xs.p;
    const Float distXsP = xsToP.length();
    if (distXsP <= Epsilon)
        return 0.0f;
    xsToP /= distXsP;

    // Direction and distance: p -> xe
    Vector pToXe = xe.p - p;
    const Float distPXe = pToXe.length();
    if (distPXe <= Epsilon)
        return 0.0f;
    pToXe /= distPXe;

    // Cosines at each vertex
    const Float cos_xs    = std::max((Float) 0.0f, dot(Vector(xs.n),  xsToP));
    const Float cos_x_in  = std::max((Float) 0.0f, dot(Vector(n),    -xsToP));
    const Float cos_x_out = std::max((Float) 0.0f, dot(Vector(n),     pToXe));
    const Float cos_xe    = std::max((Float) 0.0f, dot(Vector(xe.n), -pToXe));

    if (cos_xs <= 0.0f || cos_x_in <= 0.0f || cos_x_out <= 0.0f || cos_xe <= 0.0f)
        return 0.0f;

    if (checkVisibility && scene) {
        Ray shadowRay1(xs.p + xsToP * Epsilon, xsToP, Epsilon,
                       std::max((Float) 0.0f, distXsP - 2 * Epsilon), 0.0f);
        if (scene->rayIntersect(shadowRay1))
            return 0.0f;

        Ray shadowRay2(p + pToXe * Epsilon, pToXe, Epsilon,
                       std::max((Float) 0.0f, distPXe - 2 * Epsilon), 0.0f);
        if (scene->rayIntersect(shadowRay2))
            return 0.0f;
    }

    return (cos_xs * cos_x_in * cos_x_out * cos_xe)
           / (distXsP * distXsP * distPXe * distPXe);
}

static Float estimateImportanceExplicitOverPrims(const GeometryBVH &bvh,
    const Scene *scene, const std::vector<uint32_t> &primIndices,
    const BVHNodeInfo &nodeInfo, const SurfaceSample &xs, const EmitterSample &xe,
    int numSamples, bool checkVisibility) {
    if (!scene || primIndices.empty() || numSamples <= 0)
        return 0.0f;

    const std::vector<TriMesh *> &meshes = scene->getMeshes();

    struct WeightedPrim { uint32_t primIndex; Float area; };
    std::vector<WeightedPrim> weighted;
    weighted.reserve(primIndices.size());

    DiscreteDistribution areaDistr;
    Float totalArea = 0.0f;

    for (size_t i = 0; i < primIndices.size(); ++i) {
        const BVHPrimitive &prim = bvh.getPrimitive(primIndices[i]);
        const TriMesh *mesh = meshes[prim.meshIndex];
        const Triangle &tri = mesh->getTriangles()[prim.triangleIndex];
        const Float area = tri.surfaceArea(mesh->getVertexPositions());
        if (area <= Epsilon)
            continue;

        weighted.push_back({ primIndices[i], area });
        areaDistr.append(area);
        totalArea += area;
    }

    if (weighted.empty() || areaDistr.normalize() == 0.0f || totalArea <= Epsilon)
        return 0.0f;

    std::mt19937 rng(424242u + (uint32_t) primIndices.size() * 131u + (uint32_t) numSamples * 17u);
    std::uniform_real_distribution<Float> uni((Float) 0.0f, (Float) 1.0f);

    const Float invPi = (Float) (1.0 / M_PI);
    const Float bsdfProxy = std::max((Float) 0.0f, nodeInfo.getMeanDiffuseAlbedoLum()) * invPi;

    Float contributionSum = 0.0f;
    for (int s = 0; s < numSamples; ++s) {
        Float primProb;
        const size_t local = areaDistr.sample(uni(rng), primProb);
        const BVHPrimitive &prim = bvh.getPrimitive(weighted[local].primIndex);
        const TriMesh *mesh = meshes[prim.meshIndex];
        const Triangle &tri = mesh->getTriangles()[prim.triangleIndex];

        PositionSamplingRecord pRec;
        pRec.p = tri.sample(mesh->getVertexPositions(), mesh->getVertexNormals(),
            mesh->getVertexTexcoords(), pRec.n, pRec.uv, Point2(uni(rng), uni(rng)));

        contributionSum += bsdfProxy * evalContributionProxy(pRec.p, pRec.n, xs, xe, checkVisibility, scene);
    }

    return totalArea * (contributionSum / (Float) numSamples);
}

static size_t findFirstLeaf(const GeometryBVH &bvh) {
    for (size_t i = 0; i < bvh.getNodeCount(); ++i)
        if (bvh.getNode(i).isLeaf())
            return i;
    return std::numeric_limits<size_t>::max();
}

static size_t findFirstInteriorNonRoot(const GeometryBVH &bvh) {
    for (size_t i = 1; i < bvh.getNodeCount(); ++i)
        if (bvh.getNode(i).isInterior())
            return i;
    return std::numeric_limits<size_t>::max();
}

static bool testLeafAndInteriorSmoke() {
    std::cout << "Test: Ground-truth node importance on leaf/interior... ";

    TriMesh *mesh = createTestMesh(24, 0.7f);
    Scene *scene = createTestScene(mesh);

    GeometryBVH bvh(2);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    const size_t leafIdx = findFirstLeaf(bvh);
    if (leafIdx == std::numeric_limits<size_t>::max()) {
        std::cout << "FAILED: no leaf node found" << std::endl;
        return false;
    }

    size_t interiorIdx = findFirstInteriorNonRoot(bvh);
    if (interiorIdx == std::numeric_limits<size_t>::max())
        interiorIdx = 0;

    SurfaceSample xs(Point(0.3f, 0.3f, -1.0f), Normal(0.0f, 0.0f, 1.0f));
    EmitterSample xe(Point(0.7f, 0.2f, 1.0f), Normal(0.0f, 0.0f, -1.0f));

    const Float impLeaf = computeNodeImportanceGroundTruth(
        leafIdx, bvh.getNodeInfo(leafIdx), xs, xe, scene, 2048, false);
    const Float impInterior = computeNodeImportanceGroundTruth(
        interiorIdx, bvh.getNodeInfo(interiorIdx), xs, xe, scene, 2048, false);

    if (!std::isfinite((double) impLeaf) || impLeaf < 0.0f) {
        std::cout << "FAILED: leaf importance invalid (" << impLeaf << ")" << std::endl;
        return false;
    }

    if (!std::isfinite((double) impInterior) || impInterior < 0.0f) {
        std::cout << "FAILED: interior importance invalid (" << impInterior << ")" << std::endl;
        return false;
    }

    std::cout << "PASSED (leaf=" << impLeaf << ", interior=" << impInterior << ")" << std::endl;
    return true;
}

static bool testRootSimilarToExplicitWholeScene() {
    std::cout << "Test: Root GT importance similar to explicit whole-scene estimator... ";

    TriMesh *mesh = createTestMesh(28, 0.75f);
    Scene *scene = createTestScene(mesh);

    GeometryBVH bvh(2);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    SurfaceSample xs(Point(0.2f, 0.4f, -1.2f), Normal(0.0f, 0.0f, 1.0f));
    EmitterSample xe(Point(0.8f, 0.1f, 1.3f), Normal(0.0f, 0.0f, -1.0f));

    const int sampleCount = 8192;

    const Float impRootGT = computeNodeImportanceGroundTruth(
        0, bvh.getNodeInfo(0), xs, xe, scene, sampleCount, false);

    std::vector<uint32_t> rootPrims;
    collectSubtreePrims(&bvh, 0, rootPrims);

    const Float impExplicit = estimateImportanceExplicitOverPrims(
        bvh, scene, rootPrims, bvh.getNodeInfo(0), xs, xe, sampleCount, false);

    if (!std::isfinite((double) impRootGT) || !std::isfinite((double) impExplicit)) {
        std::cout << "FAILED: non-finite value(s): gt=" << impRootGT
                  << ", explicit=" << impExplicit << std::endl;
        return false;
    }

    const Float denom = std::max(std::max(std::abs(impRootGT), std::abs(impExplicit)), (Float) 1e-8f);
    const Float relErr = std::abs(impRootGT - impExplicit) / denom;

    // Monte Carlo vs Monte Carlo comparison: keep tolerance reasonably loose.
    const Float tol = 0.25f;
    if (relErr > tol) {
        std::cout << "FAILED: relErr too high (gt=" << impRootGT << ", explicit="
                  << impExplicit << ", relErr=" << relErr << ")" << std::endl;
        return false;
    }

    std::cout << "PASSED (gt=" << impRootGT << ", explicit=" << impExplicit
              << ", relErr=" << relErr << ")" << std::endl;
    return true;
}

static bool testVisibilityOcclusionEffect() {
    std::cout << "Test: Visibility check lowers root GT importance with occluder... ";

    // Target geometry sampled by root node
    TriMesh *targetMesh = createTestMesh(24, 0.75f);
    // Occluder placed between xs and target/xe paths
    TriMesh *occluderMesh = createOccluderQuad(0.0f, 2.0f, 0.5f);
    Scene *scene = createTestSceneWithOccluder(targetMesh, occluderMesh);

    GeometryBVH bvh(2);
    bvh.buildBVH(scene);
    bvh.buildAggregates(scene);

    SurfaceSample xs(Point(0.5f, 0.2f, -1.2f), Normal(0.0f, 0.0f, 1.0f));
    EmitterSample xe(Point(0.5f, 0.2f, 1.2f), Normal(0.0f, 0.0f, -1.0f));

    const int sampleCount = 8192;

    const Float impNoVis = computeNodeImportanceGroundTruth(
        0, bvh.getNodeInfo(0), xs, xe, scene, sampleCount, false);
    const Float impVis = computeNodeImportanceGroundTruth(
        0, bvh.getNodeInfo(0), xs, xe, scene, sampleCount, true);

    if (!std::isfinite((double) impNoVis) || !std::isfinite((double) impVis)) {
        std::cout << "FAILED: non-finite values (noVis=" << impNoVis
                  << ", vis=" << impVis << ")" << std::endl;
        return false;
    }

    if (impNoVis < 0.0f || impVis < 0.0f) {
        std::cout << "FAILED: negative values (noVis=" << impNoVis
                  << ", vis=" << impVis << ")" << std::endl;
        return false;
    }

    // Visibility-aware estimate should not exceed no-visibility estimate.
    if (impVis > impNoVis * (1.0f + 1e-3f)) {
        std::cout << "FAILED: visibility-aware importance should be <= no-visibility"
                  << " (noVis=" << impNoVis << ", vis=" << impVis << ")" << std::endl;
        return false;
    }

    const Float ratio = (impNoVis > (Float) 1e-8f) ? (impVis / impNoVis) : 0.0f;
    std::cout << "PASSED (noVis=" << impNoVis << ", vis=" << impVis
              << ", vis/noVis=" << ratio << ")" << std::endl;
    return true;
}

int runBVHPhase3NodeImportanceGTTests() {
    int failures = 0;

    if (!testLeafAndInteriorSmoke())
        failures++;

    if (!testRootSimilarToExplicitWholeScene())
        failures++;

    if (!testVisibilityOcclusionEffect())
        failures++;

    return failures;
}

int main(int argc, char **argv) {
    Class::staticInitialization();
    Object::staticInitialization();
    PluginManager::staticInitialization();
    Thread::staticInitialization();
    FileStream::staticInitialization();
    Spectrum::staticInitialization();
    Logger::staticInitialization();
    Scheduler::staticInitialization();

    int result = runBVHPhase3NodeImportanceGTTests();

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
