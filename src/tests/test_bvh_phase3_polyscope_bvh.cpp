//
// BVH phase-3 utility test:
// 1) Load a scene from XML
// 2) Build GeometryBVH + aggregates
// 3) Compute one Geometry PDF sample
// 4) Export one CSV per BVH level containing AABB and surface area
//

#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/logger.h>

#include <mitsuba/render/scene.h>
#include <mitsuba/render/scenehandler.h>
#include <mitsuba/render/trimesh.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

MTS_NAMESPACE_BEGIN

static ref<Scene> loadSceneAbsolute(const fs::path &scenePath) {
    ref<Scene> scene = SceneHandler::loadScene(scenePath);
    scene->configure();
    scene->initialize();
    return scene;
}

static Point triangleCentroid(const TriMesh *mesh, uint32_t triIdx) {
    const Triangle &tri = mesh->getTriangles()[triIdx];
    const Point *v = mesh->getVertexPositions();
    return (v[tri.idx[0]] + v[tri.idx[1]] + v[tri.idx[2]]) * (1.0f / 3.0f);
}

static Normal triangleNormal(const TriMesh *mesh, uint32_t triIdx) {
    const Triangle &tri = mesh->getTriangles()[triIdx];
    const Point *v = mesh->getVertexPositions();
    Vector e1 = v[tri.idx[1]] - v[tri.idx[0]];
    Vector e2 = v[tri.idx[2]] - v[tri.idx[0]];
    return Normal(normalize(cross(e1, e2)));
}

static void exportBvhLevelsToCsv(const GeometryBVH &bvh, const std::string &csvPrefix) {
    if (bvh.getNodeCount() == 0)
        throw std::runtime_error("exportBvhLevelsToCsv(): BVH has no nodes");

    // BFS over node indices by level
    std::vector<size_t> currentLevel;
    currentLevel.push_back(0);
    size_t level = 0;

    while (!currentLevel.empty()) {
        std::ostringstream filename;
        filename << csvPrefix << "_level_" << level << ".csv";

        std::ofstream out(filename.str().c_str());
        if (!out)
            throw std::runtime_error("Unable to open CSV for writing: " + filename.str());

        out << "level,node_index,is_leaf,left_child,right_child,"
               "aabb_min_x,aabb_min_y,aabb_min_z,aabb_max_x,aabb_max_y,aabb_max_z,surface_area\n";

        std::vector<size_t> nextLevel;
        for (size_t i = 0; i < currentLevel.size(); ++i) {
            const size_t nodeIdx = currentLevel[i];
            const BVHNode &node = bvh.getNode(nodeIdx);
            const BVHNodeInfo &info = bvh.getNodeInfo(nodeIdx);

            long long leftChild = -1;
            long long rightChild = -1;

            if (!node.isLeaf()) {
                leftChild = (long long) bvh.getLeftChild(nodeIdx);
                rightChild = (long long) bvh.getRightChild(nodeIdx);
                nextLevel.push_back((size_t) leftChild);
                nextLevel.push_back((size_t) rightChild);
            }

            out << level << ","
                << nodeIdx << ","
                << (node.isLeaf() ? 1 : 0) << ","
                << leftChild << ","
                << rightChild << ","
                << node.bounds.min.x << ","
                << node.bounds.min.y << ","
                << node.bounds.min.z << ","
                << node.bounds.max.x << ","
                << node.bounds.max.y << ","
                << node.bounds.max.z << ","
                << info.surfaceArea << "\n";
        }

        out.close();
        std::cout << "Wrote " << filename.str() << " (" << currentLevel.size() << " nodes)" << std::endl;

        currentLevel.swap(nextLevel);
        ++level;
    }
}

static bool exportBvhCsvsFromXml(const fs::path &sceneXmlPath, const std::string &csvPrefix) {
    FileResolver *resolver = Thread::getThread()->getFileResolver();
    const fs::path absXml = resolver->resolveAbsolute(sceneXmlPath);
    resolver->appendPath(absXml.parent_path());

    ref<Scene> scene = loadSceneAbsolute(absXml);

    GeometryBVH bvh;
    bvh.buildBVH(scene.get());
    bvh.buildAggregates(scene.get());

    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    if (meshes.empty() || meshes[0]->getTriangleCount() == 0)
        throw std::runtime_error("Scene has no triangles for Geometry PDF sample");

    const TriMesh *mesh0 = meshes[0];
    const uint32_t triIdx = 0;
    const Point samplePos = triangleCentroid(mesh0, triIdx);
    const Normal sampleN = triangleNormal(mesh0, triIdx);

    SurfaceSample xs(samplePos, sampleN, mesh0->getBSDF());

    PositionSamplingRecord pRec;
    Spectrum eVal = scene->sampleEmitterPosition(pRec, Point2(0.5f, 0.5f));
    if (eVal.isZero())
        throw std::runtime_error("sampleEmitterPosition() returned zero; cannot form EmitterSample");

    EmitterSample xe(pRec.p, pRec.n);

    Float geomPdf = bvh.pdfGeometry(scene.get(), xs, xe, 0, triIdx, samplePos);
    std::cout << "Geometry PDF sample (mesh=0, tri=0): " << geomPdf << std::endl;

    exportBvhLevelsToCsv(bvh, csvPrefix);
    return true;
}

int main(int argc, char **argv) {
    int result = 0;

    Class::staticInitialization();
    Object::staticInitialization();
    PluginManager::staticInitialization();
    Thread::staticInitialization();
    FileStream::staticInitialization();
    Spectrum::staticInitialization();
    Logger::staticInitialization();
    Thread::getThread()->getLogger()->setLogLevel(EWarn);
    Bitmap::staticInitialization();
    Scheduler::staticInitialization();
    {
        Scheduler *scheduler = Scheduler::getInstance();
        scheduler->registerWorker(new LocalWorker(-1, "wrk0"));
        scheduler->start();
    }
    SceneHandler::staticInitialization();

    ref<FileResolver> fileResolver = new FileResolver();
    fileResolver->appendPath(".");
    Thread::getThread()->setFileResolver(fileResolver);

    try {
        const fs::path sceneXml = argc > 1
            ? fs::path(argv[1])
            : fs::path("scenes/aabb_spherical_sampling/path_scene_areaprojected_emitter.xml");

        const std::string csvPrefix = argc > 2
            ? std::string(argv[2])
            : std::string("bvh_nodes");

        std::cout << "Loading scene XML: " << sceneXml.string() << std::endl;
        std::cout << "CSV prefix: " << csvPrefix << std::endl;

        exportBvhCsvsFromXml(sceneXml, csvPrefix);
    } catch (const std::exception &e) {
        std::cout << "BVH export failed: " << e.what() << std::endl;
        result = 1;
    }

    SceneHandler::staticShutdown();
    Scheduler::staticShutdown();
    Bitmap::staticShutdown();
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
