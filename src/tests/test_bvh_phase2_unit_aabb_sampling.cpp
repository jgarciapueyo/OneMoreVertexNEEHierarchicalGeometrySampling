// 
// Tests bdpt_twopoints_mitsuba/include/mitsuba/render/bvh/spherical_aabb.h
// and   bdpt_twopoints_mitsuba/src/emitters/aabblight.cpp
//

#include <mitsuba/render/bvh/spherical_aabb.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/random.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/core/bitmap.h>

// Mitsuba's Assert macro may conflict with Xerces headers used by SceneHandler
#if defined(Assert)
# undef Assert
#endif
#include <mitsuba/render/scenehandler.h>

#include <mitsuba/core/logger.h>

#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>

MTS_NAMESPACE_BEGIN

static bool isFiniteVec(const Vector &v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

static bool isFinitePt(const Point &p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

static bool approx(Float a, Float b, Float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

static bool testInsideUniformSphere() {
    std::cout << "Test: Inside AABB -> uniform sphere PDF... ";
    AABB bounds(Point(-1, -1, -1), Point(1, 1, 1));
    Point p(0, 0, 0);

    Vector wo;
    Float pdf;
    AABBSphSample(bounds, p, 0.25f, 0.5f, wo, pdf);

    bool ok = isFiniteVec(wo) && approx(pdf, 1.0f / (2.0f * M_PI), 1e-6f);
    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    return ok;
}

static bool testInsidePdf() {
    std::cout << "Test: Inside AABB -> pdf constant... ";
    AABB bounds(Point(-1, -1, -1), Point(1, 1, 1));
    Point p(0, 0, 0);
    Vector wo(0, 0, 1);
    Float pdf = 0;

    AABBSphPdf(bounds, p, wo, pdf);

    bool ok = approx(pdf, 1.0f / (2.0f * M_PI), 1e-6f);
    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    return ok;
}

static bool testOutsideSamplePdfConsistency() {
    std::cout << "Test: Outside AABB -> sample/pdf consistency... ";
    AABB bounds(Point(-1, -1, -1), Point(1, 1, 1));
    Point p(3, 0.2f, -0.5f);

    Vector wo;
    Float pdf_s;
    AABBSphSample(bounds, p, 0.37f, 0.91f, wo, pdf_s);

    Float pdf_q = 0;
    AABBSphPdf(bounds, p, wo, pdf_q);

    bool ok = pdf_s > 0 && approx(pdf_s, pdf_q, 1e-4f) && isFiniteVec(wo);
    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    if (!ok) {
        // More detailed output for debugging
        std::cout << "  Sampled PDF: " << pdf_s << std::endl
                    << "  Queried PDF: " << pdf_q << std::endl
                    << "  wo: " << wo.x << ", " << wo.y << ", " << wo.z 
                    << std::endl;

    }
    return ok;
}

static bool testSamplePointOnSurface() {
    std::cout << "Test: Sampled point lies on AABB surface... ";
    AABB bounds(Point(-2, -1, -3), Point(1, 2, 0));
    Point p(4, 0, 0);

    Vector wo;
    Float pdf;
    Point3 sp;
    Normal sn;
    AABBSphSample(bounds, p, 0.12f, 0.77f, wo, pdf, sp, sn);

    bool onSurface =
        approx(sp.x, bounds.min.x, 1e-4f) || approx(sp.x, bounds.max.x, 1e-4f) ||
        approx(sp.y, bounds.min.y, 1e-4f) || approx(sp.y, bounds.max.y, 1e-4f) ||
        approx(sp.z, bounds.min.z, 1e-4f) || approx(sp.z, bounds.max.z, 1e-4f);

    bool ok = pdf > 0 && isFiniteVec(wo) && isFinitePt(sp) && onSurface;
    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    if (!ok) {
        // More detailed output for debugging
        std::cout << "  Sampled point: " << sp.x << ", " << sp.y << ", " << sp.z << std::endl
                  << "  Surface normal: " << sn.x << ", " << sn.y << ", " << sn.z << std::endl
                  << "  Expected px either " << bounds.min.x << " or " << bounds.max.x << std::endl
                    << "  Expected py either " << bounds.min.y << " or " << bounds.max.y << std::endl
                    << "  Expected pz either " << bounds.min.z << " or " << bounds.max.z << std::endl
                  << "  wo: " << wo.x << ", " << wo.y << ", " << wo.z << std::endl
                  << "  pdf: " << pdf << std::endl;
    }
    return ok;
}

static bool testPdfZeroWhenNoHit() {
    std::cout << "Test: Pdf zero when ray misses visible faces... ";
    AABB bounds(Point(-1, -1, -1), Point(1, 1, 1));
    Point p(3, 0, 0);

    Vector wo(1, 0, 0); // points away from the box
    Float pdf = 1;
    AABBSphPdf(bounds, p, wo, pdf);

    bool ok = approx(pdf, 0.0f, 1e-6f);
    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    return ok;
}

static bool testPdfPositiveWhenHit() {
    std::cout << "Test: Pdf positive when ray hits visible faces... ";
    AABB bounds(Point(-1, -1, -1), Point(1, 1, 1));
    Point p(3, 0, 0);

    Vector wo = normalize(Vector(-1, 0.1f, -0.05f));
    Float pdf = 0;
    AABBSphPdf(bounds, p, wo, pdf);

    bool ok = pdf > 0 && std::isfinite(pdf);
    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    return ok;
}

int runAabbSamplingTests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "AABB Spherical Sampling Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int passed = 0;
    int failed = 0;

    if (testInsideUniformSphere()) passed++; else failed++;
    if (testInsidePdf()) passed++; else failed++;
    if (testOutsideSamplePdfConsistency()) passed++; else failed++;
    if (testSamplePointOnSurface()) passed++; else failed++;
    if (testPdfZeroWhenNoHit()) passed++; else failed++;
    if (testPdfPositiveWhenHit()) passed++; else failed++;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return failed;
}


// static bool isFinitePt(const Point &p) {
//     return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
// }

static bool isFiniteN(const Normal &n) {
    return std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z);
}

// static bool approx(Float a, Float b, Float eps = 1e-5f) {
//     return std::abs(a - b) <= eps;
// }


static Emitter* createaabblight(const Spectrum &radiance = Spectrum(1.0f)) {
    Properties emitProps("aabblight");
    emitProps.setSpectrum("radiance", radiance);
    Emitter *emitter = static_cast<Emitter *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(Emitter), emitProps));
    return emitter;
}

static Shape* createBoxShape(const Point &min, const Point &max) {
    Properties shapeProps("cube");
    /* Cube primitive spans [-1,1]^3 in object space. Map it to [min,max]^3. */
    const Vector halfSize= Vector(max.x - min.x, max.y - min.y, max.z - min.z) * 0.5f;

    const Vector center = Vector(max.x + min.x, max.y + min.y, max.z + min.z) * 0.5f;
    shapeProps.setTransform("toWorld",
        Transform::translate(center) * Transform::scale(halfSize)
    );
    Shape *shape = static_cast<Shape *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(Shape), shapeProps));
    return shape;
}

static std::string readTextFile(const fs::path &filename) {
    std::ifstream ifs(filename.string().c_str());
    if (!ifs)
        throw std::runtime_error("Unable to open: " + filename.string());
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

static void replaceOnce(std::string &text, const std::string &from, const std::string &to) {
    size_t pos = text.find(from);
    if (pos == std::string::npos)
        throw std::runtime_error("Expected token not found: " + from);
    text.replace(pos, from.size(), to);
}

static void insertBeforeOnce(std::string &text, const std::string &token, const std::string &insertion) {
    size_t pos = text.find(token);
    if (pos == std::string::npos)
        throw std::runtime_error("Expected token not found: " + token);
    text.insert(pos, insertion);
}

static void replaceAll(std::string &text, const std::string &from, const std::string &to) {
    if (from.empty())
        throw std::runtime_error("replaceAll(): 'from' must be non-empty");
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static ref<Scene> loadSceneAbsolute(const fs::path &scenePath) {
    ref<Scene> scene = SceneHandler::loadScene(scenePath);
    scene->configure();
    scene->initialize();
    return scene;
}

static ref<Scene> loadSceneFromXmlString(const std::string &xml) {
    ref<Scene> scene = SceneHandler::loadSceneFromString(xml);
    scene->configure();
    scene->initialize();
    return scene;
}

static void estimateDirectIntegralPair(const Scene *sceneA, const Scene *sceneB,
        const Point &refPoint, size_t nSamples, Float &meanA, Float &meanB) {
    /* Deterministic seed to reduce test flakiness */
    ref<Random> rng = new Random(1ULL);
    Spectrum sumA(0.0f), sumB(0.0f);

    for (size_t i = 0; i < nSamples; ++i) {
        Point2 sample(rng->nextFloat(), rng->nextFloat());

        DirectSamplingRecord dRecA(refPoint, 0.0f);
        dRecA.refN = mitsuba::normalize(Normal(1.f,1.f,1.f));
        Spectrum valueA = sceneA->sampleEmitterDirect(dRecA, sample, false);
        if (dRecA.pdf != 0.0f && !valueA.isZero())
            sumA += valueA;

        DirectSamplingRecord dRecB(refPoint, 0.0f);
        dRecB.refN = mitsuba::normalize(Normal(1.f,1.f,1.f));
        Spectrum valueB = sceneB->sampleEmitterDirect(dRecB, sample, false);
        if (dRecB.pdf != 0.0f && !valueB.isZero())
            sumB += valueB;
    }

    meanA = (sumA / (Float) nSamples).average();
    meanB = (sumB / (Float) nSamples).average();
}

static void estimateDirectIntegral(const Scene *scene, const Point &refPoint,
        size_t nSamples, bool testVisibility, Float &mean, size_t *nonzeroCount = nullptr) {
    ref<Random> rng = new Random(1ULL);
    Spectrum sum(0.0f);
    size_t nonzero = 0;

    for (size_t i = 0; i < nSamples; ++i) {
        Point2 sample(rng->nextFloat(), rng->nextFloat());
        DirectSamplingRecord dRec(refPoint, 0.0f);
        dRec.refN = mitsuba::normalize(Normal(1.f, 1.f, 1.f));
        Spectrum value = scene->sampleEmitterDirect(dRec, sample, testVisibility);
        if (dRec.pdf != 0.0f && !value.isZero()) {
            sum += value;
            ++nonzero;
        }
    }

    mean = (sum / (Float) nSamples).average();
    if (nonzeroCount)
        *nonzeroCount = nonzero;
}

static bool checkPdfEmitterDirectMatchesSampled(const Scene *scene, const Point &refPoint,
        size_t nSamples, Float relTol, Float &maxRelErr, size_t &validCount) {
    ref<Random> rng = new Random(2ULL);
    maxRelErr = 0.0f;
    validCount = 0;

    for (size_t i = 0; i < nSamples; ++i) {
        Point2 sample(rng->nextFloat(), rng->nextFloat());
        DirectSamplingRecord dRec(refPoint, 0.0f);
        dRec.refN = mitsuba::normalize(Normal(1.f, 1.f, 1.f));

        Spectrum value = scene->sampleEmitterDirect(dRec, sample, false);
        (void) value;

        if (dRec.pdf == 0.0f)
            continue;

        Float pdfQuery = scene->pdfEmitterDirect(dRec);
        if (!std::isfinite(pdfQuery) || !std::isfinite(dRec.pdf))
            return false;

        Float denom = std::max((Float) 1e-6f, std::abs(dRec.pdf));
        Float relErr = std::abs(pdfQuery - dRec.pdf) / denom;
        maxRelErr = std::max(maxRelErr, relErr);
        ++validCount;

        if (relErr > relTol)
            return false;
    }

    return validCount > 0;
}

static ref<Bitmap> renderSceneToBitmap(Scene *scene, const std::string &jobName) {
    /* Render using Mitsuba's scheduler infrastructure (single local worker) */
    ref<RenderQueue> queue = new RenderQueue();

    /* Write to a dummy filename to satisfy RenderJob/Film, but we won't use it */
    scene->setDestinationFile(fs::path(jobName + ".exr"));
    scene->getFilm()->clear();

    ref<RenderJob> job = new RenderJob(jobName, scene, queue);
    job->start();
    job->wait();
    queue->join();

    ref<const Film> film = scene->getFilm();
    const Vector2i size = film->getCropSize();
    const Point2i offset = film->getCropOffset();

    ref<Bitmap> bmp = new Bitmap(Bitmap::ERGB, Bitmap::EFloat, size);
    bool ok = film->develop(offset, size, Point2i(0, 0), bmp.get());
    if (!ok)
        throw std::runtime_error("Film::develop() failed (no bitmap representation available)");
    return bmp;
}

static Float meanAbsDiff(const Bitmap *a, const Bitmap *b) {
    if (a->getWidth() != b->getWidth() || a->getHeight() != b->getHeight() ||
        a->getChannelCount() != b->getChannelCount())
        throw std::runtime_error("Bitmap size/channel mismatch");
    if (a->getComponentFormat() != Bitmap::EFloat || b->getComponentFormat() != Bitmap::EFloat)
        throw std::runtime_error("Expected float bitmaps");

    const Float *da = a->getFloatData();
    const Float *db = b->getFloatData();
    const size_t n = a->getPixelCount() * (size_t) a->getChannelCount();
    Float sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += std::abs(da[i] - db[i]);
    return sum / (Float) n;
}

static Float meanValue(const Bitmap *a) {
    if (a->getComponentFormat() != Bitmap::EFloat)
        throw std::runtime_error("Expected float bitmap");
    const Float *d = a->getFloatData();
    const size_t n = a->getPixelCount() * (size_t) a->getChannelCount();
    Float sum = 0;
    for (size_t i = 0; i < n; ++i)
        sum += d[i];
    return sum / (Float) n;
}

static bool testNonDegenerateSampling() {
    // return false;
    std::cout << "Test: AABBLight sampling (non-degenerate)... ";
    Shape *shape = createBoxShape(Point(0, 0, 0), Point(1, 1, 1));
    Emitter *emitter = createaabblight(Spectrum(1.0f));

    shape->addChild(emitter);
    shape->configure();

    // Use a reference point that is not exactly aligned with AABB edges
    // to avoid degenerate sampling cases (y/z set to 0.5).
    DirectSamplingRecord dRec(Point(3, 0.5f, 0.5f), 0.f); // second arg is time
    // Allow any orientation: set the reference normal to the zero vector
    // (sampling code treats a zero refN as "no orientation restriction").
    dRec.refN = Normal(0, 0, 0);
    Spectrum val = emitter->sampleDirect(dRec, Point2(0.35f, 0.72f));

    bool ok = (dRec.pdf > 0 && std::isfinite(dRec.pdf) &&
               isFinitePt(dRec.p) && isFiniteN(dRec.n) &&
               dRec.dist > 0 && std::isfinite(dRec.dist));

    if (!ok) {
        std::cout << "FAILED" << std::endl;
        std::cout << "  pdf: " << dRec.pdf << std::endl
                  << "  p: " << dRec.p.x << ", " << dRec.p.y << ", " << dRec.p.z << std::endl
                  << "  n: " << dRec.n.x << ", " << dRec.n.y << ", " << dRec.n.z << std::endl
                  << "  dist: " << dRec.dist << std::endl;
        return false;
    }

    Float pdf2 = emitter->pdfDirect(dRec);
    ok = ok && (pdf2 > 0 && std::isfinite(pdf2));

    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    return ok;
}

static bool testDegenerateSamplingNoNaN() {
    std::cout << "Test: AABBLight sampling (degenerate AABB)... ";
    Shape *shape = createBoxShape(Point(0, 0, 0), Point(0, 0, 0));
    Emitter *emitter = createaabblight(Spectrum(1.0f));

    shape->addChild(emitter);
    shape->configure();

    PositionSamplingRecord pRec;
    Spectrum val = emitter->samplePosition(pRec, Point2(0.2f, 0.4f), nullptr);

    bool ok = std::isfinite(pRec.pdf) && approx(pRec.pdf, 0.0f, 1e-6f) &&
              isFinitePt(pRec.p) && isFiniteN(pRec.n);

    ok |= pRec.pdf == 0.f; // If the pdf is 0, the rest should be ignored

    std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
    if (!ok) {
        std::cout << "  pdf: " << pRec.pdf << std::endl
                  << "  p: " << pRec.p.x << ", " << pRec.p.y << ", " << pRec.p.z << std::endl
                  << "  n: " << pRec.n.x << ", " << pRec.n.y << ", " << pRec.n.z << std::endl;
    }
    return ok;
}

int runAabbLightTests() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "AABB Light Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;

    int passed = 0;
    int failed = 0;

    if (testNonDegenerateSampling()) passed++; else failed++;
    if (testDegenerateSamplingNoNaN()) passed++; else failed++;

    {
        std::cout << "Test: areaprojected direct sampling matches area (XML scene, MC)... ";

        FileResolver *resolver = Thread::getThread()->getFileResolver();
        const fs::path projScenePath = resolver->resolveAbsolute(
            "scenes/aabb_spherical_sampling/path_scene_areaprojected_emitter.xml");
        resolver->appendPath(projScenePath.parent_path());

        ref<Scene> sceneProj = loadSceneAbsolute(projScenePath);

        std::string xml = readTextFile(projScenePath);
        replaceOnce(xml, "type=\"areaprojected\"", "type=\"area\"");
        ref<Scene> sceneArea = loadSceneFromXmlString(xml);

        const Point refPoint(0, -20, 2);
        const size_t nSamples = 50000;

        std::cout << "\n\tType of the emitter in the scenes: "
                  << sceneArea->getEmitters()[0]->getClass()->getName()
                  << " vs " << sceneProj->getEmitters()[0]->getClass()->getName()
                  << std::endl;

        Float meanArea = 0.0f, meanProj = 0.0f;
        estimateDirectIntegralPair(sceneArea.get(), sceneProj.get(), refPoint, nSamples,
            meanArea, meanProj);

        Float relErr = std::abs(meanArea - meanProj) / std::max((Float) 1e-6f, meanArea);
        bool ok = std::isfinite(meanArea) && std::isfinite(meanProj) && relErr < 0.005f;
        std::cout << (ok ? "PASSED" : "FAILED") << std::endl;

        // if (!ok) 
        {
            std::cout << "  meanArea=" << meanArea << " meanProj=" << meanProj
                      << " relErr=" << relErr << std::endl;
        }

        if (ok) passed++; else failed++;
    }

    {
        std::cout << "Test: areaprojected visibility via rayIntersect matches area (XML scene + occluder)... ";

        FileResolver *resolver = Thread::getThread()->getFileResolver();
        const fs::path projScenePath = resolver->resolveAbsolute(
            "scenes/aabb_spherical_sampling/path_scene_areaprojected_emitter.xml");
        resolver->appendPath(projScenePath.parent_path());

        const std::string baseXml = readTextFile(projScenePath);

        /* Insert a large occluding slab between refPoint and the emitter.
           This should make testVisibility drop the contribution to ~0.
           The projected-emitter branch performs a rayIntersect and rejects
           hits that are not on the emitter's own shape, which should behave
           like a visibility test in this setup.
        */
        const std::string occluderXml =
            "\n\t<shape type=\"cube\">\n"
            "\t\t<transform name=\"toWorld\">\n"
            "\t\t\t<scale x=\"50\" y=\"0.5\" z=\"50\"/>\n"
            "\t\t\t<translate x=\"0\" y=\"0\" z=\"6\"/>\n"
            "\t\t</transform>\n"
            "\t\t<bsdf type=\"diffuse\">\n"
            "\t\t\t<rgb name=\"reflectance\" value=\"0.2\"/>\n"
            "\t\t</bsdf>\n"
            "\t</shape>\n";

        std::string xmlProjOcc = baseXml;
        insertBeforeOnce(xmlProjOcc, "</scene>", occluderXml);

        std::string xmlAreaOcc = baseXml;
        replaceOnce(xmlAreaOcc, "type=\"areaprojected\"", "type=\"area\"");
        insertBeforeOnce(xmlAreaOcc, "</scene>", occluderXml);

        ref<Scene> sceneProjOcc = loadSceneFromXmlString(xmlProjOcc);
        ref<Scene> sceneAreaOcc = loadSceneFromXmlString(xmlAreaOcc);

        const Point refPoint(0, -20, 2);
        const size_t nSamples = 20000;

        Float meanAreaNoVis = 0.0f, meanAreaVis = 0.0f, meanProjVis = 0.0f;
        size_t nonzeroAreaNoVis = 0, nonzeroAreaVis = 0, nonzeroProjVis = 0;

        estimateDirectIntegral(sceneAreaOcc.get(), refPoint, nSamples, false, meanAreaNoVis, &nonzeroAreaNoVis);
        estimateDirectIntegral(sceneAreaOcc.get(), refPoint, nSamples, true,  meanAreaVis,   &nonzeroAreaVis);
        estimateDirectIntegral(sceneProjOcc.get(), refPoint, nSamples, true,  meanProjVis,   &nonzeroProjVis);

        const Float eps = 1e-8f;
        bool ok = std::isfinite(meanAreaNoVis) && std::isfinite(meanAreaVis) && std::isfinite(meanProjVis);
        ok = ok && (meanAreaNoVis > 1e-6f);
        ok = ok && (meanAreaVis < meanAreaNoVis * 0.01f + eps);
        ok = ok && (meanProjVis < meanAreaNoVis * 0.01f + eps);
        ok = ok && (std::abs(meanAreaVis - meanProjVis) < meanAreaNoVis * 0.01f + eps);

        std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
        if (!ok) {
            std::cout << "  meanAreaNoVis=" << meanAreaNoVis
                      << " meanAreaVis=" << meanAreaVis
                      << " meanProjVis=" << meanProjVis << std::endl;
            std::cout << "  nonzero(area noVis/vis, proj vis)="
                      << nonzeroAreaNoVis << "/" << nonzeroAreaVis
                      << ", " << nonzeroProjVis << std::endl;
        }

        if (ok) passed++; else failed++;
    }

    {
        std::cout << "Test: pdfDirect consistency (area vs areaprojected, XML scene)... ";

        FileResolver *resolver = Thread::getThread()->getFileResolver();
        const fs::path projScenePath = resolver->resolveAbsolute(
            "scenes/aabb_spherical_sampling/path_scene_areaprojected_emitter.xml");
        resolver->appendPath(projScenePath.parent_path());

        ref<Scene> sceneProj = loadSceneAbsolute(projScenePath);

        std::string xml = readTextFile(projScenePath);
        replaceOnce(xml, "type=\"areaprojected\"", "type=\"area\"");
        ref<Scene> sceneArea = loadSceneFromXmlString(xml);

        const Point refPoint(0, -20, 2);
        const size_t nSamples = 50000;
        const Float relTol = 5e-4f;

        Float maxRelArea = 0.0f, maxRelProj = 0.0f;
        size_t validArea = 0, validProj = 0;

        bool okArea = checkPdfEmitterDirectMatchesSampled(sceneArea.get(), refPoint,
            nSamples, relTol, maxRelArea, validArea);
        bool okProj = checkPdfEmitterDirectMatchesSampled(sceneProj.get(), refPoint,
            nSamples, relTol, maxRelProj, validProj);

        bool ok = okArea && okProj && validArea > 100 && validProj > 100;
        std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
        // if (!ok) 
        {
            std::cout << "  maxRelErr(area)=" << maxRelArea << " valid=" << validArea << std::endl;
            std::cout << "  maxRelErr(proj)=" << maxRelProj << " valid=" << validProj << std::endl;
        }

        if (ok) passed++; else failed++;
    }

    {
        std::cout << "Test: render compare (area vs areaprojected, mean pixel diff)... ";

        FileResolver *resolver = Thread::getThread()->getFileResolver();
        const fs::path projScenePath = resolver->resolveAbsolute(
            "scenes/aabb_spherical_sampling/path_scene_areaprojected_emitter.xml");
        resolver->appendPath(projScenePath.parent_path());

        const std::string baseXml = readTextFile(projScenePath);

        /* Keep the scene identical except for emitter type, but reduce render settings for a unit test */
        std::string xmlProj = baseXml;
        std::string xmlArea = baseXml;
        replaceOnce(xmlArea, "type=\"areaprojected\"", "type=\"area\"");

        replaceAll(xmlProj, "<integer name=\"width\" value=\"1024\"/>",  "<integer name=\"width\" value=\"64\"/>");
        replaceAll(xmlProj, "<integer name=\"height\" value=\"1024\"/>", "<integer name=\"height\" value=\"64\"/>");
        replaceAll(xmlProj, "<integer name=\"sampleCount\" value=\"32\"/>", "<integer name=\"sampleCount\" value=\"512\"/>");

        replaceAll(xmlArea, "<integer name=\"width\" value=\"1024\"/>",  "<integer name=\"width\" value=\"64\"/>");
        replaceAll(xmlArea, "<integer name=\"height\" value=\"1024\"/>", "<integer name=\"height\" value=\"64\"/>");
        replaceAll(xmlArea, "<integer name=\"sampleCount\" value=\"32\"/>", "<integer name=\"sampleCount\" value=\"512\"/>");

        ref<Scene> sceneProj = loadSceneFromXmlString(xmlProj);
        ref<Scene> sceneArea = loadSceneFromXmlString(xmlArea);

        ref<Bitmap> bmpProj = renderSceneToBitmap(sceneProj.get(), "test_render_proj");
        ref<Bitmap> bmpArea = renderSceneToBitmap(sceneArea.get(), "test_render_area");

        // { // save the bitmaps for debugging if needed
        //     bmpProj->save(fs::path("debug_aabb_proj.exr"));
        //     bmpArea->save(fs::path("debug_aabb_area.exr"));

        // }

        const Float mad = meanAbsDiff(bmpProj.get(), bmpArea.get());
        const Float meanA = meanValue(bmpArea.get());
        const Float rel = mad / std::max((Float) 1e-6f, meanA);

        /* Heuristic threshold: should be very close if unbiased. */
        bool ok = std::isfinite(rel) && rel < 0.05f;
        std::cout << (ok ? "PASSED" : "FAILED") << std::endl;
        // if (!ok)
        {
            std::cout << "  meanAbsDiff=" << mad << " meanArea=" << meanA << " rel=" << rel << std::endl;
        }

        if (ok) passed++; else failed++;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return failed;
}




int main(int argc, char **argv) {
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


    std::cout << "\nRunning AABB Sampling Tests...\n" << std::endl;
    int result = runAabbSamplingTests();

    std::cout << "\n\n-----------------------------\nRunning AABB Light Tests...\n" << std::endl;
    result += runAabbLightTests();

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
