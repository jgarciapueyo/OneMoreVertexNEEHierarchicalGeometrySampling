/*
    Tests of the hierarchical geometry sampling (SamplingBVH).
    Run inside the container with:  mtsutil test_hgs
*/

#include <mitsuba/render/testcase.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/hgs/bvh.h>

MTS_NAMESPACE_BEGIN

class TestHGS : public TestCase {
public:
    MTS_BEGIN_TESTCASE()
    MTS_DECLARE_TEST(test01_samplePdfConsistency)
    MTS_DECLARE_TEST(test02_densityNormalization)
    MTS_DECLARE_TEST(test03_uniformDensityNormalization)
    MTS_END_TESTCASE()

    /// Floor, nine cube meshes of different sizes and orientations, spot light
    ref<Scene> createScene(const std::string &samplingMode, bool specular, bool uniform) {
        std::ostringstream oss;
        oss << "<scene version=\"0.6.0\">"
            << "<integrator type=\"oneehgspath_2step\"/>"
            << "<geometrybvh type=\"samplingbvh\">"
            << "  <integer name=\"maxDepth\" value=\"6\"/>"
            << "  <string name=\"samplingMode\" value=\"" << samplingMode << "\"/>"
            << "  <boolean name=\"specularImportance\" value=\"" << (specular ? "true" : "false") << "\"/>"
            << "  <boolean name=\"uniformImportance\" value=\"" << (uniform ? "true" : "false") << "\"/>"
            << "  <float name=\"defensivePDF\" value=\"1e-6\"/>"
            << "</geometrybvh>"
            << "<sensor type=\"perspective\">"
            << "  <transform name=\"toWorld\"><lookAt origin=\"0, -8, 3\" target=\"0, 0, 1\" up=\"0, 0, 1\"/></transform>"
            << "  <sampler type=\"independent\"><integer name=\"sampleCount\" value=\"1\"/></sampler>"
            << "  <film type=\"hdrfilm\"><integer name=\"width\" value=\"8\"/><integer name=\"height\" value=\"8\"/></film>"
            << "</sensor>"
            << "<emitter type=\"spot\">"
            << "  <transform name=\"toWorld\"><lookAt origin=\"-4, 0, 4\" target=\"0, 0, 1\" up=\"0, 0, 1\"/></transform>"
            << "  <rgb name=\"intensity\" value=\"100\"/><float name=\"cutoffAngle\" value=\"40\"/>"
            << "</emitter>"
            << "<shape type=\"rectangle\">"
            << "  <transform name=\"toWorld\"><scale value=\"10\"/></transform>"
            << "  <bsdf type=\"diffuse\"/>"
            << "</shape>";
        for (int i = 0; i < 9; ++i) {
            Float x = (Float) (i % 3) - 1, y = (Float) (i / 3) - 1;
            oss << "<shape type=\"cube\"><transform name=\"toWorld\">"
                << "  <scale x=\"" << 0.1f + 0.05f * i << "\" y=\"0.2\" z=\"" << 0.3f - 0.02f * i << "\"/>"
                << "  <rotate z=\"1\" angle=\"" << 20 * i << "\"/>"
                << "  <translate x=\"" << 1.5f * x << "\" y=\"" << 1.5f * y << "\" z=\"" << 1.0f + 0.2f * i << "\"/>"
                << "</transform>"
                << "<bsdf type=\"roughplastic\"><float name=\"alpha\" value=\"0.2\"/></bsdf>"
                << "</shape>";
        }
        oss << "</scene>";
        ref<Scene> scene = loadSceneFromString(oss.str());
        scene->initialize();
        return scene;
    }

    /// Upward-facing vertex x_s on the floor, using the floor's BSDF
    Intersection floorVertex(const Scene *scene, const Point &p) {
        Intersection its;
        its.p = p;
        its.geoFrame = its.shFrame = Frame(Normal(0, 0, 1));
        its.wi = Vector(0, 0, 1);
        its.t = 1.0f;
        for (size_t i = 0; i < scene->getShapes().size(); ++i)
            if (!scene->getShapes()[i]->getClass()->derivesFrom(MTS_CLASS(TriMesh)))
                its.shape = scene->getShapes()[i].get();
        return its;
    }

    void test01_samplePdfConsistency() {
        const char *modes[] = { "Primitive", "SphericalAABB", "SphericalAABB_anyNode" };
        for (int m = 0; m < 3; ++m) {
            for (int spec = 0; spec < 2; ++spec) {
                ref<Scene> scene = createScene(modes[m], spec == 1, false);
                const SamplingBVH *bvh = scene->getSamplingBVH();
                assertTrue(bvh->isBuilt());
                Sampler *sampler = scene->getSampler();

                Intersection its_xs = floorVertex(scene, Point(0.3f, -0.4f, 0.0f));
                int checked = 0;
                for (int i = 0; i < 2000; ++i) {
                    PositionSamplingRecord pRec_xe;
                    scene->sampleEmitterPosition(pRec_xe, sampler->next2D());

                    Intersection its_xp;
                    Float pdf = 0;
                    if (!bvh->sampleGeometry(scene, sampler, its_xs, pRec_xe, its_xp, pdf))
                        continue;
                    Float pdf2 = bvh->pdfGeometry(scene, its_xs, pRec_xe, its_xp);
                    assertEqualsEpsilon(pdf2 / pdf, (Float) 1, (Float) 1e-3);
                    ++checked;
                }
                Log(EInfo, "%s, specular=%i: %i samples, sampling and PDF densities agree",
                    modes[m], spec, checked);
                assertTrue(checked > 100);
            }
        }
    }

    /// With Primitive leaf sampling the density is constant over each
    /// triangle, so the probabilities of all triangles (density per unit
    /// area times area) must sum to one: checks that the traversal and leaf
    /// probabilities are normalized
    void checkNormalization(bool uniform) {
        ref<Scene> scene = createScene("Primitive", false, uniform);
        const SamplingBVH *bvh = scene->getSamplingBVH();

        Intersection its_xs = floorVertex(scene, Point(0.2f, 0.1f, 0.0f));
        PositionSamplingRecord pRec_xe;
        scene->sampleEmitterPosition(pRec_xe, Point2(0.5f));

        double total = 0;
        const std::vector<TriMesh *> &meshes = scene->getMeshes();
        for (size_t m = 0; m < meshes.size(); ++m) {
            const Point *v = meshes[m]->getVertexPositions();
            for (size_t t = 0; t < meshes[m]->getTriangleCount(); ++t) {
                const Triangle &tri = meshes[m]->getTriangles()[t];
                Intersection its;
                its.p = (v[tri.idx[0]] + v[tri.idx[1]] + v[tri.idx[2]]) / 3.0f;
                its.shape = meshes[m];
                its.primIndex = (uint32_t) t;
                Normal n(normalize(cross(v[tri.idx[1]] - v[tri.idx[0]], v[tri.idx[2]] - v[tri.idx[0]])));
                its.geoFrame = its.shFrame = Frame(n);

                /* Solid angle at x_s -> area, times the area of the triangle */
                Float pdf_sa = bvh->pdfGeometry(scene, its_xs, pRec_xe, its);
                Vector d = its.p - its_xs.p;
                Float dist2 = d.lengthSquared();
                Float cosTheta = absDot(n, d) / std::sqrt(dist2);
                total += (double) pdf_sa * cosTheta / dist2 * tri.surfaceArea(v);
            }
        }
        Log(EInfo, "uniform=%i: sum of the triangle probabilities = %f", (int) uniform, total);
        assertEqualsEpsilon((Float) total, (Float) 1, (Float) 1e-3);
    }

    void test02_densityNormalization() { checkNormalization(false); }
    void test03_uniformDensityNormalization() { checkNormalization(true); }
};

MTS_EXPORT_TESTCASE(TestHGS, "Hierarchical geometry sampling (SamplingBVH)")
MTS_NAMESPACE_END
