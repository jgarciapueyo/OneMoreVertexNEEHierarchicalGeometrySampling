/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/core/statistics.h>
#include <mitsuba/core/sfcurve.h>
#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/render/scene.h>
#include "oneehgsptracermis_proc.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                   MIS for the first camera vertex                    */
/* ==================================================================== */

/**
 * \brief Power-heuristic weights between the three techniques that generate
 * the paths x_0 -> x_n -> x_e (direct illumination of the first visible
 * vertex x_n):
 *   - camera ray to x_n, then BSDF sampling at x_n hitting the emitter,
 *   - camera ray to x_n, then emitter sampling (NEE) at x_n,
 *   - the camera technique: x_e by emitter position sampling, x_n sampled
 *     from the SamplingBVH as seen from the camera origin.
 *
 * Densities are per unit solid angle at the camera (for x_n) and per unit
 * area at x_e (for x_e); the conversion factors common to all techniques
 * cancel in the weights. All three techniques evaluate the same functions
 * of (x_n, x_e), so the weights of a path always sum to one.
 */
class CameraTechniqueMIS : public FirstVertexDirectMIS {
public:
    CameraTechniqueMIS() : m_scene(NULL), m_sensor(NULL), m_bvh(NULL),
        m_useNEE(true), m_useCameraTechnique(true) { }

    void configure(const Scene *scene, const Sensor *sensor, const SamplingBVH *bvh,
            const Point &cameraOrigin, const OneMoreVertexPTracerMISConfiguration &config) {
        m_scene = scene;
        m_sensor = sensor;
        m_bvh = bvh;
        m_cameraOrigin = cameraOrigin;
        m_useNEE = !config.path.disableNee && !config.path.disableFirstBounceNEE;
        m_useCameraTechnique = config.lightImage;
    }

    struct Densities {
        Float bsdfHit, nee, camera;
    };

    /// Densities of the three techniques for the path x_0 -> its_xn -> (p_e, n_e)
    Densities densities(const Intersection &its_xn, const Point &p_e, const Normal &n_e,
            const Emitter *emitter) const {
        Densities d = { 0.0f, 0.0f, 0.0f };

        /* Canonical record of x_e, so that every technique evaluates the
           emitter terms with the same measure */
        PositionSamplingRecord pRec_xe(its_xn.time);
        pRec_xe.p = p_e;
        pRec_xe.n = n_e;
        pRec_xe.object = emitter;
        pRec_xe.measure = emitter->needsPositionSample() ? EArea : EDiscrete;

        Float pdfCamera = pdfCameraRay(its_xn.p);
        Float pdfLight = m_scene->pdfEmitterPosition(pRec_xe);

        if (m_useNEE)
            d.nee = pdfCamera * pdfLight;
        if (m_useCameraTechnique)
            d.camera = m_bvh->pdfGeometryCamera(m_scene, m_cameraOrigin, m_sensor, pRec_xe, its_xn) * pdfLight;

        /* Emitters with a position density can also be hit by BSDF sampling */
        if (emitter->needsPositionSample()) {
            Vector d_ne = p_e - its_xn.p;
            Float dist2 = d_ne.lengthSquared();
            if (dist2 > 0) {
                d_ne /= std::sqrt(dist2);
                const BSDF *bsdf = its_xn.getBSDF();
                BSDFSamplingRecord bRec(its_xn, its_xn.toLocal(d_ne), ERadiance);
                d.bsdfHit = pdfCamera * bsdf->pdf(bRec) * absDot(n_e, d_ne) / dist2;
            }
        }
        return d;
    }

    static Float power(Float num, const Densities &d) {
        Float denom = d.bsdfHit * d.bsdfHit + d.nee * d.nee + d.camera * d.camera;
        return denom > 0 ? (num * num) / denom : 0.0f;
    }

    Float neeWeight(const Intersection &its, const DirectSamplingRecord &dRec) const {
        Densities d = densities(its, dRec.p, dRec.n, static_cast<const Emitter *>(dRec.object));
        return power(d.nee, d);
    }

    Float hitWeight(const Intersection &its, const DirectSamplingRecord &dRec) const {
        Densities d = densities(its, dRec.p, dRec.n, static_cast<const Emitter *>(dRec.object));
        return power(d.bsdfHit, d);
    }

    Float cameraTechniqueWeight(const Intersection &its_xn, const PositionSamplingRecord &pRec_xe) const {
        Densities d = densities(its_xn, pRec_xe.p, pRec_xe.n, static_cast<const Emitter *>(pRec_xe.object));
        return power(d.camera, d);
    }

private:
    /// Density of a camera ray towards x_n, per unit solid angle at the camera origin
    Float pdfCameraRay(const Point &x_n) const {
        Vector d = x_n - m_cameraOrigin;
        Float dist2 = d.lengthSquared();
        if (dist2 == 0.f)
            return 0.f;
        PositionSamplingRecord pRec_cam(0.f);
        pRec_cam.p = m_cameraOrigin;
        pRec_cam.object = m_sensor;
        DirectionSamplingRecord dRec_cam(d / std::sqrt(dist2));
        return m_sensor->pdfDirection(dRec_cam, pRec_cam);
    }

    const Scene *m_scene;
    const Sensor *m_sensor;
    const SamplingBVH *m_bvh;
    Point m_cameraOrigin;
    bool m_useNEE, m_useCameraTechnique;
};

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

class OneMoreVertexPTracerMISRenderer : public WorkProcessor {
public:
    OneMoreVertexPTracerMISRenderer(const OneMoreVertexPTracerMISConfiguration &config) : m_config(config) { }

    OneMoreVertexPTracerMISRenderer(Stream *stream, InstanceManager *manager)
        : WorkProcessor(stream, manager), m_config(stream) { }

    virtual ~OneMoreVertexPTracerMISRenderer() { }

    void serialize(Stream *stream, InstanceManager *manager) const {
        m_config.serialize(stream);
    }

    ref<WorkUnit> createWorkUnit() const {
        return new RectangularWorkUnit();
    }

    ref<WorkResult> createWorkResult() const {
        return new OneMoreVertexPTracerMISWorkResult(m_config, m_rfilter.get(),
            Vector2i(m_config.blockSize));
    }

    void prepare() {
        Scene *scene = static_cast<Scene *>(getResource("scene"));
        m_scene = new Scene(scene);
        m_sampler = static_cast<Sampler *>(getResource("sampler"));
        m_sensor = static_cast<Sensor *>(getResource("sensor"));
        m_rfilter = m_sensor->getFilm()->getReconstructionFilter();
        m_scene->removeSensor(scene->getSensor());
        m_scene->addSensor(m_sensor);
        m_scene->setSensor(m_sensor);
        m_scene->setSampler(m_sampler);
        m_scene->wakeup(NULL, m_resources);
        m_scene->initializeBidirectional();

        /* World-space camera origin (lens center) used by the camera technique */
        const AnimatedTransform *trafo = m_sensor->getWorldTransform();
        m_cameraOrigin = trafo->eval(0.f)(Point(0.f, 0.f, 0.f));

        m_mis.configure(m_scene.get(), m_sensor.get(), m_scene->getSamplingBVH(),
            m_cameraOrigin, m_config);
    }

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
        const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
        OneMoreVertexPTracerMISWorkResult *result = static_cast<OneMoreVertexPTracerMISWorkResult *>(workResult);

        result->setOffset(rect->getOffset());
        result->setSize(rect->getSize());
        result->clear();
        m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));

        const SamplingBVH *bvh = m_scene->getSamplingBVH();
        if (!bvh || !bvh->isBuilt()) {
            Log(EWarn, "OneMoreVertexPTracerMIS: SamplingBVH not available or not built!");
            return;
        }

        for (size_t i = 0; i < m_hilbertCurve.getPointCount(); ++i) {
            Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
            m_sampler->generate(offset);

            for (size_t j = 0; j < m_sampler->getSampleCount(); ++j) {
                if (stop) break;
                if (m_config.lightImage)
                    cameraTechniqueSample(result, bvh);
                cameraPathSample(result, offset);
                m_sampler->advance();
            }
        }
    }

    ref<WorkProcessor> clone() const {
        return new OneMoreVertexPTracerMISRenderer(m_config);
    }

    MTS_DECLARE_CLASS()
private:
    /// Camera technique: x_e by emitter position sampling, x_n from the
    /// SamplingBVH as seen from the camera, then connected to the sensor
    void cameraTechniqueSample(OneMoreVertexPTracerMISWorkResult *result, const SamplingBVH *bvh) {
        PositionSamplingRecord pRec_xe(0.f);
        m_scene->sampleEmitterPosition(pRec_xe, m_sampler->next2D());
        if (pRec_xe.pdf <= 0.f || !pRec_xe.object)
            return;

        /* x_n and its density per unit area */
        Intersection its_xn;
        Float pdf_xn = 0.f;
        if (!bvh->sampleGeometryCamera(m_scene, m_sampler, m_cameraOrigin,
                                        m_sensor.get(), pRec_xe, its_xn, pdf_xn))
            return;
        if (pdf_xn <= 0.f || !its_xn.isValid())
            return;

        /* Connect x_n to the sensor (pixel position, W_e and visibility) */
        DirectSamplingRecord dRec(its_xn);
        Spectrum W_e = m_scene->sampleSensorDirect(dRec, m_sampler->next2D(), true);
        if (W_e.isZero())
            return;

        /* Local direction towards the camera, so that the BSDF at x_n is well-defined */
        its_xn.wi = its_xn.toLocal(dRec.d);

        /* f(x_n) * G(x_n, x_o) cosine * G(x_n, x_e) * Le, with x_n -> x_e visibility */
        Spectrum contrib = evalGlintContribution(
            m_scene, its_xn, pRec_xe, dRec.d, /* checkVisibility */ true);
        if (contrib.isZero())
            return;

        Spectrum value = W_e * contrib / (pdf_xn * pRec_xe.pdf);
        if (!value.isValid())
            return;

        result->putLightSample(dRec.uv, value * m_mis.cameraTechniqueWeight(its_xn, pRec_xe));
    }

    /// Camera path traced with the one-more-vertex path tracer
    void cameraPathSample(OneMoreVertexPTracerMISWorkResult *result, const Point2i &pixelOffset) {
        Float diffScaleFactor = 1.0f /
            std::sqrt((Float) m_sampler->getSampleCount());

        bool needsApertureSample = m_sensor->needsApertureSample();
        bool needsTimeSample = m_sensor->needsTimeSample();

        RadianceQueryRecord rRec(m_scene.get(), m_sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!m_sensor->getFilm()->hasAlpha()) /* Don't compute an alpha channel if we don't have to */
            queryType &= ~RadianceQueryRecord::EOpacity;

        rRec.newQuery(queryType, m_sensor->getMedium());
        Point2 samplePos(Point2(pixelOffset) + Vector2(rRec.nextSample2D()));

        if (needsApertureSample)
            apertureSample = rRec.nextSample2D();
        if (needsTimeSample)
            timeSample = rRec.nextSample1D();

        Spectrum spec = m_sensor->sampleRayDifferential(
            sensorRay, samplePos, apertureSample, timeSample);

        sensorRay.scaleDifferential(diffScaleFactor);
        spec *= onemorevertex::Li(m_config.path, sensorRay, rRec,
            m_config.lightImage ? &m_mis : NULL);
        result->putSample(samplePos, spec);
    }

    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    ref<Sampler> m_sampler;
    ref<ReconstructionFilter> m_rfilter;
    OneMoreVertexPTracerMISConfiguration m_config;
    HilbertCurve2D<uint8_t> m_hilbertCurve;
    Point m_cameraOrigin;
    CameraTechniqueMIS m_mis;
};


/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

OneMoreVertexPTracerMISProcess::OneMoreVertexPTracerMISProcess(const RenderJob *parent, RenderQueue *queue,
        const OneMoreVertexPTracerMISConfiguration &config) :
    BlockedRenderProcess(parent, queue, config.blockSize), m_config(config) {
    m_refreshTimer = new Timer();
}

ref<WorkProcessor> OneMoreVertexPTracerMISProcess::createWorkProcessor() const {
    return new OneMoreVertexPTracerMISRenderer(m_config);
}

void OneMoreVertexPTracerMISProcess::develop() {
    if (!m_config.lightImage)
        return;
    LockGuard lock(m_resultMutex);
    const ImageBlock *lightImage = m_result->getLightImage();
    m_film->setBitmap(m_result->getImageBlock()->getBitmap());
    m_film->addBitmap(lightImage->getBitmap(), 1.0f / m_config.sampleCount);
    m_refreshTimer->reset();
    m_queue->signalRefresh(m_parent);
}

void OneMoreVertexPTracerMISProcess::processResult(const WorkResult *wr, bool cancelled) {
    if (cancelled)
        return;
    const OneMoreVertexPTracerMISWorkResult *result = static_cast<const OneMoreVertexPTracerMISWorkResult *>(wr);
    ImageBlock *block = const_cast<ImageBlock *>(result->getImageBlock());
    LockGuard lock(m_resultMutex);
    m_progress->update(++m_resultCount);
    if (m_config.lightImage) {
        const ImageBlock *lightImage = m_result->getLightImage();
        m_result->put(result);
        if (m_parent->isInteractive()) {
            /* Modify the finished image block so that it includes the light image contributions,
               which creates a more intuitive preview of the rendering process. This is
               not 100% correct but doesn't matter, as the shown image will be properly re-developed
               every 2 seconds and once more when the rendering process finishes */

            Float invSampleCount = 1.0f / m_config.sampleCount;
            const Bitmap *sourceBitmap = lightImage->getBitmap();
            Bitmap *destBitmap = block->getBitmap();
            int borderSize = block->getBorderSize();
            Point2i offset = block->getOffset();
            Vector2i size = block->getSize();

            for (int y=0; y<size.y; ++y) {
                const Float *source = sourceBitmap->getFloatData()
                    + (offset.x + (y+offset.y) * sourceBitmap->getWidth()) * SPECTRUM_SAMPLES;
                Float *dest = destBitmap->getFloatData()
                    + (borderSize + (y + borderSize) * destBitmap->getWidth()) * (SPECTRUM_SAMPLES + 2);

                for (int x=0; x<size.x; ++x) {
                    Float weight = dest[SPECTRUM_SAMPLES + 1] * invSampleCount;
                    for (int k=0; k<SPECTRUM_SAMPLES; ++k)
                        *dest++ += *source++ * weight;
                    dest += 2;
                }
            }
        }
    }

    m_film->put(block);

    /* Re-develop the entire image every two seconds if partial results are
       visible (e.g. in a graphical user interface). This only applies when
       there is a light image. */
    bool developFilm = m_config.lightImage &&
        (m_parent->isInteractive() && m_refreshTimer->getMilliseconds() > 2000);

    m_queue->signalWorkEnd(m_parent, result->getImageBlock(), false);

    if (developFilm)
        develop();
}

void OneMoreVertexPTracerMISProcess::bindResource(const std::string &name, int id) {
    BlockedRenderProcess::bindResource(name, id);
    if (name == "sensor" && m_config.lightImage) {
        /* If needed, allocate memory for the light image */
        m_result = new OneMoreVertexPTracerMISWorkResult(m_config, NULL, m_film->getCropSize());
        m_result->clear();
    }
}


MTS_IMPLEMENT_CLASS_S(OneMoreVertexPTracerMISRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(OneMoreVertexPTracerMISProcess, false, BlockedRenderProcess)
MTS_NAMESPACE_END
