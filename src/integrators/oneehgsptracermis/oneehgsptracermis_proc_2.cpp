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
#include <mitsuba/bidir/util.h>
#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/render/scene.h>
#include "oneehgsptracermis_proc.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Worker implementation                        */
/* ==================================================================== */

class ONEEEHGSPTracerMISRenderer : public WorkProcessor {
public:
    ONEEEHGSPTracerMISRenderer(const ONEEEHGSPTracerMISConfiguration &config) : m_config(config) { }

    ONEEEHGSPTracerMISRenderer(Stream *stream, InstanceManager *manager)
        : WorkProcessor(stream, manager), m_config(stream) { }

    virtual ~ONEEEHGSPTracerMISRenderer() { }

    void serialize(Stream *stream, InstanceManager *manager) const {
        m_config.serialize(stream);
    }

    ref<WorkUnit> createWorkUnit() const {
        return new RectangularWorkUnit();
    }

    ref<WorkResult> createWorkResult() const {
        return new ONEEEHGSPTracerMISWorkResult(m_config, m_rfilter.get(),
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

        // Extract pinhole camera world-space origin for glint BVH sampling
        const AnimatedTransform *trafo = m_sensor->getWorldTransform();
        m_cameraOrigin = trafo->eval(0.f)(Point(0.f, 0.f, 0.f));
    }

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
        const RectangularWorkUnit *rect = static_cast<const RectangularWorkUnit *>(workUnit);
        ONEEEHGSPTracerMISWorkResult *result = static_cast<ONEEEHGSPTracerMISWorkResult *>(workResult);

        result->setOffset(rect->getOffset());
        result->setSize(rect->getSize());
        result->clear();
        m_hilbertCurve.initialize(TVector2<uint8_t>(rect->getSize()));

        SamplingBVH *bvh = m_scene->getSamplingBVH();
        if (!bvh || !bvh->isBuilt()) {
            Log(EWarn, "ONEEEHGSPTracerMIS: SamplingBVH not available or not built!");
            return;
        }
        m_bvh = bvh;

        for (size_t i = 0; i < m_hilbertCurve.getPointCount(); ++i) {
            Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
            m_sampler->generate(offset);

            for (size_t j = 0; j < m_sampler->getSampleCount(); ++j) {
                if (stop) break;
                // traceSample(result, bvh);
                pathTraceSample(result, offset);
                m_sampler->advance();
            }
        }
    }

    ref<WorkProcessor> clone() const {
        return new ONEEEHGSPTracerMISRenderer(m_config);
    }

    MTS_DECLARE_CLASS()
private:
    // -----------------------------------------------------------------------
    // MIS pdf helpers
    // -----------------------------------------------------------------------

    /// Solid-angle pdf of sampling x_n via the BVH camera technique.
    /// pRec_xe carries the emitter vertex x_e (its position is used by
    /// computeNodeImportanceGlint during BVH traversal).
    Float pdfGlintXn(const Intersection &its_xn,
                     const PositionSamplingRecord &pRec_xe) const {
        if (!m_bvh) return 0.f;
        return m_bvh->pdfGeometryCamera(
            m_scene.get(), m_cameraOrigin, m_sensor.get(), pRec_xe, its_xn);
    }

    /// Solid-angle pdf of the camera ray passing through its_xn (PT technique).
    /// Evaluates the sensor's direct-sampling pdf from its_xn toward the camera;
    /// for a pinhole camera this is deterministic so no random sample is consumed.
    Float pdfCameraRay(const Intersection &its_xn) const {
        DirectSamplingRecord dRec(its_xn);
        Spectrum W_e = m_scene->sampleSensorDirect(dRec, Point2(0.5f), false);
        return W_e.isZero() ? 0.f : dRec.pdf;
    }

    void traceSample(ONEEEHGSPTracerMISWorkResult *result, SamplingBVH *bvh) {
        std::cout << "Tracing sample " << std::endl;
        // ----------------------------------------------------------------
        // 1. Sample emitter position x_e
        // ----------------------------------------------------------------
        PositionSamplingRecord pRec_xe(0.f);
        m_scene->sampleEmitterPosition(pRec_xe, m_sampler->next2D());
        if (pRec_xe.pdf <= 0.f || !pRec_xe.object)
            return;

        // ----------------------------------------------------------------
        // 2. BVH camera sampling: find intermediate vertex x_n
        // ----------------------------------------------------------------
        Intersection its_xn;
        Float pdf_xn = 0.f;
        if (!bvh->sampleGeometryCamera(m_scene, m_sampler, m_cameraOrigin,
                                        m_sensor.get(), pRec_xe, its_xn, pdf_xn))
            return;

        if (pdf_xn <= 0.f || !its_xn.isValid())
            return;

        // ----------------------------------------------------------------
        // 3. Sensor connection: determine which pixel x_n maps to and W_e.
        //    Also checks visibility x_n -> camera.
        // ----------------------------------------------------------------
        DirectSamplingRecord dRec(its_xn);
        Spectrum W_e = m_scene->sampleSensorDirect(dRec, m_sampler->next2D(), true);
        if (W_e.isZero())
            return;

        // Set local incoming direction at x_n so BSDF eval is well-defined
        its_xn.wi = its_xn.toLocal(dRec.d);

        // ----------------------------------------------------------------
        // 4. Evaluate x_n -> x_e contribution: f_xn * G_ne * Le
        // ----------------------------------------------------------------
        Spectrum contrib = evalGlintContribution(
            m_scene, its_xn, pRec_xe, dRec.d, /* checkVisibility */ true);
        if (contrib.isZero())
            return;

        // ----------------------------------------------------------------
        // 5. MIS weight and full estimator.
        //
        // Two techniques compete for the path (camera_o, x_n, x_e):
        //   p_glint = pdf_xn [solid angle from o]  * p_e(x_e) [area]
        //   p_pt    = p_cam(x_n) [solid angle from o] * p_nee(x_e) [area]
        //
        // Both are expressed in the (solid_angle × area) product measure.
        // p_cam(x_n) = dRec.pdf from sampleSensorDirect (same ray direction
        // as pdf_xn, same solid-angle measure).
        // p_e(x_e) and p_nee(x_e) are both area pdfs from the emitter's
        // position distribution (equal for standard emitters in Mitsuba).
        //
        // w_glint = p_glint / (p_glint + p_pt)
        // estimator = W_e * contrib * w_glint / p_glint
        //           = W_e * contrib / (p_glint + p_pt)
        // ----------------------------------------------------------------
        Float p_cam_xn   = dRec.pdf;  // solid-angle pdf from sampleSensorDirect
        Float p_nee_area = m_scene->pdfEmitterPosition(pRec_xe);  // area pdf for PT-NEE
        Float p_glint    = pdf_xn * pRec_xe.pdf;
        Float p_pt       = p_cam_xn * p_nee_area;
        Float mis_denom  = p_glint + p_pt;
        // if (mis_denom <= 0.f)
        //     return;

        Spectrum value = W_e * contrib / p_glint; // / mis_denom;
        if (!value.isValid())
            return;

        result->putLightSample(dRec.uv, value);
    }

    // -----------------------------------------------------------------------
    // Path tracing sample — equivalent to MIPathTracer::Li() but invoked
    // directly inside the worker.  The camera ray is generated here (normally
    // done by SamplingIntegrator's render loop before calling Li()).
    // At depth=0, the NEE contribution is MIS-weighted against the BVH glint
    // technique via w_pt_nee = p_pt / (p_pt + p_glint).
    // -----------------------------------------------------------------------
    void pathTraceSample(ONEEEHGSPTracerMISWorkResult *result, const Point2i &pixelOffset) {
        // Generate camera ray for this pixel
        RayDifferential ray;
        Point2 samplePos   = Point2(pixelOffset) + m_sampler->next2D();
        Point2 apertureSample = m_sampler->next2D();
        Float  timeSample     = m_sampler->next1D();

        Spectrum sensorWeight = m_sensor->sampleRayDifferential(
            ray, samplePos, apertureSample, timeSample);
        if (sensorWeight.isZero())
            return;

        // Run path tracing Li loop (copied from MIPathTracer::Li)
        Spectrum Li = pathLi(ray);
        if (Li.isZero())
            return;

        result->putSample(samplePos, sensorWeight * Li);
    }

    Spectrum pathLi(const RayDifferential &r) {
        RadianceQueryRecord rRec(m_scene, m_sampler);
        rRec.newQuery(RadianceQueryRecord::ESensorRay, m_sensor->getMedium());

        const Scene *scene = rRec.scene;
        Intersection &its  = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        while (rRec.depth <= m_config.maxDepth || m_config.maxDepth < 0) {
            if (!its.isValid()) {
                if (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance))
                Li += throughput * its.Le(-ray.d);

            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if (rRec.depth >= m_config.maxDepth && m_config.maxDepth > 0)
                break;

            /* Direct illumination sampling */
            DirectSamplingRecord dRec(its);
            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);
                    if (!bsdfVal.isZero()) {
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;
                        Li += throughput * value * bsdfVal * miWeight(dRec.pdf, bsdfPdf);

                        /*
                        // Intra-PT MIS: NEE vs BSDF sampling.
                        Float w_nee_bsdf = miWeight(dRec.pdf, bsdfPdf);

                        // Cross-integrator MIS at depth=0 (first surface hit):
                        // PT-NEE competes with the BVH glint technique.
                        // w_pt_nee = p_pt / (p_pt + p_glint)
                        // where p_pt  = p_cam(x_n) * p_nee_area(x_e)   [solid_angle × area]
                        //       p_glint = pdf_xn_bvh * p_e_area(x_e)   [solid_angle × area]
                        if (rRec.depth == 0 && m_bvh) {
                            PositionSamplingRecord pRec_xe_mis(its.time);
                            pRec_xe_mis.p      = dRec.p;
                            pRec_xe_mis.n      = dRec.n;
                            pRec_xe_mis.object = dRec.object;

                            Float p_glint_xn   = pdfGlintXn(its, pRec_xe_mis);
                            Float p_e_area     = scene->pdfEmitterPosition(pRec_xe_mis);
                            Float p_cam_xn     = pdfCameraRay(its);
                            // p_nee_area == p_e_area for standard emitters
                            // (both sampleEmitterPosition and sampleEmitterDirect
                            // use the same underlying area distribution)
                            Float p_nee_area   = p_e_area;

                            Float p_glint_path = p_glint_xn * p_e_area;
                            Float p_pt_path    = p_cam_xn   * p_nee_area;
                            Float mis_denom    = p_glint_path + p_pt_path;
                            Float w_pt_nee     = (mis_denom > 0.f)
                                                 ? p_pt_path / mis_denom : 1.f;

                            Li += throughput * value * bsdfVal * w_nee_bsdf; // * w_pt_nee;
                        } else {
                            Li += throughput * value * bsdfVal * w_nee_bsdf;
                        }
                            */
                    }
                }
            }

            /* BSDF sampling */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;

            scattered |= bRec.sampledType != BSDF::ENull;
            const Vector wo = its.toWorld(bRec.wo);

            bool hitEmitter = false;
            Spectrum value;

            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its)) {
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                const Emitter *env = scene->getEnvironmentEmitter();
                if (env) {
                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;
                    hitEmitter = true;
                } else {
                    break;
                }
            }

            throughput *= bsdfWeight;
            eta *= bRec.eta;

            if (hitEmitter && (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta))
                    ? scene->pdfEmitterDirect(dRec) : 0;
                Li += throughput * value * miWeight(bsdfPdf, lumPdf);
            }

            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_config.rrDepth) {
                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
        }

        return Li;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    ref<Sampler> m_sampler;
    ref<ReconstructionFilter> m_rfilter;
    MemoryPool m_pool;
    ONEEEHGSPTracerMISConfiguration m_config;
    HilbertCurve2D<uint8_t> m_hilbertCurve;
    Point m_cameraOrigin;
    SamplingBVH *m_bvh;
};


/* ==================================================================== */
/*                           Parallel process                           */
/* ==================================================================== */

ONEEEHGSPTracerMISProcess::ONEEEHGSPTracerMISProcess(const RenderJob *parent, RenderQueue *queue,
        const ONEEEHGSPTracerMISConfiguration &config) :
    BlockedRenderProcess(parent, queue, config.blockSize), m_config(config) {
    m_refreshTimer = new Timer();
}

ref<WorkProcessor> ONEEEHGSPTracerMISProcess::createWorkProcessor() const {
    return new ONEEEHGSPTracerMISRenderer(m_config);
}

void ONEEEHGSPTracerMISProcess::develop() {
    if (!m_config.lightImage)
        return;
    LockGuard lock(m_resultMutex);
    const ImageBlock *lightImage = m_result->getLightImage();
    m_film->setBitmap(m_result->getImageBlock()->getBitmap());
    m_film->addBitmap(lightImage->getBitmap(), 1.0f / m_config.sampleCount);
    m_refreshTimer->reset();
    m_queue->signalRefresh(m_parent);
}

void ONEEEHGSPTracerMISProcess::processResult(const WorkResult *wr, bool cancelled) {
    if (cancelled)
        return;
    const ONEEEHGSPTracerMISWorkResult *result = static_cast<const ONEEEHGSPTracerMISWorkResult *>(wr);
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

void ONEEEHGSPTracerMISProcess::bindResource(const std::string &name, int id) {
    BlockedRenderProcess::bindResource(name, id);
    if (name == "sensor" && m_config.lightImage) {
        /* If needed, allocate memory for the light image */
        m_result = new ONEEEHGSPTracerMISWorkResult(m_config, NULL, m_film->getCropSize());
        m_result->clear();
    }
}

MTS_IMPLEMENT_CLASS_S(ONEEEHGSPTracerMISRenderer, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(ONEEEHGSPTracerMISProcess, false, BlockedRenderProcess)
MTS_NAMESPACE_END
