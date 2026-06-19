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

        for (size_t i = 0; i < m_hilbertCurve.getPointCount(); ++i) {
            Point2i offset = Point2i(m_hilbertCurve[i]) + Vector2i(rect->getOffset());
            m_sampler->generate(offset);

            for (size_t j = 0; j < m_sampler->getSampleCount(); ++j) {
                if (stop) break;
                traceSample(result, bvh);
                pathTraceSample(result, offset, bvh);
                m_sampler->advance();
            }
        }
    }

    ref<WorkProcessor> clone() const {
        return new ONEEEHGSPTracerMISRenderer(m_config);
    }

    MTS_DECLARE_CLASS()
private:
    void traceSample(ONEEEHGSPTracerMISWorkResult *result, SamplingBVH *bvh) {
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
        // 5. 3-way MIS weight for the glint technique.
        //    All q_i factor out the common cosθ_n/dist_on² term so it
        //    cancels in the power-heuristic ratio.
        // ----------------------------------------------------------------
        Float p_CT_sa = pdfCameraRay(its_xn.p);

        Vector d_xn_to_xe = normalize(pRec_xe.p - its_xn.p);
        Float  dist_ne    = (pRec_xe.p - its_xn.p).length();

        // BSDF pdf at x_n for direction x_n → x_e (solid angle at x_n).
        // its_xn.wi is already set to the local camera direction.
        // const BSDF *bsdf_xn = its_xn.getBSDF();
        // BSDFSamplingRecord bRec_ne(its_xn, its_xn.toLocal(d_xn_to_xe), ERadiance);
        // Float p_BSDF_sa = bsdf_xn->pdf(bRec_ne);

        // Geometry term at x_e: cosθ_e / dist_ne²  (converts SA at x_n → area at x_e)
        Float cosTheta_e = 1.f; // absDot(pRec_xe.n, d_xn_to_xe);
        Float cosTheta_ne = absDot(its_xn.geoFrame.n, -d_xn_to_xe);
        Float geom_ne    = cosTheta_ne / (dist_ne * dist_ne);
        Float pdf_xn_new = bvh->pdfGeometryCamera(
                                m_scene, m_cameraOrigin, m_sensor.get(), pRec_xe, its_xn);
        // q values in reduced joint-area measure (common factor cosθ_n/dist_on² cancelled):
        //   q1 = p_CT_sa * p_BSDF_sa * cosθ_e/dist_ne²   [CT+BSDF: BSDF in SA, geometry converts to area]
        //   q2 = p_CT_sa * p_emitter_area                  [CT+NEE: direct area sampling, no extra geom]
        //   q3 = p_emitter_area * p_BVH_sa                 [Glint]
        Float q1 = 0.f; // p_CT_sa * p_BSDF_sa * geom_ne;   // CT+BSDF
        Float q2 = p_CT_sa * pRec_xe.pdf / cosTheta_ne;            // CT+NEE: use area PDF directly
        Float q3 =  pdf_xn_new * pRec_xe.pdf;             // Glint: p_emitter_area * p_BVH_sa
        // std::cout << "N: " << pRec_xe.n.toString() << ", d: " << d_xn_to_xe.toString() << std::endl;
        /*
        std::cout << "traceSample(): cos: " << cosTheta_e << ", dist: " << dist_ne
        //           << ", CT+BSDF: " << q1 << "= " << p_CT_sa << " * " << p_BSDF_sa << " * " << geom_ne
                   << ", CT+NEE: " << q2 << "= " << p_CT_sa << " * " << pRec_xe.pdf
                   << ", Glint: " << q3 << "= " << pdf_xn << " / " << geom_ne << " * " << pRec_xe.pdf << std::endl;
                   */
        // ----------------------------------------------------------------
        // 6. Full estimator with 3-way MIS weight
        // ----------------------------------------------------------------
        Spectrum value = W_e * contrib / (pdf_xn * pRec_xe.pdf);
        if (!value.isValid())
            return;
        
        Float weight = miWeight3(q3, q1, q2, q3);
        result->putLightSample(dRec.uv, value * weight);  // CT+NEE technique
    }

    // -----------------------------------------------------------------------
    // Path tracing sample — equivalent to MIPathTracer::Li() but invoked
    // directly inside the worker.  The camera ray is generated here (normally
    // done by SamplingIntegrator's render loop before calling Li()).
    // MIS weights for length-2 paths are computed inside pathLi (depth 0)
    // using the 3-way power heuristic against traceSample.
    // -----------------------------------------------------------------------
    void pathTraceSample(ONEEEHGSPTracerMISWorkResult *result, const Point2i &pixelOffset,
                         SamplingBVH *bvh) {
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

        if (needsApertureSample) {
            apertureSample = rRec.nextSample2D();
        }
        if (needsTimeSample) {
            timeSample = rRec.nextSample1D();
        }
        
        /*
        std::cout << "Offset: " << pixelOffset.toString() 
                << " , SamplePos: " << samplePos.toString() 
                << ", ApertureSample: " << apertureSample.toString()
                << ", TimeSample: " << timeSample
                << std::endl;
        */
        Spectrum spec = m_sensor->sampleRayDifferential(
            sensorRay, samplePos, apertureSample, timeSample);

        sensorRay.scaleDifferential(diffScaleFactor);
        spec *= pathLi(sensorRay, rRec);
        result->putSample(samplePos, spec);
        return;
        
        /*
        // Generate camera ray for this pixel
        RayDifferential ray;
        Point2 random = m_sampler->next2D();
        Point2 samplePos   = Point2(pixelOffset) + m_sampler->next2D();
        // std::cout << "Offset: " << pixelOffset.toString() 
        //           << " , Random: " << random.toString()
        //           << ", SamplePos: " << samplePos.toString() << std::endl;
        Point2 apertureSample(0.5f); // TODO: support lens sampling
        Float  timeSample     (0.5f); // TODO: support motion blur

        Spectrum sensorWeight = m_sensor->sampleRayDifferential(
            ray, samplePos, apertureSample, timeSample);
        if (sensorWeight.isZero())
            return;

        // Run path tracing Li loop (copied from MIPathTracer::Li)
        Spectrum Li = pathLi(ray);
        if (Li.isZero())
            return;

        result->putSample(samplePos, Li);
        */
    }



#if ENABLE_ONEEHGSPTRACERMIS_DOUBLESTEP != 1

    Spectrum pathLi(const RayDifferential &r, RadianceQueryRecord &rRec) {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        while (rRec.depth <= m_config.maxDepth || m_config.maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_config.hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_config.hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_config.maxDepth && m_config.maxDepth > 0)
                || (m_config.strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_config.strictNormals
                            || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;

                        // std::cout << bsdfPdf << " " << rRec.depth << std::endl;
                        /* Weight using the power heuristic */
                        Float weight;
                        if (rRec.depth == 1) {
                            // 3-way MIS: CT+NEE vs CT+BSDF vs Glint
                            Float p_CT_sa   = pdfCameraRay(its.p);
                            Float cosTheta_e = 1.0f; // absDot(dRec.n, dRec.d);
                            Float cosTheta_ne = absDot(its.geoFrame.n, -dRec.d);
                            Float geom_ne    = cosTheta_e / (dRec.dist * dRec.dist);

                            PositionSamplingRecord pRec_xe;
                            pRec_xe.p      = dRec.p;
                            pRec_xe.n      = dRec.n;
                            pRec_xe.object = dRec.object;
                            pRec_xe.measure = dRec.measure;
                            Float p_emitter = scene->pdfEmitterPosition(pRec_xe);
                            Float p_BVH_sa  = bvh->pdfGeometryCamera(
                                scene, m_cameraOrigin, m_sensor.get(), pRec_xe, its);

                            // q values in reduced joint-area measure:
                            //   q1 = p_CT_sa * p_BSDF_sa * cosθ_e/dist²  [BSDF in SA → area]
                            //   q2 = p_CT_sa * p_emitter_area              [NEE uses area directly]
                            //   q3 = p_emitter_area * p_BVH_sa
                            Float q1 = 0.f; // p_CT_sa * bsdf->pdf(bRec) * geom_ne;
                            Float q2 = p_CT_sa * p_emitter / cosTheta_ne;            // CT+NEE (area PDF directly)
                            Float q3 = p_BVH_sa * p_emitter;           // Glint: p_emitter_area * p_BVH_sa
                            
                            
                            // std::cout << "N: " << pRec_xe.n.toString() << ", d: " << dRec.d.toString() << std::endl;
                            /*
                            std::cout << "pathLi(): cos: " << cosTheta_e << ", dist: " << dRec.dist
                                      << ", CT+BSDF: " << q1 << "= " << p_CT_sa << " * " << bsdf->pdf(bRec) << " * " << geom_ne
                                      << ", CT+NEE: " << q2 << "= " << p_CT_sa << " * " << p_emitter
                                      << ", Glint: " << q3 << "= " << p_BVH_sa << " * " << p_emitter << std::endl;
                                      */
                            weight = miWeight3(q2, q1, q2, q3);  // CT+NEE technique
                        } else {
                            weight = miWeight(dRec.pdf, bsdfPdf);
                        }
                        Li += throughput * value * bsdfVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Save x_n intersection for depth-1 3-way MIS (needed after BSDF bounce).
               newQuery() initialises depth=1, so depth==1 is the first surface hit. */
            Intersection its_xn;
            if (rRec.depth == 1 && bvh)
                its_xn = its;

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;

            scattered |= bRec.sampledType != BSDF::ENull;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = its.toWorld(bRec.wo);
            Float woDotGeoN = dot(its.geoFrame.n, wo);
            if (m_config.strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(its.p, wo, ray.time);
            if (scene->rayIntersect(ray, its)) {
                /* Intersected something - check if it was a luminaire */
                if (its.isEmitter()) {
                    value = its.Le(-ray.d);
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = scene->getEnvironmentEmitter();

                if (env) {
                    if (m_config.hideEmitters && !scattered)
                        break;

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;
                    hitEmitter = true;
                } else {
                    break;
                }
            }

            /* Keep track of the throughput and relative
               refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            /* If a luminaire was hit, estimate the local illumination and
               weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                   implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;

                Float weight;
                if (rRec.depth == 1 && bvh && !(bRec.sampledType & BSDF::EDelta)) {
                    // 3-way MIS: CT+BSDF vs CT+NEE vs Glint
                    // ray.o = x_n position; its.p = x_e position
                    Float p_CT_sa    = pdfCameraRay(ray.o);
                    Float cosTheta_e = absDot(its.geoFrame.n, -ray.d);
                    Float cosTheta_ne = absDot(its_xn.geoFrame.n, -ray.d);
                    Float dist_ne    = (its.p - ray.o).length();
                    Float geom_ne    = cosTheta_e / (dist_ne * dist_ne);

                    PositionSamplingRecord pRec_xe;
                    pRec_xe.p      = its.p;
                    pRec_xe.n      = its.geoFrame.n;
                    pRec_xe.object = its.shape->getEmitter();
                    Float p_emitter = scene->pdfEmitterPosition(pRec_xe);
                    Float p_BVH_sa  = bvh->pdfGeometryCamera(
                        scene, m_cameraOrigin, m_sensor.get(), pRec_xe, its_xn);

                    // Same reduced joint-area measure as NEE block:
                    Float q1 = p_CT_sa * bsdfPdf  * geom_ne;  // CT+BSDF
                    Float q2 = p_CT_sa * p_emitter / cosTheta_ne;            // CT+NEE (area PDF directly)
                    Float q3 = p_emitter * p_BVH_sa;
                    weight = miWeight3(q1, q1, q2, q3);  // CT+BSDF technique
                } else {
                    weight = miWeight(bsdfPdf, lumPdf);
                }
                Li += throughput * value * weight;
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_config.rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                   while accounting for the solid angle compression at refractive
                   index boundaries. Stop with at least some probability to avoid
                   getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
        }

        /* Store statistics */
        // avgPathLength.incrementBase();
        // avgPathLength += rRec.depth;

        return Li;
    }
// USE_PT
#else 
// use doublestep
#define NEE_MULT 1.f
#define L2_NEE_MULT 1.f
    Spectrum pathLi(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        // Unpack config:
        bool m_maxDepth = m_config.maxDepth;
        bool m_doAdditionalVertex = m_config.m_doAdditionalVertex;
        bool m_useBVH = m_config.m_useBVH;
        bool m_disableNee = m_config.m_disableNee;
        bool m_disableFirstBounceNEE = m_config.m_disableFirstBounceNEE;
        float m_debug1 = m_config.m_debug1, m_debug2 = m_config.m_debug2, m_debug3 = m_config.m_debug3, m_debug4 = m_config.m_debug4; // debug flags for bias isolation
        bool m_strictNormals = m_config.strictNormals;
        bool m_hideEmitters = m_config.hideEmitters;
        int m_rrDepth = m_config.rrDepth;

        // std::cout << "Starting path with ray: " << ray.toString() << std::endl;
#if DO_TIMING == 1
        // Log(EInfo, "Timing enabled for ONEEEHGSPath2StepIntegrator. This may impact performance.");
        ref<Timer> timer = new Timer(false);
        timer->start(); // start a single timer that accumulates total path time

        float time_2NEE = 0.0f, time_BSDF_NEE = 0.0f;
        float time_bvh_sample = 0.0f, time_2neecontrib = 0.0f, time_bsdf2neecontrib = 0.0f;
        float time_bvh_pdf = 0.0f;
        float time_total = 0.0f;
#endif 

        /* Perform the first ray intersection */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        bool prev_evaluated_l2 = false; // avoid double counting if both strategies account for the same L2 contribution

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            // std::cout << "[Doublestep] depth=" << rRec.depth
            //           << " itsValid=" << its.isValid()
            //           << " doAdditionalVertex=" << m_doAdditionalVertex
            //           << " useBVH=" << m_useBVH
            //           << " disableNEE=" << m_disableNee
            //           << " debug1=" << m_debug1
            //           << " debug2=" << m_debug2
            //           << " debug3=" << m_debug3
            //           << " debug4=" << m_debug4
            //           << std::endl;

            /* Emission (Le) from current intersection */
            bool do_nee = m_disableFirstBounceNEE ? (rRec.depth > 1) : true; // optionally disable NEE at the first bounce to isolate its contribution

            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered) && (do_nee))
                Li += throughput * its.Le(-ray.d);

            /* Max depth and normal checks */
            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0) || 
                (m_strictNormals && dot(ray.d, its.geoFrame.n) * Frame::cosTheta(its.wi) >= 0)) {
                break;
            }

            /* ==================================================================== */
            /* STRATEGY 1: Standard Direct Lighting (Length-1 NEE)                  */
            /* ==================================================================== */
            bool run_standard_nee = !m_disableNee && !prev_evaluated_l2; // disable nee if last bounce already accounted for the L2 contribution
            DirectSamplingRecord dRec(its);
            if (run_standard_nee && (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) && (bsdf->getType() & BSDF::ESmooth) && do_nee) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero() && (!m_strictNormals || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {
                        // Standard MIS between BSDF and Direct Light
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle) ? bsdf->pdf(bRec) : 0;
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += NEE_MULT * throughput * value * bsdfVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /* PREPARE SHARED SAMPLES FOR LENGTH-2 INDIRECT MIS                     */
            /* ==================================================================== */
            
            // 1. Sample the Emitter (xe) ONCE. This effectively fixes the endpoint for both strategies.
            PositionSamplingRecord pRec_L2;
            Spectrum xe_val = scene->sampleEmitterPosition(pRec_L2, rRec.nextSample2D());

            bool l2_possible = !xe_val.isZero() && m_doAdditionalVertex && (rRec.depth + 1 < m_maxDepth)
                 && (m_debug1 > 0.5f);

            // std::cout << "[Doublestep] L2 gate: xe_val_zero=" << xe_val.isZero()
            //           << " doAdditionalVertex=" << m_doAdditionalVertex
            //           << " depth+1<maxDepth=" << (rRec.depth + 1 < m_maxDepth)
            //           << " debug1>0.5=" << (m_debug1 > 0.5f)
            //           << " => l2_possible=" << l2_possible << std::endl;

            prev_evaluated_l2 = l2_possible;

            // Store results of Strategy A (BVH)
            Intersection its_y; 
            Spectrum contrib_2NEE(0.0f);
            Float pdf_2NEE_solidAngle = 0.0f;

            /* ==================================================================== */
            /* STRATEGY A: Length-2 NEE (The BVH Shortcut)                          */
            /* Path: xs -> y -> xe                                                  */
            /* ==================================================================== */



            if (l2_possible && m_useBVH) {
                // std::cout << "[Doublestep] Strategy A entered (BVH)" << std::endl;
                #if DO_TIMING == 1
                uint64_t t0_bvh_sample = timer->getNanoseconds();
                #endif
                // Note: We assume sampleGeometry returns PDF in Solid Angle (matching your pdfGeometry) 
                // or we might need to convert it here if sampleGeometry returns Area measure.
                // For this snippet, assuming consistent Solid Angle usage as per your comment.
                bool success = scene->getSamplingBVH()->sampleGeometry(
                    scene,
                    rRec.sampler,
                    its,
                    pRec_L2,
                    its_y,
                    pdf_2NEE_solidAngle
                );

                // std::cout << "[Doublestep] Strategy A sampleGeometry: success=" << success
                //           << " pdf_2NEE_solidAngle=" << pdf_2NEE_solidAngle << std::endl;

                #if DO_TIMING == 1
                uint64_t t1_bvh_sample = timer->getNanoseconds();
                time_bvh_sample += (float) (t1_bvh_sample - t0_bvh_sample);
                #endif
                
                if (success && pdf_2NEE_solidAngle > 0) {
                    its_y.wi = its_y.toLocal(normalize(its.p - its_y.p));
#if 1
                    {// Check visibility for both rays: xs->y and y->xe
                        // 1. Check visibility from xs to y

                        #if DO_TIMING == 1
                        uint64_t t0_2nee = timer->getNanoseconds();
                        #endif
                        Vector d_xy = its_y.p - its.p;
                        Float dist_xy = d_xy.length();
                        d_xy /= dist_xy;
                        Ray shadowRay_xy(its.p, d_xy, Epsilon, dist_xy * (1.0f - ShadowEpsilon), ray.time);
                        
                        // 2. Check visibility from y to xe
                        Vector d_ye = pRec_L2.p - its_y.p;
                        Float dist_ye = d_ye.length();
                        d_ye /= dist_ye;
                        Ray shadowRay_ye(its_y.p, d_ye, Epsilon, dist_ye * (1.0f - ShadowEpsilon), ray.time);


                        // Only evaluate the contribution if BOTH segments are completely unoccluded
                        bool occ_xy = scene->rayIntersect(shadowRay_xy);
                        bool occ_ye = scene->rayIntersect(shadowRay_ye);

                        // std::cout << "[Doublestep] Strategy A visibility: occ_xy=" << occ_xy
                        //           << " occ_ye=" << occ_ye << std::endl;

                        if (!occ_xy && !occ_ye) {
                            // std::cout << "[Doublestep] Calling evaluatePathContribution from Strategy A" << std::endl;
                            contrib_2NEE = L2_NEE_MULT * evaluatePathContribution(
                                scene, its, its_y, pRec_L2, throughput, bsdf
                            ) / pdf_2NEE_solidAngle;
                        }
                        //     std::cout << "[Doublestep] Strategy A contrib_2NEE=" << contrib_2NEE.toString() << std::endl;
                        // } else {
                        //     std::cout << "[Doublestep] Strategy A skipped: blocked visibility" << std::endl;
                        // }
                    }
#else
                    // old: no visibility, enable if we do vis in BVH.sampleGeometry

                    // Pass pdf_2NEE_solidAngle to the function
                    contrib_2NEE = L2_NEE_MULT * evaluatePathContribution(
                        scene, its, its_y, pRec_L2, throughput, bsdf
                    ) / pdf_2NEE_solidAngle;
#endif
                    

                    // if (contrib_2NEE.getLuminance() > 10000.f) {
                    //     std::cout << "Debug: Large contrib from 2NEE path: " << contrib_2NEE.toString() << " with pdf_2NEE_solidAngle = " << pdf_2NEE_solidAngle << std::endl;
                    // }
                }
            }

            /* ==================================================================== */
            /* STRATEGY B: BSDF Sampling (The Standard Path)                        */
            /* Path: xs -> z ...                                                    */
            /* ==================================================================== */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            
            if (bsdfWeight.isZero()) break;

            scattered |= bRec.sampledType != BSDF::ENull;
            const Vector wo = its.toWorld(bRec.wo);
            
            if (m_strictNormals && dot(its.geoFrame.n, wo) * Frame::cosTheta(bRec.wo) <= 0) break;

            // Trace the ray xs -> z
            ray = Ray(its.p, wo, ray.time);
            Intersection its_z;
            bool hit_z = scene->rayIntersect(ray, its_z);

            /* ==================================================================== */
            /* MIS: COMBINE L2 ESTIMATES (BVH vs BSDF+NEE)                          */
            /* ==================================================================== */
            if (l2_possible) {
                
                // --- 1. Weight for Strategy A (BVH) ---
                // Competing PDF: Probability that BSDF sampling would have generated direction to 'y'
                // debug2 > 0: skip Strategy A (BVH) contribution entirely
                if (!contrib_2NEE.isZero() && m_debug2 > 0.5f) {
                    Vector dir_xs_y = normalize(its_y.p - its.p);
                    BSDFSamplingRecord bRec_check(its, its.toLocal(dir_xs_y));
                    Float pdf_bsdf_at_y = bsdf->pdf(bRec_check); // Solid Angle

                    // debug4 > 0: force weight=1 to diagnose double-counting / PDF errors
                    Float weight_A = (m_debug4 > 0.5f)
                        ? miWeight(pdf_2NEE_solidAngle, pdf_bsdf_at_y)
                        : 1.0f;
                    
                    Li += weight_A * contrib_2NEE;
                }

                // --- 2. Evaluate Strategy B (BSDF + NEE) ---
                // We are at 'z' (its_z). We try to connect z -> xe.
                // Only possible if z is a valid surface and not an emitter (emitters handled by L1)
                // debug3 > 0: skip Strategy B (BSDF+NEE L2) contribution entirely
                if (hit_z && its_z.isValid() && !its_z.isEmitter() && m_debug3 > 0.5f) {
                    
                    
                    // Check visibility z -> xe
                    Ray shadowRay(its_z.p, normalize(pRec_L2.p - its_z.p), Epsilon, distance(pRec_L2.p, its_z.p)-Epsilon, ray.time);
                    if (!scene->rayIntersect(shadowRay)) {
                        #if DO_TIMING == 1
                        uint64_t t0_bsdf2nee = timer->getNanoseconds();
                        #endif

                        Spectrum contrib_BSDF_NEE = evaluatePathContribution(scene, its, its_z, pRec_L2, throughput, bsdf) / bsdfPdf;


                        #if DO_TIMING == 1
                        uint64_t t1_bsdf2nee = timer->getNanoseconds();
                        time_bsdf2neecontrib += (float) (t1_bsdf2nee - t0_bsdf2nee);
                        #endif
                        
                        if (!contrib_BSDF_NEE.isZero()) {
                            
                            // Competing PDF: Probability that BVH would have generated 'z'
                            Float pdf_2NEE_at_z = 0.0f;

                            #if DO_TIMING == 1
                            uint64_t t0_bvh_pdf = timer->getNanoseconds();
                            #endif
                            
                            if (its_z.shape && its_z.shape->getClass()->derivesFrom(MTS_CLASS(TriMesh))) {
                                const TriMesh *mesh = static_cast<const TriMesh *>(its_z.shape);
                                // NOTE: pdfGeometry must return Solid Angle.
                                pdf_2NEE_at_z = scene->getSamplingBVH()->pdfGeometry(
                                    scene, 
                                    its, 
                                    pRec_L2,
                                    its_z
                                );
                            }

                            #if DO_TIMING == 1
                            uint64_t t1_bvh_pdf = timer->getNanoseconds();
                            time_bvh_pdf += (float) (t1_bvh_pdf - t0_bvh_pdf);
                            #endif
                            
                            // debug4 > 0: force weight=1 to diagnose double-counting / PDF errors
                            Float weight_B = (m_debug4 > 0.5f)
                                ? miWeight(bsdfPdf, pdf_2NEE_at_z)
                                : 1.0f;

                            Li += weight_B * contrib_BSDF_NEE;
                        }
                    }
                }
            }
            //         std::cout << "[Doublestep] Strategy B visibility z->xe occ_ze=" << occ_ze << std::endl;

            //         if (!occ_ze) {
            //             std::cout << "[Doublestep] Calling evaluatePathContribution from Strategy B (bsdfPdf="
            //                       << bsdfPdf << ")" << std::endl;
            //             Spectrum contrib_BSDF_NEE = evaluatePathContribution(scene, its, its_z, pRec_L2, throughput, bsdf) / bsdfPdf;
            //             std::cout << "[Doublestep] Strategy B contrib_BSDF_NEE=" << contrib_BSDF_NEE.toString() << std::endl;
            //         } else {
            //             std::cout << "[Doublestep] Strategy B skipped: blocked z->xe" << std::endl;
            //         }
            //     } else {
            //         std::cout << "[Doublestep] Strategy B not entered: hit_z=" << hit_z
            //                   << " its_z_valid=" << its_z.isValid()
            //                   << " its_z_isEmitter=" << (hit_z ? its_z.isEmitter() : false)
            //                   << " debug3>0.5=" << (m_debug3 > 0.5f) << std::endl;
            //     }
            // } else {
            //     std::cout << "[Doublestep] L2 block skipped entirely (l2_possible=false)" << std::endl;
            // }

            /* ==================================================================== */
            /* CONTINUE RECURSION                                                   */
            /* ==================================================================== */
            
            throughput *= bsdfWeight; // update throughput for the sampled BSDF direction
            if (!hit_z) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            } else if (its_z.isEmitter()) {
                // Hit light directly (standard PT). 
                // Note: This is part of the L1 calculation (BSDF strategy for direct light)
                dRec.setQuery(ray, its_z);
                Spectrum value = its_z.Le(-ray.d);
                Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                Li += throughput * value * miWeight(bsdfPdf, lumPdf);
                break; // terminate path — emitter already accounted for above
            }

            its = its_z;
            eta *= bRec.eta;

            if (rRec.depth++ >= m_rrDepth) {
                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q) break;
                throughput /= q;
            }
        }

        #if DO_TIMING == 1
        if (time_bvh_pdf>0) {
            // ONLY print timing info if we have data for the pdf
            time_total = (float) timer->getNanoseconds();

            float unaccounted_time = time_total - (time_bvh_sample + time_2neecontrib + time_bsdf2neecontrib + time_bvh_pdf);
                        
            // Table-style timing report
            float total_time = time_bvh_sample + time_2neecontrib + time_bsdf2neecontrib + time_bvh_pdf;
            int effective_bounces = std::max(1, (int)rRec.depth);
    
            Log(EInfo, "Timing Report (ns):");
            Log(EInfo, "%-15s | %10s | %10s | %15s | %10s | %10s | %10s", 
                "Metric", "BVH Sample", "2NEE Contrib", "BSDF2NEE Contrib", "BVH PDF", "Other", "Total Path");
            Log(EInfo, "%s", std::string(125, '-').c_str());
            Log(EInfo, "%-15s | %10.0f | %10.0f | %15.0f | %10.0f | %10.0f | %10.0f", 
                "Total Time (ns)", time_bvh_sample, time_2neecontrib, time_bsdf2neecontrib, time_bvh_pdf, unaccounted_time, time_total);
            Log(EInfo, "%-15s | %10.0f | %10.0f | %15.0f | %10.0f | %10.0f | %10.0f", 
                "Per Bounce (ns)", time_bvh_sample / effective_bounces, time_2neecontrib / effective_bounces, 
                time_bsdf2neecontrib / effective_bounces, time_bvh_pdf / effective_bounces, unaccounted_time / effective_bounces, 
                time_total / effective_bounces);
            
            if (time_total > 0) {
                Log(EInfo, "%-15s | %10.2f%% | %10.2f%% | %15.2f%% | %10.2f%% | %10.2f%% | %5s", 
                    "Percentage (%)", (time_bvh_sample / time_total) * 100, 
                    (time_2neecontrib / time_total) * 100, 
                    (time_bsdf2neecontrib / time_total) * 100, 
                    (time_bvh_pdf / time_total) * 100, (unaccounted_time / time_total) * 100, "100%");
            }

            // Log the overhead of our approach, relative to the unaccounted time:
            if (unaccounted_time > 0) {
                float overhead = ((time_bvh_sample + time_2neecontrib + time_bsdf2neecontrib + time_bvh_pdf) / unaccounted_time) * 100;
                Log(EInfo, "Overhead of L2 strategies relative to unaccounted time: %.2f%%", overhead);
            }

        }
        
        #endif

        // avgPathLength.incrementBase();
        // avgPathLength += rRec.depth;
        return Li;
    }

// Helper to evaluate raw throughput of a 2-segment path (xs -> p -> xe)
    // Used for both the 2NEE path and the BSDF+NEE path
    // Spectrum evaluatePathContribution(const Scene *scene, const Intersection &xs, const Intersection &p, 
    //                                   const PositionSamplingRecord &pRec, const Spectrum &pathThroughput, 
    //                                   const BSDF *bsdf_xs) const {
        
    //     Vector d_sp = p.p - xs.p;
    //     Float dist_sp = d_sp.length();
    //     d_sp /= dist_sp;

    //     Vector d_pe = pRec.p - p.p;
    //     Float dist_pe = d_pe.length();
    //     d_pe /= dist_pe;

    //     // BSDF at xs (connection to p)
    //     BSDFSamplingRecord bRec_xs(xs, xs.toLocal(d_sp));
    //     Spectrum f_xs = bsdf_xs->eval(bRec_xs);
        
    //     // BSDF at p (connection from xs to xe)
    //     const BSDF *bsdf_p = p.getBSDF();
    //     BSDFSamplingRecord bRec_p(p, p.toLocal(-d_sp), p.toLocal(d_pe));
    //     Spectrum f_p = bsdf_p->eval(bRec_p);

    //     if (f_xs.isZero() || f_p.isZero()) return Spectrum(0.0f);

    //     // Geometry terms
    //     Float cos_theta_p_in = std::abs(dot(p.shFrame.n, -d_sp));
    //     Float cos_theta_p_out = std::abs(dot(p.shFrame.n, d_pe));
    //     Float G_sp = cos_theta_p_in;// / (dist_sp * dist_sp);
    //     Float G_pe = 1.0f / (dist_pe * dist_pe); // Note: cos_theta at Light is handled by Emitter::eval usually, or added here

    //     // Emitter Radiance
    //     const Emitter *emitter = static_cast<const Emitter *>(pRec.object);
    //     DirectionSamplingRecord dRec_direction(-d_pe);
    //     dRec_direction.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
    //     Spectrum Le = emitter->evalPosition(pRec) * emitter->evalDirection(dRec_direction, pRec);

    //     // Combined contribution
    //     // Note: We divide by pRec.pdf (light selection pdf) here or in the main loop? 
    //     // In the main loop we sum (Weight * Contrib), so Contrib should include the 1/pdf_light term.
        
    //     return pathThroughput * f_xs * G_sp * f_p * G_pe * Le * (1.0f / pRec.pdf);
    // }
    Spectrum evaluatePathContribution(const Scene *scene, const Intersection &xs, const Intersection &y, 
                                  const PositionSamplingRecord &pRec, const Spectrum &pathThroughput, 
                                  const BSDF *bsdf_xs) const { 
    
        // std::cout << "[Doublestep] >>> evaluatePathContribution ENTERED"
        //       << " xs=" << xs.p.toString()
        //       << " y=" << y.p.toString()
        //       << " light=" << pRec.p.toString()
        //       << " pRec.pdf=" << pRec.pdf
        //       << std::endl;

        Vector d_sp = y.p - xs.p;
        d_sp = normalize(d_sp); // SA measure, no distance squared needed for this leg

        Vector d_pe = pRec.p - y.p;
        Float dist_pe = d_pe.length();
        d_pe /= dist_pe;

        // 1. BSDF at xs (Includes outgoing cos_theta_xs)
        BSDFSamplingRecord bRec_xs(xs, xs.toLocal(d_sp));
        Spectrum f_xs = bsdf_xs->eval(bRec_xs);

        // 2. BSDF at y (Includes outgoing cos_theta_y)
        const BSDF *bsdf_y = y.getBSDF();
        BSDFSamplingRecord bRec_y(y, y.toLocal(-d_sp), y.toLocal(d_pe));
        Spectrum f_p = bsdf_y->eval(bRec_y);

        if (f_xs.isZero() || f_p.isZero()) return Spectrum(0.0f);

        // 3. Geometry Term for Light Connection ONLY (No cos_theta_light, it's in Le!)
        Float G_pe = 1.0f / (dist_pe * dist_pe);

        // 4. Emitter Radiance (Includes cos_theta_light implicitly via evalDirection)
        const Emitter *emitter = static_cast<const Emitter *>(pRec.object);
        DirectionSamplingRecord dRec_direction(-d_pe);
        dRec_direction.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
        Spectrum Le = emitter->evalPosition(pRec) * emitter->evalDirection(dRec_direction, pRec);

        // std::cout << "evaluatePathContribution Debug Info:\n";
        // std::cout << "d_sp: (" << d_sp.x << ", " << d_sp.y << ", " << d_sp.z << "), length = " << (y.p - xs.p).length() << "\n";
        // std::cout << "d_pe: (" << d_pe.x << ", " << d_pe.y << ", " << d_pe.z << "), length = " << dist_pe << "\n";
        // std::cout << "f_xs: " << f_xs.toString() << "\n";
        

        // 5. Final Combine
        // We divide by pRec.pdf (Light Area PDF). pdf_2NEE (Solid Angle) is handled in the MIS loop.
        return pathThroughput * f_xs * f_p * G_pe * Le * (1.0f / pRec.pdf);
        }

// End of Doublestep implementation
#endif 


    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    // 3-way power heuristic: qNum is the qi for the technique being weighted.
    // q1, q2, q3 are all three technique q values (qNum must equal one of them).
    inline Float miWeight3(Float qNum, Float q1, Float q2, Float q3) const {
        qNum *= qNum; q1 *= q1; q2 *= q2; q3 *= q3;
        Float denom = q1 + q2 + q3;
        return denom > 0 ? qNum / denom : 0.f;
    }

    // Camera ray direction pdf (solid angle at camera) for a surface point x_n.
    Float pdfCameraRay(const Point &x_n_pos) const {
        PositionSamplingRecord pRec_cam(0.f);
        pRec_cam.p      = m_cameraOrigin;
        pRec_cam.object = m_sensor.get();
        DirectionSamplingRecord dRec_cam(normalize(x_n_pos - m_cameraOrigin));
        Vector3 ray = x_n_pos - m_cameraOrigin;
        Float dist2 = ray.lengthSquared();
        if (dist2 == 0.f)            return 0.f;
        ray /= std::sqrt(dist2);
        return m_sensor->pdfDirection(dRec_cam, pRec_cam) * (ray.z);
    }

    ref<Scene> m_scene;
    ref<Sensor> m_sensor;
    ref<Sampler> m_sampler;
    ref<ReconstructionFilter> m_rfilter;
    MemoryPool m_pool;
    ONEEEHGSPTracerMISConfiguration m_config;
    HilbertCurve2D<uint8_t> m_hilbertCurve;
    Point m_cameraOrigin;
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