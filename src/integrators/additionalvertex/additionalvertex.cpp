#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include "sampleGeometryExplicit.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

class AdditionalVertexIntegrator : public MonteCarloIntegrator {
public:
    AdditionalVertexIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) { 
        m_doAdditionalVertex = props.getBoolean("doAdditionalVertex", false);
        m_useBVH = props.getBoolean("useBVH", false);
        m_disableNee = props.getBoolean("disableNEE", false);
    }

    /// Unserialize from a binary data stream
    AdditionalVertexIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        bool disableNee = m_disableNee; // local copy of the flag to disable NEE for the next bounces to avoid double counting

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
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

            if (!disableNee && // flag to disable NEE if needed
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                            || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                           using BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0;

                        /* Weight using the power heuristic */
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /*                     Length-2 NEE (Additional vertex)                 */
            /* ==================================================================== */
            /*
            std::cout << "additional vertex integrator - length 2 nee - before ifloop "
                      << "m_doAdditionalVertex: " << m_doAdditionalVertex << " disableNee: " << disableNee
                      << "rRec.depth: " << rRec.depth << " maxDepth: " << m_maxDepth
                      << "(rRec.depth + 1) <= maxDepth: " << ((rRec.depth + 1) <= m_maxDepth) << "\n";
            */
            if (m_doAdditionalVertex && ((rRec.depth + 1) <= m_maxDepth || m_maxDepth < 0)) {
                // Log(EInfo, "additional vertex integrator - length 2 nee - ifloop start\n");
                Spectrum additionalLi = estimateLength2NEE(scene, rRec.sampler, its, bsdf, rRec, ray);
                // Log(EInfo, "additional vertex integrator - length 2 nee - additionalLi: %f\n", additionalLi.max());
                Li += throughput * additionalLi;
                // Log(EInfo, "additional vertex integrator - length 2 nee - ifloop end\n");
                disableNee = true; // disable NEE for the next bounces to avoid double counting
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

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
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
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
                    if (m_hideEmitters && !scattered)
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
                Li += throughput * value * miWeight(bsdfPdf, lumPdf);
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
               BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;
            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
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
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        // Log(EInfo, "additional vertex integrator - end\n");
        return Li;
    }

    /* Function to estimate a length-2 path with NEE (1 bounce).
        The name of the three vertices is:
        x_s: the current vertex (intersection of the incoming ray with the geometry)
        x_p: the intermediate vertex (sampled stochastically)
        x_e: the vertex on the emitter
    */
    Spectrum estimateLength2NEE(
        const Scene *scene,
        Sampler *sampler,
        const Intersection &its_xs,
        const BSDF *bsdf,
        RadianceQueryRecord &rRec,
        const Ray &ray
    ) const {
        // Log(EInfo, "additional vertex integrator - length 2 nee - start\n");
        // 1. Sample an emitter and a point on it
        PositionSamplingRecord pRec_xe;
        Spectrum emitted_radiance = scene->sampleEmitterPosition(pRec_xe, sampler->next2D()); 
        // TODO(jorge): not really understand how sample position can return a spectrum value without 
        // knowing which direction it casts to. For now, I'll will not use that emited_radiance and reevaluate it later when
        // knowing the direction after having sampled x_e and x_s.

        // Define samples for importance heuristic

        Intersection its_xp;
        Float xp_pdf;

        // 2. Sample intermediate geometry p using brute-force stochastic logic
        if (m_useBVH) {
            if (!scene->getGeometryBVH()->sampleGeometry(scene, sampler, its_xs, pRec_xe, its_xp, xp_pdf))
                return Spectrum(0.0f);
        } else {
            if (!sampleGeometryExplicit(scene, sampler, its_xs, pRec_xe, its_xp, xp_pdf))
                return Spectrum(0.0f);
        }
        // Fill remaining Intersection fields that sampleGeometry might not set
        its_xp.wi = its_xp.toLocal(normalize(its_xs.p - its_xp.p));
        its_xp.hasUVPartials = false;

        if (xp_pdf <= 0) return Spectrum(0.0f);

        Spectrum contribution = evalLength2Contribution(scene, its_xs, its_xp, pRec_xe, true);
        Float pdf = xp_pdf * pRec_xe.pdf;

        return (contribution / pdf);
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "AdditionalVertexIntegrator[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << "," << endl
            << "  doAdditionalVertex = " << m_doAdditionalVertex << "," << endl
            << "  useBVH = " << m_useBVH << endl
            << "  disableNEE = " << m_disableNee << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_doAdditionalVertex;
    bool m_useBVH;
    bool m_disableNee; 
};

MTS_IMPLEMENT_CLASS_S(AdditionalVertexIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(AdditionalVertexIntegrator, "Additional vertex integrator");
MTS_NAMESPACE_END