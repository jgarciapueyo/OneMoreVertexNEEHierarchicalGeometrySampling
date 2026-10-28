#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

/*!\plugin{oneehgspath_2step}{One-more-vertex path tracer}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Longest path depth (-1 = infinite) \default{-1}}
 *     \parameter{rrDepth}{\Integer}{Depth at which Russian roulette starts \default{5}}
 *     \parameter{strictNormals}{\Boolean}{See the \pluginref{path} plugin \default{false}}
 *     \parameter{hideEmitters}{\Boolean}{See the \pluginref{path} plugin \default{false}}
 *     \parameter{doAdditionalVertex}{\Boolean}{Enable one-more-vertex NEE \default{true}}
 *     \parameter{useBVH}{\Boolean}{Sample the intermediate vertex with the SamplingBVH
 *         (strategy A). When false, only the BSDF strategy B estimates the length-2
 *         connection \default{true}}
 *     \parameter{disableNEE}{\Boolean}{Disable standard (direct) next-event estimation
 *         \default{false}}
 *     \parameter{disableFirstBounceNEE}{\Boolean}{Skip direct illumination at the first
 *         path vertex, e.g. to render indirect illumination only \default{false}}
 * }
 *
 * Unidirectional path tracer extended with one-more-vertex next-event estimation
 * (Sec. 5 of the paper). At every path vertex x_s an emitter position x_e is
 * sampled, and the three-vertex subpath x_s -> x_n -> x_e is estimated with two
 * strategies combined with the power heuristic:
 *   A. x_n is sampled from the SamplingBVH (hierarchical geometry sampling).
 *   B. x_n is found by BSDF sampling at x_s and tracing a ray; x_e is then
 *      connected deterministically (this is the standard NEE of the next vertex).
 * Because strategy B replaces the standard NEE of the next vertex, that NEE is
 * skipped whenever the length-2 estimate was computed at the previous vertex.
 */
class OneMoreVertexPath2StepIntegrator : public MonteCarloIntegrator {
public:
    OneMoreVertexPath2StepIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) {
        m_doAdditionalVertex = props.getBoolean("doAdditionalVertex", true);
        m_useBVH = props.getBoolean("useBVH", true);
        m_disableNee = props.getBoolean("disableNEE", false);
        m_disableFirstBounceNEE = props.getBoolean("disableFirstBounceNEE", false);
    }

    /// Unserialize from a binary data stream
    OneMoreVertexPath2StepIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) {
        m_doAdditionalVertex = stream->readBool();
        m_useBVH = stream->readBool();
        m_disableNee = stream->readBool();
        m_disableFirstBounceNEE = stream->readBool();
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        /* Perform the first ray intersection */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        /* True when the length-2 estimate of the previous vertex already
           accounts for the direct illumination of the current one */
        bool prev_evaluated_l2 = false;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Direct illumination can be disabled at the first vertex */
            bool do_nee = m_disableFirstBounceNEE ? (rRec.depth > 1) : true;

            /* Emission (Le) from current intersection */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered) && do_nee)
                Li += throughput * its.Le(-ray.d);

            /* Max depth and normal checks */
            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0) ||
                (m_strictNormals && dot(ray.d, its.geoFrame.n) * Frame::cosTheta(its.wi) >= 0)) {
                break;
            }

            /* ==================================================================== */
            /*  Standard next-event estimation (length-1 connection)                */
            /* ==================================================================== */
            bool run_standard_nee = !m_disableNee && !prev_evaluated_l2;
            DirectSamplingRecord dRec(its);
            if (run_standard_nee && (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) && (bsdf->getType() & BSDF::ESmooth) && do_nee) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero() && (!m_strictNormals || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {
                        /* MIS between emitter sampling and BSDF sampling */
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle) ? bsdf->pdf(bRec) : 0;
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += throughput * value * bsdfVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /*  One-more-vertex NEE: the emitter vertex x_e is shared by both       */
            /*  strategies of the length-2 connection x_s -> x_n -> x_e             */
            /* ==================================================================== */
            PositionSamplingRecord pRec_L2;
            Spectrum xe_val = scene->sampleEmitterPosition(pRec_L2, rRec.nextSample2D());

            bool l2_possible = !xe_val.isZero() && m_doAdditionalVertex && (rRec.depth + 1 < m_maxDepth);
            prev_evaluated_l2 = l2_possible;

            /* Strategy A: sample x_n (here 'y') from the SamplingBVH */
            Intersection its_y;
            Spectrum contrib_2NEE(0.0f);
            Float pdf_2NEE_solidAngle = 0.0f;

            if (l2_possible && m_useBVH) {
                /* The returned PDF is in solid angle at x_s */
                bool success = scene->getSamplingBVH()->sampleGeometry(
                    scene, rRec.sampler, its, pRec_L2, its_y, pdf_2NEE_solidAngle);

                if (success && pdf_2NEE_solidAngle > 0) {
                    its_y.wi = its_y.toLocal(normalize(its.p - its_y.p));

                    /* Visibility of both segments x_s -> y and y -> x_e */
                    Vector d_xy = its_y.p - its.p;
                    Float dist_xy = d_xy.length();
                    d_xy /= dist_xy;
                    Ray shadowRay_xy(its.p, d_xy, Epsilon, dist_xy * (1.0f - ShadowEpsilon), ray.time);

                    Vector d_ye = pRec_L2.p - its_y.p;
                    Float dist_ye = d_ye.length();
                    d_ye /= dist_ye;
                    Ray shadowRay_ye(its_y.p, d_ye, Epsilon, dist_ye * (1.0f - ShadowEpsilon), ray.time);

                    bool occ_xy = scene->rayIntersect(shadowRay_xy);
                    bool occ_ye = scene->rayIntersect(shadowRay_ye);

                    if (!occ_xy && !occ_ye) {
                        contrib_2NEE = evaluatePathContribution(
                            scene, its, its_y, pRec_L2, throughput, bsdf) / pdf_2NEE_solidAngle;
                    }
                }
            }

            /* ==================================================================== */
            /*  BSDF sampling: continues the path and provides strategy B           */
            /* ==================================================================== */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());

            if (bsdfWeight.isZero()) break;

            scattered |= bRec.sampledType != BSDF::ENull;
            const Vector wo = its.toWorld(bRec.wo);

            if (m_strictNormals && dot(its.geoFrame.n, wo) * Frame::cosTheta(bRec.wo) <= 0) break;

            /* Trace the ray x_s -> z */
            ray = Ray(its.p, wo, ray.time);
            Intersection its_z;
            bool hit_z = scene->rayIntersect(ray, its_z);

            if (l2_possible) {
                /* Strategy A, weighted against the BSDF density of the same direction */
                if (!contrib_2NEE.isZero()) {
                    Vector dir_xs_y = normalize(its_y.p - its.p);
                    BSDFSamplingRecord bRec_check(its, its.toLocal(dir_xs_y));
                    Float pdf_bsdf_at_y = bsdf->pdf(bRec_check); // solid angle
                    Li += miWeight(pdf_2NEE_solidAngle, pdf_bsdf_at_y) * contrib_2NEE;
                }

                /* Strategy B: connect the BSDF-sampled vertex z to x_e */
                if (hit_z && its_z.isValid() && !its_z.isEmitter()) {
                    Ray shadowRay(its_z.p, normalize(pRec_L2.p - its_z.p), Epsilon, distance(pRec_L2.p, its_z.p)-Epsilon, ray.time);
                    if (!scene->rayIntersect(shadowRay)) {
                        Spectrum contrib_BSDF_NEE = evaluatePathContribution(scene, its, its_z, pRec_L2, throughput, bsdf) / bsdfPdf;

                        if (!contrib_BSDF_NEE.isZero()) {
                            /* Density with which strategy A would have generated z (solid angle) */
                            Float pdf_2NEE_at_z = 0.0f;
                            if (its_z.shape && its_z.shape->getClass()->derivesFrom(MTS_CLASS(TriMesh)))
                                pdf_2NEE_at_z = scene->getSamplingBVH()->pdfGeometry(scene, its, pRec_L2, its_z);

                            Li += miWeight(bsdfPdf, pdf_2NEE_at_z) * contrib_BSDF_NEE;
                        }
                    }
                }
            }

            /* ==================================================================== */
            /*  Continue the path                                                   */
            /* ==================================================================== */
            throughput *= bsdfWeight;
            if (!hit_z) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            } else if (its_z.isEmitter()) {
                /* Emitter hit by BSDF sampling, weighted against emitter sampling */
                dRec.setQuery(ray, its_z);
                Spectrum value = its_z.Le(-ray.d);
                Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                Li += throughput * value * miWeight(bsdfPdf, lumPdf);
                break;
            }

            its = its_z;
            eta *= bRec.eta;

            if (rRec.depth++ >= m_rrDepth) {
                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q) break;
                throughput /= q;
            }
        }

        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        return Li;
    }

    /**
     * Throughput of the subpath x_s -> y -> x_e divided by the emitter
     * position density, excluding the density of y (handled by the caller).
     * BSDF::eval includes the cosine of its outgoing direction, so f_xs
     * carries cos(theta_s) and f_p carries cos(theta_y -> x_e); the emitter's
     * evalDirection carries the cosine at x_e. Only 1/|y - x_e|^2 remains: the
     * x_s -> y segment is in solid angle measure.
     */
    Spectrum evaluatePathContribution(const Scene *scene, const Intersection &xs, const Intersection &y,
                                      const PositionSamplingRecord &pRec, const Spectrum &pathThroughput,
                                      const BSDF *bsdf_xs) const {
        Vector d_sp = normalize(y.p - xs.p);

        Vector d_pe = pRec.p - y.p;
        Float dist_pe = d_pe.length();
        d_pe /= dist_pe;

        /* BSDF at x_s towards y */
        BSDFSamplingRecord bRec_xs(xs, xs.toLocal(d_sp));
        Spectrum f_xs = bsdf_xs->eval(bRec_xs);

        /* BSDF at y from x_s towards x_e */
        const BSDF *bsdf_y = y.getBSDF();
        BSDFSamplingRecord bRec_y(y, y.toLocal(-d_sp), y.toLocal(d_pe));
        Spectrum f_p = bsdf_y->eval(bRec_y);

        if (f_xs.isZero() || f_p.isZero()) return Spectrum(0.0f);

        Float G_pe = 1.0f / (dist_pe * dist_pe);

        const Emitter *emitter = static_cast<const Emitter *>(pRec.object);
        DirectionSamplingRecord dRec_direction(-d_pe);
        dRec_direction.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
        Spectrum Le = emitter->evalPosition(pRec) * emitter->evalDirection(dRec_direction, pRec);

        return pathThroughput * f_xs * f_p * G_pe * Le * (1.0f / pRec.pdf);
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
        stream->writeBool(m_doAdditionalVertex);
        stream->writeBool(m_useBVH);
        stream->writeBool(m_disableNee);
        stream->writeBool(m_disableFirstBounceNEE);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "OneMoreVertexPath2StepIntegrator[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << "," << endl
            << "  doAdditionalVertex = " << m_doAdditionalVertex << "," << endl
            << "  useBVH = " << m_useBVH << "," << endl
            << "  disableNEE = " << m_disableNee << "," << endl
            << "  disableFirstBounceNEE = " << m_disableFirstBounceNEE << endl
            << "]";
        return oss.str();
    }

    /// Intermediate vertices are sampled from the scene's SamplingBVH
    bool needsSamplingBVH() const { return true; }

    MTS_DECLARE_CLASS()
private:
    bool m_doAdditionalVertex;
    bool m_useBVH;
    bool m_disableNee, m_disableFirstBounceNEE;
};

MTS_IMPLEMENT_CLASS_S(OneMoreVertexPath2StepIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(OneMoreVertexPath2StepIntegrator, "One-more-vertex path tracer");
MTS_NAMESPACE_END
