#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include "sampleGeometryExplicit.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLengthFastDoubleStep("Path tracer", "Average path length", EAverage);

#define L2_NEE_MULT 1
#define NEE_MULT 1

class FastDoublestepIntegrator : public MonteCarloIntegrator {
public:
    FastDoublestepIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) {
        m_doAdditionalVertex = props.getBoolean("doAdditionalVertex", true);
        m_useBVH = props.getBoolean("useBVH", true);
        m_disableNee = props.getBoolean("disableNEE", false);

        // Probabilistic L2 scheduling (unbiased via 1/p compensation).
        m_l2Probability = props.getFloat("l2Probability", 1.0f);
        m_l2MinThroughput = props.getFloat("l2MinThroughput", 0.0f);
        m_l2MaxDepth = props.getInteger("l2MaxDepth", -1);

        // Compatibility with the original debug toggles.
        m_debug1 = props.getFloat("debug1", 1.f); // gate entire L2 block
        m_debug2 = props.getFloat("debug2", 1.f); // gate strategy A contribution
        m_debug4 = props.getFloat("debug4", 1.f); // MIS on/off
    }

    FastDoublestepIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        bool contributed_l2_last_bounce = false;
        const GeometryBVH *geometryBVH = scene->getGeometryBVH();

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0) ||
                (m_strictNormals && dot(ray.d, its.geoFrame.n) * Frame::cosTheta(its.wi) >= 0)) {
                break;
            }

            bool run_standard_nee = !m_disableNee && !contributed_l2_last_bounce;
            DirectSamplingRecord dRec(its);
            if (run_standard_nee && (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) && (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero() && (!m_strictNormals || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle) ? bsdf->pdf(bRec) : 0;
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += NEE_MULT * throughput * value * bsdfVal * weight;
                    }
                }
            }

            bool contributed_l2_this_bounce = false;
            bool finiteDepthAllowsL2 = (m_maxDepth < 0 || rRec.depth + 1 < m_maxDepth);
            bool depthScheduleAllowsL2 = (m_l2MaxDepth < 0 || static_cast<int>(rRec.depth) <= m_l2MaxDepth);
            bool throughputAllowsL2 = throughput.getLuminance() >= m_l2MinThroughput;
            bool canAttemptL2 = m_doAdditionalVertex && m_useBVH && geometryBVH
                && (m_debug1 > 0.5f)
                && finiteDepthAllowsL2
                && depthScheduleAllowsL2
                && throughputAllowsL2
                && (bsdf->getType() & BSDF::ESmooth);

            if (canAttemptL2) {
                Float pSel = std::min((Float) 1.0f, std::max((Float) 0.0f, (Float) m_l2Probability));
                if (pSel > 0.0f && rRec.nextSample1D() < pSel) {
                    const Float invSel = 1.0f / pSel;

                    PositionSamplingRecord pRec_L2;
                    Spectrum xeVal = scene->sampleEmitterPosition(pRec_L2, rRec.nextSample2D());

                    if (!xeVal.isZero() && pRec_L2.pdf > 0) {
                        Intersection its_y;
                        Float pdf_2NEE_solidAngle = 0.0f;
                        bool success = geometryBVH->sampleGeometry(
                            scene,
                            rRec.sampler,
                            its,
                            pRec_L2,
                            its_y,
                            pdf_2NEE_solidAngle
                        );

                        if (success && pdf_2NEE_solidAngle > 0) {
                            Vector d_xy = its_y.p - its.p;
                            Float dist_xy = d_xy.length();
                            d_xy /= dist_xy;

                            Vector d_ye = pRec_L2.p - its_y.p;
                            Float dist_ye = d_ye.length();
                            d_ye /= dist_ye;

                            Ray shadowRay_xy(its.p, d_xy, Epsilon, dist_xy * (1.0f - ShadowEpsilon), ray.time);
                            Ray shadowRay_ye(its_y.p, d_ye, Epsilon, dist_ye * (1.0f - ShadowEpsilon), ray.time);

                            bool occ_xy = scene->rayIntersect(shadowRay_xy);
                            bool occ_ye = scene->rayIntersect(shadowRay_ye);

                            if (!occ_xy && !occ_ye && m_debug2 > 0.5f) {
                                Spectrum contrib_2NEE = L2_NEE_MULT * invSel * evaluatePathContribution(
                                    scene, its, its_y, pRec_L2, throughput, bsdf
                                ) / pdf_2NEE_solidAngle;

                                if (!contrib_2NEE.isZero()) {
                                    Vector dir_xs_y = normalize(its_y.p - its.p);
                                    BSDFSamplingRecord bRec_check(its, its.toLocal(dir_xs_y));
                                    Float pdf_bsdf_at_y = bsdf->pdf(bRec_check);

                                    Float weightA = (m_debug4 > 0.5f)
                                        ? miWeight(pdf_2NEE_solidAngle, pdf_bsdf_at_y)
                                        : 1.0f;

                                    Li += weightA * contrib_2NEE;
                                    contributed_l2_this_bounce = true;
                                }
                            }
                        }
                    }
                }
            }

            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            if (bsdfWeight.isZero())
                break;

            scattered |= bRec.sampledType != BSDF::ENull;
            const Vector wo = its.toWorld(bRec.wo);

            if (m_strictNormals && dot(its.geoFrame.n, wo) * Frame::cosTheta(bRec.wo) <= 0)
                break;

            ray = Ray(its.p, wo, ray.time);
            Intersection its_z;
            bool hit_z = scene->rayIntersect(ray, its_z);

            throughput *= bsdfWeight;
            if (!hit_z) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            } else if (its_z.isEmitter()) {
                dRec.setQuery(ray, its_z);
                Spectrum value = its_z.Le(-ray.d);
                Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                Li += throughput * value * miWeight(bsdfPdf, lumPdf);
                break;
            }

            contributed_l2_last_bounce = contributed_l2_this_bounce;

            its = its_z;
            eta *= bRec.eta;

            if (rRec.depth++ >= m_rrDepth) {
                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }
        }

        avgPathLengthFastDoubleStep.incrementBase();
        avgPathLengthFastDoubleStep += rRec.depth;
        return Li;
    }

    Spectrum evaluatePathContribution(const Scene *scene, const Intersection &xs, const Intersection &y,
            const PositionSamplingRecord &pRec, const Spectrum &pathThroughput,
            const BSDF *bsdf_xs) const {
        Vector d_sp = y.p - xs.p;
        d_sp = normalize(d_sp);

        Vector d_pe = pRec.p - y.p;
        Float dist_pe = d_pe.length();
        d_pe /= dist_pe;

        BSDFSamplingRecord bRec_xs(xs, xs.toLocal(d_sp));
        Spectrum f_xs = bsdf_xs->eval(bRec_xs);

        const BSDF *bsdf_y = y.getBSDF();
        BSDFSamplingRecord bRec_y(y, y.toLocal(-d_sp), y.toLocal(d_pe));
        Spectrum f_y = bsdf_y->eval(bRec_y);

        if (f_xs.isZero() || f_y.isZero())
            return Spectrum(0.0f);

        Float G_pe = 1.0f / (dist_pe * dist_pe);

        const Emitter *emitter = static_cast<const Emitter *>(pRec.object);
        DirectionSamplingRecord dRec_direction(-d_pe);
        dRec_direction.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
        Spectrum Le = emitter->evalPosition(pRec) * emitter->evalDirection(dRec_direction, pRec);

        return pathThroughput * f_xs * f_y * G_pe * Le * (1.0f / pRec.pdf);
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        Float denom = pdfA + pdfB;
        return denom > 0 ? (pdfA / denom) : 0.0f;
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "FastDoublestepIntegrator[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << "," << endl
            << "  doAdditionalVertex = " << m_doAdditionalVertex << "," << endl
            << "  useBVH = " << m_useBVH << "," << endl
            << "  disableNEE = " << m_disableNee << "," << endl
            << "  l2Probability = " << m_l2Probability << "," << endl
            << "  l2MinThroughput = " << m_l2MinThroughput << "," << endl
            << "  l2MaxDepth = " << m_l2MaxDepth << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_doAdditionalVertex;
    bool m_useBVH;
    bool m_disableNee;

    float m_l2Probability;
    float m_l2MinThroughput;
    int m_l2MaxDepth;

    float m_debug1, m_debug2, m_debug4;
};

MTS_IMPLEMENT_CLASS_S(FastDoublestepIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(FastDoublestepIntegrator, "Fast double step integrator");
MTS_NAMESPACE_END
