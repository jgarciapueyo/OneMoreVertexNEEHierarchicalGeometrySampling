
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
#include "microfacet.h"

#include <algorithm>
#include <cmath>

MTS_NAMESPACE_BEGIN

class BasicPrincipled : public BSDF {
public:
    BasicPrincipled(const Properties &props) : BSDF(props) {
        m_reflectance = new ConstantSpectrumTexture(props.getSpectrum("reflectance", Spectrum(.5f)));
        m_roughness = new ConstantFloatTexture(props.getFloat("roughness", 0.5f));
        m_metallic = new ConstantFloatTexture(props.getFloat("metallic", 0.0f));
        m_specular = new ConstantFloatTexture(props.getFloat("specular", 0.5f));
    }

    BasicPrincipled(Stream *stream, InstanceManager *manager)
        : BSDF(stream, manager) {
        m_reflectance = ensureEnergyConservation(m_reflectance, "reflectance", 1.0f);
        m_roughness = ensureEnergyConservation(m_roughness, "roughness", 1.0f);
        m_metallic = ensureEnergyConservation(m_metallic, "metallic", 1.0f);
        m_specular = ensureEnergyConservation(m_specular, "specular", 1.0f);
    }

    void configure() {
        unsigned int extraFlags = 0;
        if (m_metallic->getMaximum().max() < 1 && m_reflectance->getMaximum().max() > 0)
            extraFlags |= EDiffuseReflection | EFrontSide;
        if (m_metallic->getMaximum().max() < 1 && m_specular->getMaximum().max() > 0)
            extraFlags |= EGlossyReflection | EFrontSide;
        if (m_metallic->getMaximum().max() > 0 && m_reflectance->getMaximum().max() > 0)
            extraFlags |= EGlossyReflection | EFrontSide;
        
        if (!m_reflectance->isConstant() || !m_roughness->isConstant() || !m_metallic->isConstant() || !m_specular->isConstant())
            extraFlags |= ESpatiallyVarying;

        m_components.clear();
        m_components.push_back(extraFlags);
   
        m_usesRayDifferentials = m_reflectance->usesRayDifferentials() || m_roughness->usesRayDifferentials() || m_metallic->usesRayDifferentials() || m_specular->usesRayDifferentials();

        BSDF::configure();
    }

    Spectrum getDiffuseReflectance(const Intersection &its) const {
        return (Spectrum(1.0f) - m_metallic->eval(its)) * m_reflectance->eval(its);
    }

    Float getRoughness(const Intersection &its, int index) const {
        return m_roughness->eval(its).average();
    }

    Float getMetallic(const Intersection &its) const {
        return saturate(m_metallic->eval(its).average());
    }

    Float getSpecularF0(const Intersection &its) const {
        return saturate(m_specular->eval(its).average());
    }

    Spectrum getRawReflectance(const Intersection &its) const {
        return m_reflectance->eval(its);
    }

    inline Vector reflect(const Vector &wi, const Normal &m) const {
        return 2 * dot(wi, m) * Vector(m) - wi;
    }

    inline Float saturate(Float value) const {
        return std::min((Float) 1.0f, std::max((Float) 0.0f, value));
    }

    inline Spectrum schlickFresnel(Float cosTheta, const Spectrum &r0) const {
        Float ct = saturate(std::abs(cosTheta));
        Float oneMinus = 1.0f - ct;
        Float Fc = oneMinus * oneMinus * oneMinus * oneMinus * oneMinus;
        return r0 * (1.0f - Fc) + Spectrum(Fc);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);
        bool hasGlossy = (bRec.typeMask & EGlossyReflection);

        Float cosThetaI = Frame::cosTheta(bRec.wi);
        Float cosThetaO = Frame::cosTheta(bRec.wo);

        if (measure != ESolidAngle || cosThetaI <= 0 || cosThetaO <= 0 || (!hasDiffuse && !hasGlossy)
            || (bRec.component != -1 && bRec.component != 0))
            return Spectrum(0.0f);

        const Spectrum reflectance = m_reflectance->eval(bRec.its);
        const Float roughness = std::max((Float) 1e-4f, m_roughness->eval(bRec.its).average());
        const Float metallic = saturate(m_metallic->eval(bRec.its).average());
        const Float specular = saturate(m_specular->eval(bRec.its).average());

        Spectrum result(0.0f);

        if (hasDiffuse) {
            Spectrum f_d = reflectance * ((1.0f - metallic) * INV_PI);
            result += f_d * cosThetaO;
        }

        if (hasGlossy) {
            Vector H = bRec.wi + bRec.wo;
            if (H.lengthSquared() > 0.0f) {
                H = normalize(H);
                MicrofacetDistribution distr(MicrofacetDistribution::EGGX, roughness, true);
                const Float D = distr.eval(H);
                const Float G = distr.G(bRec.wi, bRec.wo, H);

                Spectrum F_metal = schlickFresnel(dot(H, bRec.wo), reflectance);
                Spectrum F_dielectric = schlickFresnel(dot(H, bRec.wo), Spectrum(specular));
                Spectrum F = metallic * F_metal + (1.0f - metallic) * F_dielectric;

                Float mfTimesCos = D * G / (4.0f * cosThetaI);
                result += F * mfTimesCos;
            }
        }

        return result;
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);
        bool hasGlossy = (bRec.typeMask & EGlossyReflection);

        Float cosThetaI = Frame::cosTheta(bRec.wi);
        Float cosThetaO = Frame::cosTheta(bRec.wo);

        if (measure != ESolidAngle || cosThetaI <= 0 || cosThetaO <= 0 || (!hasDiffuse && !hasGlossy)
            || (bRec.component != -1 && bRec.component != 0))
            return 0.0f;

        const Spectrum reflectance = m_reflectance->eval(bRec.its);
        const Float roughness = std::max((Float) 1e-4f, m_roughness->eval(bRec.its).average());
        const Float metallic = saturate(m_metallic->eval(bRec.its).average());
        const Float specular = saturate(m_specular->eval(bRec.its).average());

        Float diffuseWeight = hasDiffuse ? (1.0f - metallic) * reflectance.average() : 0.0f;
        Float glossyWeight = hasGlossy ? (metallic * reflectance.average() + (1.0f - metallic) * specular) : 0.0f;
        diffuseWeight = std::max((Float) 0.0f, diffuseWeight);
        glossyWeight = std::max((Float) 0.0f, glossyWeight);

        Float weightSum = diffuseWeight + glossyWeight;
        if (weightSum == 0.0f)
            return 0.0f;

        Float probDiffuse = diffuseWeight / weightSum;
        Float probGlossy = glossyWeight / weightSum;

        Float diffusePdf = warp::squareToCosineHemispherePdf(bRec.wo);

        Float glossyPdf = 0.0f;
        if (hasGlossy) {
            Vector H = bRec.wi + bRec.wo;
            if (H.lengthSquared() > 0.0f) {
                H = normalize(H);
                MicrofacetDistribution distr(MicrofacetDistribution::EGGX, roughness, true);
                glossyPdf = distr.eval(H) * distr.smithG1(bRec.wi, H)
                    / (4.0f * cosThetaI);
            }
        }

        return probDiffuse * diffusePdf + probGlossy * glossyPdf;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        Float pdf;
        return BasicPrincipled::sample(bRec, pdf, sample);
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &_sample) const {
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection);
        bool hasGlossy = (bRec.typeMask & EGlossyReflection);

        Float cosThetaI = Frame::cosTheta(bRec.wi);
        if (cosThetaI <= 0 || (!hasDiffuse && !hasGlossy)
            || (bRec.component != -1 && bRec.component != 0))
            return Spectrum(0.0f);

        const Spectrum reflectance = m_reflectance->eval(bRec.its);
        const Float roughness = std::max((Float) 1e-4f, m_roughness->eval(bRec.its).average());
        const Float metallic = saturate(m_metallic->eval(bRec.its).average());
        const Float specular = saturate(m_specular->eval(bRec.its).average());

        Float diffuseWeight = hasDiffuse ? (1.0f - metallic) * reflectance.average() : 0.0f;
        Float glossyWeight = hasGlossy ? (metallic * reflectance.average() + (1.0f - metallic) * specular) : 0.0f;
        diffuseWeight = std::max((Float) 0.0f, diffuseWeight);
        glossyWeight = std::max((Float) 0.0f, glossyWeight);

        Float weightSum = diffuseWeight + glossyWeight;
        if (weightSum == 0.0f)
            return Spectrum(0.0f);

        Float probDiffuse = diffuseWeight / weightSum;

        bool chooseDiffuse = hasDiffuse;
        Point2 sample(_sample);

        if (hasDiffuse && hasGlossy) {
            if (sample.y < probDiffuse) {
                sample.y /= probDiffuse;
                chooseDiffuse = true;
            } else {
                sample.y = (sample.y - probDiffuse) / (1.0f - probDiffuse);
                chooseDiffuse = false;
            }
        }

        if (chooseDiffuse) {
            bRec.wo = warp::squareToCosineHemisphere(sample);
            bRec.sampledType = EDiffuseReflection;
        } else {
            MicrofacetDistribution distr(MicrofacetDistribution::EGGX, roughness, true);
            Normal m = distr.sample(bRec.wi, sample);
            bRec.wo = reflect(bRec.wi, m);
            bRec.sampledType = EGlossyReflection;

            if (Frame::cosTheta(bRec.wo) <= 0)
                return Spectrum(0.0f);
        }

        bRec.eta = 1.0f;
        bRec.sampledComponent = 0;

        pdf = this->pdf(bRec, ESolidAngle);
        if (pdf == 0.0f)
            return Spectrum(0.0f);

        return this->eval(bRec, ESolidAngle) / pdf;
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "reflectance")
                m_reflectance = static_cast<Texture *>(child);
            else if (name == "roughness")
                m_roughness = static_cast<Texture *>(child);
            else if (name == "metallic")
                m_metallic = static_cast<Texture *>(child);
            else if (name == "specular")
                m_specular = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_reflectance.get());
    }
    
    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_reflectance;
    ref<Texture> m_roughness;
    ref<Texture> m_metallic;
    ref<Texture> m_specular;
};

MTS_IMPLEMENT_CLASS_S(BasicPrincipled, false, BSDF)
MTS_EXPORT_PLUGIN(BasicPrincipled, "Basic Principled BRDF")
MTS_NAMESPACE_END