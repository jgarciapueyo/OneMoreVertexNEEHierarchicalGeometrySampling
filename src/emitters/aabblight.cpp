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

#include <mitsuba/render/emitter.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/hw/gpuprogram.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/hgs/spherical_aabb.h>
#include <stdexcept>


MTS_NAMESPACE_BEGIN

/*!\plugin{aabbemitter}{AABB light}
 * \icon{emitter_area}
 * \order{2}
 * \parameters{
 *     \parameter{radiance}{\Spectrum}{
 *         Specifies the emitted radiance in units of
 *         power per unit area per unit steradian.
 *     }
 *     \parameter{samplingWeight}{\Float}{
 *         Specifies the relative amount of samples
 *         allocated to this emitter. \default{1}
 *     }
 * }
 *
 * This plugin implements an area light that emits diffuse illumination
 * from the exterior of a parent AABB shape.
 * Since the emission profile of an area light is completely diffuse, it
 * has the same apparent brightness regardless of the observer's viewing
 * direction. Furthermore, since it occupies a nonzero amount of space, an
 * area light generally causes scene objects to cast soft shadows.
 *
 * When modeling scenes involving area lights, it is preferable
 * to use spheres as the emitter shapes, since they provide a
 * particularly good direct illumination sampling strategy (see
 * the \pluginref{sphere} plugin for an example).
 *
 * To create an AABB light source, instantiate an \code{aabbshape}
 * and specify an \code{aabbemitter} instance as its child:
 *
 * \vspace{4mm}
 * \begin{xml}
 * <!-- Create a spherical light source at the origin -->
 * <shape type="aabbshape">
 *     <point name="min" x="-1" y="0" z="-1"/>
 *     <point name="max" x="1" y="2" z="1"/>
 *     <emitter type="aabbemitter">
 *         <spectrum name="radiance" value="1"/>
 *     </emitter>
 * </shape>
 * \end{xml}
 */



class AABBLight : public Emitter {
public:
    AABBLight(const Properties &props) : Emitter(props) {
        m_type |= EOnSurface;
        m_radiance = props.getSpectrum("radiance", Spectrum::getD65());
        m_power = Spectrum(0.0f); /// Don't know the power yet
        m_surfaceArea = 0.0f;
    }

    AABBLight(Stream *stream, InstanceManager *manager)
        : Emitter(stream, manager) {
        m_radiance = Spectrum(stream);
        m_power = Spectrum(stream);
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Emitter::serialize(stream, manager);
        m_radiance.serialize(stream);
        m_power.serialize(stream);
    }

    Spectrum samplePosition(PositionSamplingRecord &pRec,
            const Point2 &sample, const Point2 *extra) const {
          Point p;
          Normal n;
          sampleSurface(sample, p, n);
          pRec.p = p;
          pRec.n = n;
          pRec.pdf = (m_surfaceArea > 0.0f) ? (1.0f / m_surfaceArea) : 0.0f;
          pRec.measure = EArea;
          pRec.object = this;
        return m_power;
    }

    Spectrum evalPosition(const PositionSamplingRecord &pRec) const {
        return m_radiance * M_PI;
    }

    Spectrum eval(const Intersection &its, const Vector &d) const {
        if (dot(its.shFrame.n, d) <= 0)
            return Spectrum(0.0f);
        else
            return m_radiance;
    }

    Float pdfPosition(const PositionSamplingRecord &pRec) const {
        return (m_surfaceArea > 0.0f) ? (1.0f / m_surfaceArea) : 0.0f;
    }

    Spectrum sampleDirection(DirectionSamplingRecord &dRec,
            PositionSamplingRecord &pRec,
            const Point2 &sample, const Point2 *extra) const {
        Vector local = warp::squareToCosineHemisphere(sample);
        dRec.d = Frame(pRec.n).toWorld(local);
        dRec.pdf = warp::squareToCosineHemispherePdf(local);
        dRec.measure = ESolidAngle;
        return Spectrum(1.0f);
    }

    Spectrum evalDirection(const DirectionSamplingRecord &dRec,
            const PositionSamplingRecord &pRec) const {
        Float dp = dot(dRec.d, pRec.n);

        if (dRec.measure != ESolidAngle || dp < 0)
            dp = 0.0f;

        return Spectrum(INV_PI * dp);
    }

    Float pdfDirection(const DirectionSamplingRecord &dRec,
            const PositionSamplingRecord &pRec) const {
        Float dp = dot(dRec.d, pRec.n);

        if (dRec.measure != ESolidAngle || dp < 0)
            dp = 0.0f;

        return INV_PI * dp;
    }

    Spectrum sampleRay(Ray &ray,
            const Point2 &spatialSample,
            const Point2 &directionalSample,
            Float time) const {
        // std::cerr << "AABBLight::sampleRay is unimplemented!!" << std::endl;
        throw std::runtime_error("AABBLight::sampleRay is unimplemented!!");
    }

    Spectrum sampleDirect(DirectSamplingRecord &dRec, const Point2 &sample) const {
        Vector wi;
        Float pdf;
        Normal n;
        Point3 selectedPoint;
        AABBSphSample(m_aabb, dRec.ref, sample.x, sample.y, wi, pdf, selectedPoint, n);

        if (pdf <= 0.0f) {
            dRec.pdf = 0.0f;
            return Spectrum(0.0f);
        }

        dRec.p = selectedPoint;
        dRec.n = n;
        dRec.d = wi;
        dRec.dist = (dRec.p - dRec.ref).length();
        dRec.pdf = pdf;
        dRec.measure = ESolidAngle;
        dRec.object = this;

        // std::cout << "AABBLight::sampleDirect: pdf = " << dRec.pdf << std::endl;
        // std::cout << "  selectedPoint: " << selectedPoint.x << ", " << selectedPoint.y << ", " << selectedPoint.z << std::endl;
        // std::cout << "  normal: " << n.x << ", " << n.y << ", " << n.z << std::endl;

        
        /* Check that the emitter and reference position are oriented correctly
           with respect to each other. Note that the >= 0 check
           for 'refN' is intentional -- those sampling requests that specify
           a reference point within a medium or on a transmissive surface
           will set dRec.refN = 0, hence they should always be accepted. */
        if (dot(dRec.d, dRec.refN) >= 0 && dot(dRec.d, dRec.n) < 0 && dRec.pdf != 0) {
            return m_radiance / dRec.pdf;
        } else {
            dRec.pdf = 0.0f;
            return Spectrum(0.0f);
        }
    }

    Float pdfDirect(const DirectSamplingRecord &dRec) const {
        /* Check that the emitter and receiver are oriented correctly
           with respect to each other. */
        if (dot(dRec.d, dRec.refN) >= 0 && dot(dRec.d, dRec.n) < 0) {
            Float pdf;
            AABBSphPdf(m_aabb, dRec.ref, dRec.d, pdf);
            return pdf;
        } else {
            return 0.0f;
        }
    }

    

    AABB getAABB() const {
        return m_aabb;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "AABBLight[" << endl
            << "  radiance = " << m_radiance.toString() << "," << endl
            << "  samplingWeight = " << m_samplingWeight << "," << endl
                << "  bounds = " << m_aabb.toString() << "," << endl
                << "  surfaceArea = " << m_surfaceArea;
        oss << "," << endl
            << "  medium = " << indent(m_medium.toString()) << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
protected:
    Spectrum m_radiance, m_power;
        AABB m_aabb;
        Float m_surfaceArea;

        static Float surfaceArea(const AABB &aabb) {
            const Vector d = aabb.max - aabb.min;
            return 2.0f * (d.x * d.y + d.y * d.z + d.x * d.z);
        }

        void sampleSurface(const Point2 &sample, Point &p, Normal &n) const {
            const Vector d = m_aabb.max - m_aabb.min;
            const Float areaX = d.y * d.z;
            const Float areaY = d.x * d.z;
            const Float areaZ = d.x * d.y;
            const Float totalArea = 2.0f * (areaX + areaY + areaZ);

            Float u = sample.x * totalArea;
            Float v = sample.y;

            if (u < areaX) {
                const Float uFace = u / areaX;
                p = Point(m_aabb.min.x, m_aabb.min.y + uFace * d.y, m_aabb.min.z + v * d.z);
                n = Normal(-1.0f, 0.0f, 0.0f);
            } else if ((u -= areaX) < areaX) {
                const Float uFace = u / areaX;
                p = Point(m_aabb.max.x, m_aabb.min.y + uFace * d.y, m_aabb.min.z + v * d.z);
                n = Normal(1.0f, 0.0f, 0.0f);
            } else if ((u -= areaX) < areaY) {
                const Float uFace = u / areaY;
                p = Point(m_aabb.min.x + uFace * d.x, m_aabb.min.y, m_aabb.min.z + v * d.z);
                n = Normal(0.0f, -1.0f, 0.0f);
            } else if ((u -= areaY) < areaY) {
                const Float uFace = u / areaY;
                p = Point(m_aabb.min.x + uFace * d.x, m_aabb.max.y, m_aabb.min.z + v * d.z);
                n = Normal(0.0f, 1.0f, 0.0f);
            } else if ((u -= areaY) < areaZ) {
                const Float uFace = u / areaZ;
                p = Point(m_aabb.min.x + uFace * d.x, m_aabb.min.y + v * d.y, m_aabb.min.z);
                n = Normal(0.0f, 0.0f, -1.0f);
            } else {
                const Float uFace = (u - areaZ) / areaZ;
                p = Point(m_aabb.min.x + uFace * d.x, m_aabb.min.y + v * d.y, m_aabb.max.z);
                n = Normal(0.0f, 0.0f, 1.0f);
            }
        }

        Normal surfaceNormal(const Point &p) const {
            const Float eps = (Float) 1e-5f;
            if (std::abs(p.x - m_aabb.min.x) < eps) return Normal(-1.0f, 0.0f, 0.0f);
            if (std::abs(p.x - m_aabb.max.x) < eps) return Normal(1.0f, 0.0f, 0.0f);
            if (std::abs(p.y - m_aabb.min.y) < eps) return Normal(0.0f, -1.0f, 0.0f);
            if (std::abs(p.y - m_aabb.max.y) < eps) return Normal(0.0f, 1.0f, 0.0f);
            if (std::abs(p.z - m_aabb.min.z) < eps) return Normal(0.0f, 0.0f, -1.0f);
            return Normal(0.0f, 0.0f, 1.0f);
        }

        void setParent(ConfigurableObject *parent) {
            Emitter::setParent(parent);

            if (parent->getClass()->derivesFrom(MTS_CLASS(Shape))) {
                Shape *shape = static_cast<Shape *>(parent);
                if (m_shape == shape || shape->isCompound())
                    return;

                if (m_shape != NULL)
                    Log(EError, "An aabb light cannot be parent of multiple shapes");

                m_shape = shape;
                m_shape->configure();
                m_aabb = m_shape->getAABB();
                m_surfaceArea = m_shape->getSurfaceArea();
                m_power = m_radiance * M_PI * m_surfaceArea;
            } else {
                Log(EError, "An aabb light must be child of a shape instance");
            }
        }
private:
};

// ================ Hardware shader implementation ================

class AABBLightShader : public Shader {
public:
    AABBLightShader(Renderer *renderer, const Spectrum &radiance)
        : Shader(renderer, EEmitterShader), m_radiance(radiance) {
    }

    void resolve(const GPUProgram *program, const std::string &evalName,
            std::vector<int> &parameterIDs) const {
        parameterIDs.push_back(program->getParameterID(evalName + "_radiance", false));
    }

    void generateCode(std::ostringstream &oss, const std::string &evalName,
            const std::vector<std::string> &depNames) const {
        oss << "uniform vec3 " << evalName << "_radiance;" << endl
            << endl
            << "vec3 " << evalName << "_area(vec2 uv) {" << endl
            << "    return " << evalName << "_radiance * pi;" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_dir(vec3 wo) {" << endl
            << "    if (cosTheta(wo) < 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    return vec3(inv_pi);" << endl
            << "}" << endl;
    }

    void bind(GPUProgram *program, const std::vector<int> &parameterIDs,
        int &textureUnitOffset) const {
        program->setParameter(parameterIDs[0], m_radiance);
    }

    MTS_DECLARE_CLASS()
private:
    Spectrum m_radiance;
};
 
Shader *AABBLight::createShader(Renderer *renderer) const {
    return new AABBLightShader(renderer, m_radiance);
}

MTS_IMPLEMENT_CLASS(AABBLightShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(AABBLight, false, Emitter)
MTS_EXPORT_PLUGIN(AABBLight, "AABB light");
MTS_NAMESPACE_END
