#pragma once
#ifndef __MITSUBA_RENDER_BVH_SPHERICAL_AABB_H_
#define __MITSUBA_RENDER_BVH_SPHERICAL_AABB_H_

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/aabb.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/render/common.h>


MTS_NAMESPACE_BEGIN

#define EPS Epsilon

// ------------------------------------------------------
// Spherical rectangles https://blogs.autodesk.com/media-and-entertainment/wp-content/uploads/sites/162/egsr2013_spherical_rectangle.pdf
struct SphQuad {
    Point o;
    Vector x, y, z; // local reference system ’R’
    float z0, z0sq; //
    float x0, y0, y0sq; // rectangle coords in ’R’
    float x1, y1, y1sq; //
    float b0, b1, b0sq, k; // misc precomputed constants
    float S; // solid angle of ’Q’

    inline void init(Point s, Vector ex, Vector ey, Point o_) {
        o = o_;
        float exl = (ex).length(), eyl = (ey).length();
        // compute local reference system ’R’
        x = ex / exl;
        y = ey / eyl;
        z = cross(x, y);
        // compute rectangle coords in local reference system
        Vector3 d = o - s;
        z0 = dot(d, z);
        // flip ’z’ to make it point against ’Q’
        if (z0 > 0) {
            z *= -1;
            z0 *= -1;
        }
        z0sq = z0 * z0;
        x0 = dot(d, x);
        y0 = dot(d, y);
        x1 = x0 + exl;
        y1 = y0 + eyl;
        y0sq = y0 * y0;
        y1sq = y1 * y1;
        // create vectors to four vertices
        Vector3 v00 = {x0, y0, z0};
        Vector3 v01 = {x0, y1, z0};
        Vector3 v10 = {x1, y0, z0};
        Vector3 v11 = {x1, y1, z0};
        // compute normals to edges
        Vector3 n0 = normalize(cross(v00, v10));
        Vector3 n1 = normalize(cross(v10, v11));
        Vector3 n2 = normalize(cross(v11, v01));
        Vector3 n3 = normalize(cross(v01, v00));

        // compute internal angles (gamma_i)
        float g0 = acos(-dot(n0,n1));
        float g1 = acos(-dot(n1,n2));
        float g2 = acos(-dot(n2,n3));
        float g3 = acos(-dot(n3,n0));
        // compute predefined constants
        b0 = n0.z;
        b1 = n2.z;
        b0sq = b0 * b0;
        k = 2*M_PI - g2 - g3;
        // compute solid angle from internal angles
        S = g0 + g1 - k;
    }

    // Sample a direction from p towards the spherical rectangle uniformly w.r.t. solid angle
    inline void sample(const Point &p, float u, float v, Vector &w_out, float &pdf) const {
        // 1. compute ’cu’
        float au = u * S + k;
        float fu = (cos(au) * b0 - b1) / sin(au);
        float cu = 1.f/sqrt(fu*fu + b0sq) * (fu>0 ? +1.f : -1.f);
        cu = mitsuba::math::clamp(cu, -1.f, 1.f); // avoid NaNs
        // 2. compute ’xu’
        float xu = -(cu * z0) / sqrt(1.f - cu*cu);
        xu = mitsuba::math::clamp(xu, x0, x1); // avoid Infs
        // 3. compute ’yv’
        float d = sqrt(xu*xu + z0sq);
        float h0 = y0 / sqrt(d*d + y0sq);
        float h1 = y1 / sqrt(d*d + y1sq);
        float hv = h0 + v * (h1-h0), hv2 = hv*hv;
        float yv = (hv2 < 1-EPS) ? (hv*d)/sqrt(1-hv2) : y1;
        // 4. transform (xu,yv,z0) to world coords
        Point q = (p + xu*x + yv*y + z0*z);
        w_out = normalize(q - p);
        
        pdf = 1.f/S; // S is the solid angle of the rectangle
    }
    
    // Sample a direction from p towards the spherical rectangle uniformly w.r.t. solid angle
    // Also returns the sampled point on the rectangle surface
    inline void sample(const Point &p, float u, float v, Vector &w_out, Point& p_out, float &pdf) const {
        // 1. compute ’cu’
        float au = u * S + k;
        float fu = (cos(au) * b0 - b1) / sin(au);
        float cu = 1.f/sqrt(fu*fu + b0sq) * (fu>0 ? +1.f : -1.f);
        cu = mitsuba::math::clamp(cu, -1.f, 1.f); // avoid NaNs
        // 2. compute ’xu’
        float xu = -(cu * z0) / sqrt(1.f - cu*cu);
        xu = mitsuba::math::clamp(xu, x0, x1); // avoid Infs
        // 3. compute ’yv’
        float d = sqrt(xu*xu + z0sq);
        float h0 = y0 / sqrt(d*d + y0sq);
        float h1 = y1 / sqrt(d*d + y1sq);
        float hv = h0 + v * (h1-h0), hv2 = hv*hv;
        float yv = (hv2 < 1-EPS) ? (hv*d)/sqrt(1-hv2) : y1;
        // 4. transform (xu,yv,z0) to world coords
        p_out = (p + xu*x + yv*y + z0*z);
        w_out = normalize(p_out - p);
        
        pdf = 1.f/S; // S is the solid angle of the rectangle
    }
};

// ------------------------------------------------------ 
// reference: VXPG implementation by Haolin Lu. 
// See: https://github.com/SuikaSibyl/SIByLEngine2023/blob/07bf2db4d44edab74e3b9fba2c3a4c57143455da/Engine/Shaders/SRenderer/addon/vxguiding/include/vxguiding_interface.hlsli#L185
// ------------------------------------------------------

// Sample a direction from p towards the spherical rectangles of the bounds AABB facing towards it. 
// This sampling is uniform with respect to solid angle
// bounds: AABB to sample towards
// p: observation point
// u,v: uniform random numbers in [0,1)^2
// wo: output sampled direction
// pdf: output PDF value (1 / solid angle)
// Sample a direction from p towards the spherical rectangles of the bounds AABB facing towards it. 
// This sampling is uniform with respect to solid angle
MTS_EXPORT_RENDER void AABBSphSample(
    const AABB &bounds,
    const Point &p,
    float u1, float u2,
    Vector &wo,
    Float &pdf
);


// Same to the above, but also returning the sampled point and normal on the AABB surface
// Sample a direction from p towards the spherical rectangles of the bounds AABB facing towards it. 
// This sampling is uniform with respect to solid angle
// bounds: AABB to sample towards
// p: observation point
// u,v: uniform random numbers in [0,1)^2
// wo: output sampled direction
// pdf: output PDF value (1 / solid angle)
// selectedPoint: output sampled point on the AABB surface
// selectedNormal: output normal at the sampled point
// Same to the above, but also returning the sampled point and normal on the AABB surface
MTS_EXPORT_RENDER void AABBSphSample(
    const AABB &bounds,
    const Point &p,
    float u1, float u2,
    Vector &wo,
    Float &pdf,
    Point3 &selectedPoint,
    Normal &selectedNormal
);

// yeap...
inline Vector cwiseProduct(const Vector &a, const Vector &b) {
    return Vector(a.x * b.x, a.y * b.y, a.z * b.z);
}

// Compute the PDF for sampling a direction from p towards the spherical rectangles of the bounds AABB facing towards it.
// bounds: AABB to sample towards
// p: observation point
// wo: direction to evaluate
// pdf: output PDF value (1 / solid angle)
MTS_EXPORT_RENDER void AABBSphPdf(
    const AABB &bounds,
    const Point &p,
    const Vector &wo,
    Float &pdf
);

// Compute the solid angle subtended by the spherical rectangles of the bounds AABB facing towards p
// bounds: AABB to sample towards
// p: observation point
// return: solid angle value
MTS_EXPORT_RENDER Float AABBSphSolidAngle(
    const AABB &bounds,
    const Point &p
);

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_BVH_SPHERICAL_AABB_H_ */
