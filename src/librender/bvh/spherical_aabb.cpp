#include <mitsuba/render/bvh/spherical_aabb.h>

MTS_NAMESPACE_BEGIN



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
) 
{
    
    // Inside: uniform on sphere
    if (bounds.contains(p)) {
        Float z = 1 - 2 * u1;
        Float r2 = 1 - z * z;
        Float r = r2 > 0 ? std::sqrt(r2) : 0;
        Float phi = 2 * M_PI * u2;
        wo = Vector(r * std::cos(phi), r * std::sin(phi), z);
        pdf = 1 / (2 * M_PI);
        return;
    }

    // --- FIX: Use Bounds instead of Center for visibility ---
    // Determine which faces are visible. 
    // If p is between min and max in a dimension, that face is not visible (s=0).
    Float sx = 0, sy = 0, sz = 0;
    if (p.x < bounds.min.x) sx = -1;
    else if (p.x > bounds.max.x) sx = 1;

    if (p.y < bounds.min.y) sy = -1;
    else if (p.y > bounds.max.y) sy = 1;

    if (p.z < bounds.min.z) sz = -1;
    else if (p.z > bounds.max.z) sz = 1;
    // --------------------------------------------------------

    const Point bmin = bounds.min;
    const Point bmax = bounds.max;

    SphQuad squads[3];
    Float s[3] = { 0, 0, 0 };
    
    // Pre-calculate dimensions to check for zero-area faces
    Float dx = bmax.x - bmin.x;
    Float dy = bmax.y - bmin.y;
    Float dz = bmax.z - bmin.z;

    // X face
    if (sx != 0 && dy > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float xf = (sx > 0) ? bmax.x : bmin.x;
        Point o(xf, bmin.y, bmin.z);
        Vector ex(0, dy, 0);
        Vector ey(0, 0, dz);
        squads[0].init(p, ex, ey, o);
        if (squads[0].S > 0) s[0] = squads[0].S;
    }

    // Y face
    if (sy != 0 && dx > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float yf = (sy > 0) ? bmax.y : bmin.y;
        Point o(bmin.x, yf, bmin.z);
        Vector ex(dx, 0, 0);
        Vector ey(0, 0, dz);
        squads[1].init(p, ex, ey, o);
        if (squads[1].S > 0) s[1] = squads[1].S;
    }

    // Z face
    if (sz != 0 && dx > 0 && dy > 0) { // Only consider this face if it's visible and has non-zero area
        Float zf = (sz > 0) ? bmax.z : bmin.z;
        Point o(bmin.x, bmin.y, zf);
        Vector ex(dx, 0, 0);
        Vector ey(0, dy, 0);
        squads[2].init(p, ex, ey, o);
        if (squads[2].S > 0) s[2] = squads[2].S;
    }

    Float sum = s[0] + s[1] + s[2];
    if (sum <= 0) {
        wo = Vector(0, 0, 0);
        pdf = 0;
        return;
    }

    // Face selection and remap
    Float t = u1 * sum;
    int face = 0;
    Float prev = 0;
    if (t > s[0]) { prev = s[0]; face = 1; }
    if (t > s[0] + s[1]) { prev = s[0] + s[1]; face = 2; }

    // Remap u1 to get a new random value:
    u1 = (s[face] > 0) ? (t - prev) / s[face] : 0;
    Float pdf_face = s[face] / sum; // probability of selecting this face

    Float wo_pdf = 0.f;
    squads[face].sample(p, u1, u2, wo, wo_pdf);

    // Mixture pdf
    pdf = wo_pdf * pdf_face;
}


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
) 
{
    // Inside: uniform on sphere
    if (bounds.contains(p)) {
        Float z = 1 - 2 * u1;
        Float r2 = 1 - z * z;
        Float r = r2 > 0 ? std::sqrt(r2) : 0;
        Float phi = 2 * M_PI * u2;
        wo = Vector(r * std::cos(phi), r * std::sin(phi), z);
        pdf = 1 / (2 * M_PI);
        // Note: Normal/Point inside logic is undefined/arbitrary in original, keeping it minimal
        return;
    }

    // --- FIX: Use Bounds instead of Center for visibility ---
    Float sx = 0, sy = 0, sz = 0;
    if (p.x < bounds.min.x) sx = -1;
    else if (p.x > bounds.max.x) sx = 1;

    if (p.y < bounds.min.y) sy = -1;
    else if (p.y > bounds.max.y) sy = 1;

    if (p.z < bounds.min.z) sz = -1;
    else if (p.z > bounds.max.z) sz = 1;
    // --------------------------------------------------------

    const Point bmin = bounds.min;
    const Point bmax = bounds.max;

    SphQuad squads[3];
    Float s[3] = { 0, 0, 0 };
    
    // Pre-calculate dimensions to check for zero-area faces
    Float dx = bmax.x - bmin.x;
    Float dy = bmax.y - bmin.y;
    Float dz = bmax.z - bmin.z;

    // X face
    if (sx != 0 && dy > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float xf = (sx > 0) ? bmax.x : bmin.x;
        Point o(xf, bmin.y, bmin.z);
        Vector ex(0, dy, 0);
        Vector ey(0, 0, dz);
        squads[0].init(p, ex, ey, o);
        if (squads[0].S > 0) s[0] = squads[0].S;
    }

    // Y face
    if (sy != 0 && dx > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float yf = (sy > 0) ? bmax.y : bmin.y;
        Point o(bmin.x, yf, bmin.z);
        Vector ex(dx, 0, 0);
        Vector ey(0, 0, dz);
        squads[1].init(p, ex, ey, o);
        if (squads[1].S > 0) s[1] = squads[1].S;
    }

    // Z face
    if (sz != 0 && dx > 0 && dy > 0) { // Only consider this face if it's visible and has non-zero area
        Float zf = (sz > 0) ? bmax.z : bmin.z;
        Point o(bmin.x, bmin.y, zf);
        Vector ex(dx, 0, 0);
        Vector ey(0, dy, 0);
        squads[2].init(p, ex, ey, o);
        if (squads[2].S > 0) s[2] = squads[2].S;
    }

    Float sum = s[0] + s[1] + s[2];
    if (sum <= 0) {
        wo = Vector(0, 0, 0);
        pdf = 0;
        return;
    }

    // Face selection and remap
    Float t = u1 * sum;
    int face = 0;
    Float prev = 0;
    if (t > s[0]) { prev = s[0]; face = 1; }
    if (t > s[0] + s[1]) { prev = s[0] + s[1]; face = 2; }

    // Remap u1 to get a new random value:
    u1 = (s[face] > 0) ? (t - prev) / s[face] : 0;
    Float pdf_face = s[face] / sum; 

    Float wo_pdf = 0.f;
    squads[face].sample(p, u1, u2, wo, selectedPoint, wo_pdf);

    // Mixture pdf
    pdf = wo_pdf * pdf_face;

    // normal: use the face that was sampled 
    // Since sx/sy/sz are now guaranteed to be non-zero if that face was added to 's',
    // the conditional logic here remains safe.
    switch (face) {
        case 0: // X face
            selectedNormal = Normal((sx > 0) ? 1.0f : -1.0f, 0.0f, 0.0f);
            break;
        case 1: // Y face
            selectedNormal = Normal(0.0f, (sy > 0) ? 1.0f : -1.0f, 0.0f);
            break;
        default: // Z face
            selectedNormal = Normal(0.0f, 0.0f, (sz > 0) ? 1.0f : -1.0f);
            break;
    }
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
) {
    
    // Inside: uniform on sphere
    if (bounds.contains(p)) {
        pdf = 1 / (2 * M_PI);
        return;
    }

    // Ray-AABB intersection (slab)
    Vector invDir(1 / wo.x, 1 / wo.y, 1 / wo.z);
    Vector t0 = cwiseProduct(bounds.min - p, invDir);
    Vector t1 = cwiseProduct(bounds.max - p, invDir);

    Float tmin = std::max(std::max(std::min(t0.x, t1.x), std::min(t0.y, t1.y)), std::min(t0.z, t1.z));
    Float tmax = std::min(std::min(std::max(t0.x, t1.x), std::max(t0.y, t1.y)), std::max(t0.z, t1.z));

    if (tmax < 0 || tmin > tmax) {
        pdf = 0.f;
        return;
    }

    // Intersection point on entry
    Float t = (tmin > 0) ? tmin : tmax;
    Point q = p + wo * t;

    // --- FIX: Use Bounds instead of Center for visibility ---
    // This calculation must match AABBSphSample exactly.
    Float sx = 0, sy = 0, sz = 0;
    if (p.x < bounds.min.x) sx = -1;
    else if (p.x > bounds.max.x) sx = 1;

    if (p.y < bounds.min.y) sy = -1;
    else if (p.y > bounds.max.y) sy = 1;

    if (p.z < bounds.min.z) sz = -1;
    else if (p.z > bounds.max.z) sz = 1;
    // --------------------------------------------------------

    // Identify if the hit point q actually belongs to one of the *visible* faces.
    bool hitVisible = false;
    // We only check a face if we determined it is visible (sx != 0)
    if (sx != 0) {
        Float x_face = (sx > 0) ? bounds.max.x : bounds.min.x;
        if (std::abs(q.x - x_face) <= EPS) hitVisible = true;
    }
    if (!hitVisible && sy != 0) {
        Float y_face = (sy > 0) ? bounds.max.y : bounds.min.y;
        if (std::abs(q.y - y_face) <= EPS) hitVisible = true;
    }
    if (!hitVisible && sz != 0) {
        Float z_face = (sz > 0) ? bounds.max.z : bounds.min.z;
        if (std::abs(q.z - z_face) <= EPS) hitVisible = true;
    }

    if (!hitVisible) {
        pdf = 0.f;
        return;
    }

    // Compute sum of solid angles of visible faces
    const Point bmin = bounds.min;
    const Point bmax = bounds.max;

    SphQuad squads[3];
    Float s[3] = { 0, 0, 0 };

    // Pre-calculate dimensions to check for zero-area faces
    Float dx = bmax.x - bmin.x;
    Float dy = bmax.y - bmin.y;
    Float dz = bmax.z - bmin.z;

    // X face
    if (sx != 0 && dy > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float xf = (sx > 0) ? bmax.x : bmin.x;
        Point o(xf, bmin.y, bmin.z);
        Vector ex(0, dy, 0);
        Vector ey(0, 0, dz);
        squads[0].init(p, ex, ey, o);
        if (squads[0].S > 0) s[0] = squads[0].S;
    }

    // Y face
    if (sy != 0 && dx > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float yf = (sy > 0) ? bmax.y : bmin.y;
        Point o(bmin.x, yf, bmin.z);
        Vector ex(dx, 0, 0);
        Vector ey(0, 0, dz);
        squads[1].init(p, ex, ey, o);
        if (squads[1].S > 0) s[1] = squads[1].S;
    }

    // Z face
    if (sz != 0 && dx > 0 && dy > 0) { // Only consider this face if it's visible and has non-zero area
        Float zf = (sz > 0) ? bmax.z : bmin.z;
        Point o(bmin.x, bmin.y, zf);
        Vector ex(dx, 0, 0);
        Vector ey(0, dy, 0);
        squads[2].init(p, ex, ey, o);
        if (squads[2].S > 0) s[2] = squads[2].S;
    }

    Float sum = s[0] + s[1] + s[2];
    pdf = (sum > 0) ? (1 / sum) : 0;
}

// Compute the solid angle subtended by the spherical rectangles of the bounds AABB facing towards p
// bounds: AABB to sample towards
// p: observation point
// return: solid angle value
MTS_EXPORT_RENDER Float AABBSphSolidAngle(
    const AABB &bounds,
    const Point &p
) {
    
    // Inside: full sphere
    if (bounds.contains(p)) {
        return 2.f * M_PI;
    }

    Float sx = 0, sy = 0, sz = 0;
    if (p.x < bounds.min.x) sx = -1;
    else if (p.x > bounds.max.x) sx = 1;

    if (p.y < bounds.min.y) sy = -1;
    else if (p.y > bounds.max.y) sy = 1;

    if (p.z < bounds.min.z) sz = -1;
    else if (p.z > bounds.max.z) sz = 1;
    // --------------------------------------------------------

    const Point bmin = bounds.min;
    const Point bmax = bounds.max;

    SphQuad squads[3];
    Float s[3] = { 0, 0, 0 };

    // Pre-calculate dimensions to check for zero-area faces
    Float dx = bmax.x - bmin.x;
    Float dy = bmax.y - bmin.y;
    Float dz = bmax.z - bmin.z;

    // X face
    if (sx != 0 && dy > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float xf = (sx > 0) ? bmax.x : bmin.x;
        Point o(xf, bmin.y, bmin.z);
        Vector ex(0, dy, 0);
        Vector ey(0, 0, dz);
        
        squads[0].init(p, ex, ey, o);
        if (squads[0].S > 0) s[0] = squads[0].S;
    }

    // Y face
    if (sy != 0 && dx > 0 && dz > 0) { // Only consider this face if it's visible and has non-zero area
        Float yf = (sy > 0) ? bmax.y : bmin.y;
        Point o(bmin.x, yf, bmin.z);
        Vector ex(dx, 0, 0);
        Vector ey(0, 0, dz);
        squads[1].init(p, ex, ey, o);
        if (squads[1].S > 0) s[1] = squads[1].S;
    }

    // Z face
    if (sz != 0 && dx > 0 && dy > 0) { // Only consider this face if it's visible and has non-zero area
        Float zf = (sz > 0) ? bmax.z : bmin.z;
        Point o(bmin.x, bmin.y, zf);
        Vector ex(dx, 0, 0);
        Vector ey(0, dy, 0);
        squads[2].init(p, ex, ey, o);
        if (squads[2].S > 0) s[2] = squads[2].S;
    }
    Float sum = s[0] + s[1] + s[2];
    return sum;
}

MTS_NAMESPACE_END
