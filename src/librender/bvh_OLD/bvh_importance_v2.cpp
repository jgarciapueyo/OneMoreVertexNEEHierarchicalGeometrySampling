// V2: after Adrian chalkboard meeting first week of march


#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/core/pmf.h>

#include <algorithm>
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

MTS_NAMESPACE_BEGIN

#define NODE_IMPORTANCE_VERSION 1
// 0-> before 21/01/26: no aabb solid angle or sggx cross-section
// 1-> after  21/01/26: with aabb solid angle and sggx cross-section

#if NODE_IMPORTANCE_VERSION == 0

Float computeNodeImportance(
    const BVHNodeInfo &nodeInfo,
    const AABB &nodeBounds,
    const SurfaceSample &xs,
    const EmitterSample &xe,
    const float sggxCrossSectionScale,
    const float solidAngleScale,
    const float variancePenaltyScale,
    const float cosinePenaltyScale)
{
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= 0.0f)
    {
        return 0.0f;
    }

    // Use center of mass as representative point for the node
    const Point &nodePos = nodeInfo.centerOfMass;

    // If center of mass is not initialized (all zeros), fall back to AABB center
    Point representativePoint = nodePos;
    if (nodePos.x == 0.0f && nodePos.y == 0.0f && nodePos.z == 0.0f)
    {
        representativePoint = nodeBounds.getCenter();
    }

    // Compute directions
    Vector dirToNode = representativePoint - xs.p;
    Float distToNode = dirToNode.length();
    if (distToNode < Epsilon)
    {
        return 0.0f; // Node coincides with surface point
    }
    dirToNode /= distToNode; // normalize

    Vector dirFromNode = xe.p - representativePoint;
    Float distFromNode = dirFromNode.length();
    if (distFromNode < Epsilon)
    {
        return 0.0f; // Node coincides with emitter
    }
    dirFromNode /= distFromNode; // normalize

    // === Geometric term: G(xs, node, xe) ===
    // cos(θ_xs): angle between surface normal at xs and direction to node
    Float cosTheta_xs = dot(Vector(xs.n), dirToNode);
    if (cosTheta_xs < 0.0f)
    {
        cosTheta_xs = 0.0f; // Surface facing away from node
    }

    // cos(θ_node): angle between node mean direction and direction to emitter
    Vector meanN = nodeInfo.normalDistribution.meanDirection();
    Float cosTheta_node = dot(meanN, dirFromNode);
    if (cosTheta_node < 0.0f)
    {
        cosTheta_node = 0.0f; // Node facing away from emitter
    }

    // cos(θ_xe): angle between emitter normal and direction from node
    Vector dirToNodeFromEmitter = -dirFromNode;
    Float cosTheta_xe = dot(Vector(xe.n), dirToNodeFromEmitter);
    if (cosTheta_xe < 0.0f)
    {
        cosTheta_xe = 0.0f; // Emitter facing away from node
    }

    // Geometric term: product of cosines divided by squared distances
    Float distanceSquared = distToNode * distToNode + distFromNode * distFromNode;
    if (distanceSquared < Epsilon)
    {
        return 0.0f;
    }

    Float geometricTerm = (cosTheta_xs * cosTheta_node * cosTheta_xe) / distanceSquared;

    // === Base importance: albedo × area ===
    Float baseImportance = nodeInfo.diffuseAlbedo * nodeInfo.surfaceArea;

    // === Combined importance ===
    Float importance = baseImportance * geometricTerm;

    // Optional: Penalize high normal variance (uncertain geometry)
    // Variance proxy derived from first/second moments.
    const Float variance = nodeInfo.normalDistribution.variance();
    if (variance > 0.0f)
    {
        Float variancePenalty = 1.0f / (1.0f + variance);
        importance *= variancePenalty;
    }

    return importance;
}
#else
namespace
{
    inline Point closestPointOnAABB(const AABB &bounds, const Point &p)
    {
        Point result;
        for (int i = 0; i < 3; ++i)
        {
            if (p[i] < bounds.min[i])
                result[i] = bounds.min[i];
            else if (p[i] > bounds.max[i])
                result[i] = bounds.max[i];
            else
                result[i] = p[i];
        }
        return result;
    }

    inline Float sggxCrossSection(const Matrix3x3 &S, const Vector &w)
    {
        const Float wx = w.x, wy = w.y, wz = w.z;
        const Float sx = S.m[0][0] * wx + S.m[0][1] * wy + S.m[0][2] * wz;
        const Float sy = S.m[1][0] * wx + S.m[1][1] * wy + S.m[1][2] * wz;
        const Float sz = S.m[2][0] * wx + S.m[2][1] * wy + S.m[2][2] * wz;
        const Float wSw = wx * sx + wy * sy + wz * sz;
        return std::sqrt(std::max((Float)0.0f, wSw));
    }

    void collectSubtreePrimitiveIndices(const GeometryBVH *bvh, int nodeIndex, std::vector<uint32_t> &primitiveIndices)
    {
        const BVHNode &node = bvh->getNode(nodeIndex);
        if (node.isLeaf())
        {
            const uint32_t start = node.primitivesOffset;
            const uint32_t end = start + node.nLeafPrimitives;
            for (uint32_t prim = start; prim < end; ++prim)
                primitiveIndices.push_back(prim);
            return;
        }

        const size_t leftChild = bvh->getLeftChild(nodeIndex);
        const size_t rightChild = bvh->getRightChild(nodeIndex);
        collectSubtreePrimitiveIndices(bvh, leftChild, primitiveIndices);
        collectSubtreePrimitiveIndices(bvh, rightChild, primitiveIndices);
    }
}

Float computeNodeImportance(
    int nodeIndex,
    GeometryBVH *bvh,
    Scene *scene,
    const SurfaceSample &xs,
    const EmitterSample &xe,
    const BVHNodeInfo &nodeInfo,
    const AABB &nodeBounds,
    const float sggxCrossSectionScale,
    const float solidAngleScale,      // better to disable
    const float variancePenaltyScale, // better to disable
    const float cosinePenaltyScale)
{
    //return 1.f; // DEBUG: Node importance disabled

    // 1. Quick rejection (Safety checks)
    if (!nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon)
    {
        return 0.0f;
    }

    // if (nodeBounds.contains(xs.p)) {
    //     return .00001f*M_2_PI * nodeInfo.diffuseAlbedo * nodeInfo.surfaceArea;
    // }

    // 2. Representative Point
    // Trust centerOfMass. If it's 0,0,0, assume that's valid.
    Point representativePoint = nodeInfo.centerOfMass;

    // Optional: Fallback only if centerOfMass is clearly invalid (e.g. if you init with NaN)
    // if (std::isnan(representativePoint.x)) representativePoint = nodeBounds.getCenter();

    // 3. Vector to Node (xs -> node)
    Vector dirToNode = representativePoint - xs.p;
    Float distSqToNode = dirToNode.lengthSquared();
    Float distToNode = std::sqrt(distSqToNode);

    if (distToNode < Epsilon)
        return 0.0f;                      // Singularity check
    Vector wIn = -dirToNode / distToNode; // Normalized: Node -> xs

    // 4. Vector from Node (node -> xe)
    Vector dirFromNode = xe.p - representativePoint;
    Float distSqFromNode = dirFromNode.lengthSquared();
    Float distFromNode = std::sqrt(distSqFromNode);

    if (distFromNode < Epsilon)
        return 0.0f;
    Vector wOut = dirFromNode / distFromNode; // Normalized: Node -> xe

    // A. Surface Normal at xs (Fore-shortening at receiver)
    Float cosTheta_xs = dot(Vector(xs.n), -wIn); // direction xs -> node
    if (cosTheta_xs <= 0.0f)
        return 0.0f; // xs is looking away from node

    // B. Node Orientation (Using SGGX or Average Normal)
    // SGGX is superior for BVH nodes as it represents the aggregate normal distribution.
    // It naturally handles "backfacing" for volumetric clusters better than a single mean normal.
    Float sggxIn = sggxCrossSection(nodeInfo.normalDistribution.m2, wIn);
    Float sggxOut = sggxCrossSection(nodeInfo.normalDistribution.m2, wOut);

    Float sggxTerm = math::lerp(sggxCrossSectionScale, 1.f, sggxIn * sggxOut);

    // If SGGX is zero (node silhouette is zero from this angle), return 0
    if (sggxTerm <= 0.0f)
        return 0.0f;

    // C. Mean-normal alignment: prefer nodes whose mean normal faces the
    // emitter direction. This preserves sign information lost by SGGX's
    // second-moment-only representation (outer product is sign-insensitive).
    // Vector meanDir = nodeInfo.normalDistribution.meanDirection();
    // Float meanOut = dot(meanDir, wOut);
    // if (meanOut <= 0.0f) {
    //    // Node mean faces away from emitter -> very low importance
    //    return 0.0f;
    // }

    // 6. Solid Angle
    // Using solid angle is excellent as it handles the 1/r^2 singularity automatically.
    Float omega_xs = AABBSphSolidAngle(nodeBounds, xs.p);
    Float omega_xe = AABBSphSolidAngle(nodeBounds, xe.p);

    Float solidAngleTerm = (omega_xs * omega_xe) / nodeInfo.surfaceArea; // Normalize by area to keep units consistent
    solidAngleTerm = math::lerp(solidAngleScale, 1.f, solidAngleTerm);   // Optional scaling to control how much we care about solid angle

    // 7. Compute Final Importance
    // Heuristic: Albedo * (Visible Area Fraction) * SolidAngles

    Float baseImportance = nodeInfo.diffuseAlbedo;

    // Combine terms.
    // We use the product of solid angles to prioritize nodes that are large/close to BOTH endpoints.
    // We multiply by SGGX cross sections to prioritize nodes that are "facing" the right way.
    Float geometricTerm = sggxTerm * solidAngleTerm; 
    // ^ normalize by area to keep units consistent (otherwise importance grows with area^2 bc of solid angles)

    // Include the receiver cosine (Lambertian receiver at xs)
    geometricTerm *= math::lerp(cosinePenaltyScale, 1.f, cosTheta_xs);
    // Include mean-normal alignment factor (from above)
    // geometricTerm *= meanOut; TODO(jorge): together with lines C. don't know if this is correct

    Float importance = baseImportance * geometricTerm;

    // 8. Variance Penalty
    // Higher variance = light is scattered more = peak intensity might be lower
    // or geometry is more chaotic.
    const Float variance = nodeInfo.normalDistribution.variance();
    if (variance > 0.0f)
    {
        // Tunable parameter: controls how much we dislike high-variance nodes
        Float variancePenalty = 1.0f / (1.0f + variance * 2.0f);
        importance *= math::lerp(variancePenaltyScale, 1.f, variancePenalty);
    }

    return importance;
}

Float computeNodeImportanceGroundTruth(
    int nodeIndex,
    GeometryBVH *bvh,
    Scene *scene,
    const SurfaceSample &xs,
    const EmitterSample &xe,
    const BVHNodeInfo &nodeInfo,
    int numSamples,
    bool checkVisibility)
{
    // return 1.f;

    if (!scene || !nodeInfo.valid || nodeInfo.surfaceArea <= Epsilon || numSamples <= 0)
        return 0.0f;

    if (!bvh || !bvh->isBuilt())
        return 0.0f;

    Spectrum contributionSum(0.f);
    for (int sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx)
    {
        Intersection its;
        float pdf;
        bvh->sampleNodePrimitive(
            nodeIndex, 
            scene, xs.p,
            scene->getSampler()->next2D(),
            its, pdf
        );

        // Compute directions and cosines
        Vector xsToP = its.p - xs.p;
        const Float distXsP = xsToP.length();
        if (distXsP <= Epsilon)
            continue;
        xsToP /= distXsP;

        Vector pToXe = xe.p - its.p;
        const Float distPXe = pToXe.length();
        if (distPXe <= Epsilon)
            continue;
        pToXe /= distPXe;

        const Float cos_xs = std::max((Float)0.0f, dot(Vector(xs.n), xsToP));
        const Float cos_x = std::max((Float)0.0f, dot(Vector(its.shFrame.n), -xsToP));
        const Float cos_x_out = std::max((Float)0.0f, dot(Vector(its.shFrame.n), pToXe));
        // const Float cos_xe = std::max((Float)0.0f, dot(Vector(xe.n), -pToXe));
        if (cos_xs <= 0.0f || cos_x <= 0.0f || cos_x_out <= 0.0f)
            continue;

        if (checkVisibility)
        {
            Ray shadowRay1(xs.p + xsToP * Epsilon, xsToP, Epsilon,
                           std::max((Float)0.0f, distXsP - 2 * Epsilon), 0.0f);
            if (scene->rayIntersect(shadowRay1))
                continue;

            Ray shadowRay2(its.p + pToXe * Epsilon, pToXe, Epsilon,
                           std::max((Float)0.0f, distPXe - 2 * Epsilon), 0.0f);
            if (scene->rayIntersect(shadowRay2))
                continue;
        }

        /*
        // Fill intersection
        Intersection its;
        its.p = its.p;
        its.uv = its.uv;
        its.shape = mesh;
        its.instance = NULL;

        const Point *verts = mesh->getVertexPositions();
        Vector e1 = verts[tri.idx[1]] - verts[tri.idx[0]];
        Vector e2 = verts[tri.idx[2]] - verts[tri.idx[0]];
        its.geoFrame = Frame(normalize(cross(e1, e2)));
        its.shFrame = Frame(its.n);
        its.wi = its.toLocal(-xsToP);
        */

        // Fill BSDF record
        BSDFSamplingRecord bRec(its, its.toLocal(-xsToP), its.toLocal(pToXe));
        Spectrum f_p = its.getBSDF()->eval(bRec);

        const Float geom = (cos_xs * cos_x * cos_x_out) / (distXsP * distXsP * distPXe * distPXe);
        contributionSum += (f_p * geom)/pdf;
        // std::cout << "Sample " << sampleIdx << " p:" << its.p.toString() << ", f_p: " << f_p.toString() << ", geom: " << geom << ", contribution: " << (f_p * geom/pdf).toString() << std::endl;
    }

    float luminanceContribution = contributionSum.getLuminance();

    return (luminanceContribution / (Float)numSamples);
}

#endif

MTS_NAMESPACE_END
