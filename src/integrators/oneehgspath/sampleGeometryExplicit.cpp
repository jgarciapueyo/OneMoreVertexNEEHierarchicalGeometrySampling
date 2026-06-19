#include "sampleGeometryExplicit.h"
#include <mitsuba/core/pmf.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>

MTS_NAMESPACE_BEGIN

bool sampleGeometryExplicit(
    const Scene *scene,
    Sampler *sampler,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    Intersection &its_xp,
    Float &pdf,
    bool checkVisibility
) {
    // std::cout << "sampleGeometryExplicit - start\n";

    // 1. Initialize a discrete distribution over all triangles in the scene
    const std::vector<TriMesh *> &meshes = scene->getMeshes();
    
    size_t totalTriangles = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        totalTriangles += meshes[i]->getTriangleCount();
    }

    if (totalTriangles == 0)
        return false;

    DiscreteDistribution distr(totalTriangles);
    
    // Use a simple struct to keep track of which mesh/triangle each index refers to
    struct TriangleID {
        uint32_t meshIndex;
        uint32_t triangleIndex;
    };
    std::vector<TriangleID> triangleIDs;
    triangleIDs.reserve(totalTriangles);

    // 2. Compute potential contribution of each triangle and fill the distribution
    for (uint32_t meshIdx = 0; meshIdx < (uint32_t) meshes.size(); ++meshIdx) {
        const TriMesh *mesh = meshes[meshIdx];
        const Triangle *triangles = mesh->getTriangles();
        
        for (uint32_t triIdx = 0; triIdx < (uint32_t) mesh->getTriangleCount(); ++triIdx) {
            // 2.1. Compute the potential contribution of this triangle to the measurement point
            // 2.1.1. Sample a point on the triangle to estimate the contribution
            const Triangle &tri = triangles[triIdx];

            PositionSamplingRecord pRec_xp_temp;
            pRec_xp_temp.p = tri.sample(
                mesh->getVertexPositions(),
                mesh->getVertexNormals(),
                mesh->getVertexTexcoords(),
                pRec_xp_temp.n,
                pRec_xp_temp.uv,
                sampler->next2D()
            );

            Intersection its_xp_temp;
            its_xp_temp.p = pRec_xp_temp.p;
            its_xp_temp.geoFrame = Frame(pRec_xp_temp.n);
            its_xp_temp.shFrame = its_xp_temp.geoFrame;
            its_xp_temp.uv = pRec_xp_temp.uv;
            its_xp_temp.shape = mesh;
            its_xp_temp.instance = NULL;
            its_xp_temp.time = 0.0f; // TODO: handle time if needed
            
            // 2.1.2. Compute the contribution of this point to the measurement point
            Spectrum contribution = evalLength2Contribution(scene, its_xs, its_xp_temp, pRec_xe, checkVisibility);
            distr.append(contribution.getLuminance());
            triangleIDs.push_back({meshIdx, triIdx});

            // std::cout << "sampleGeometryExplicit - mesh: " << meshIdx << " triangle: " << triIdx << " contribution: " << contribution.max() << "\n";
        }
    }

    // std::cout << "sampleGeometryExplicit - distribution filled\n";

    // Normalize to create the PDF
    if (distr.normalize() == 0)
        return false;

    // std::cout << "sampleGeometryExplicit - total contribution: " << distr.getSum() << std::endl;

    // 3. Sample a point according to the distribution
    // 3.1. Sample a triangle index
    Float triangleProb;
    size_t sampledIdx = distr.sample(sampler->next1D(), triangleProb);
    const TriangleID &tid = triangleIDs[sampledIdx];

    // 3.2. Sample a point on the chosen triangle
    const TriMesh *mesh = meshes[tid.meshIndex];
    const Triangle &tri = mesh->getTriangles()[tid.triangleIndex];

    PositionSamplingRecord pRec_xp;
    pRec_xp.p = tri.sample(
        mesh->getVertexPositions(),
        mesh->getVertexNormals(),
        mesh->getVertexTexcoords(),
        pRec_xp.n,
        pRec_xp.uv,
        sampler->next2D()
    );

    Float area = tri.surfaceArea(mesh->getVertexPositions());
    if (area <= 0)
        return false;

    // 3.3. Fill Intersection record
    its_xp.p = pRec_xp.p;
    its_xp.geoFrame = Frame(pRec_xp.n);
    its_xp.shFrame = its_xp.geoFrame;
    its_xp.uv = pRec_xp.uv;
    its_xp.shape = mesh;
    its_xp.instance = NULL;
    its_xp.time = 0.0f; // TODO: handle time if needed
    
    // Final PDF: selection_prob * uniform_area_sampling_pdf
    pdf = triangleProb * (1.0f / area);

    return true;
}

MTS_NAMESPACE_END
