#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <algorithm>
#include <fstream>
#include <mitsuba/core/matrix.h>


#include <mitsuba/core/vmf.h>

MTS_NAMESPACE_BEGIN

GeometryBVH::GeometryBVH(uint32_t maxLeafSize, GeometryBVHSamplingMode samplingMode)
    : ConfigurableObject(Properties()), m_maxLeafSize(maxLeafSize), m_samplingMode(samplingMode) {
    if (m_maxLeafSize < 1) {
        m_maxLeafSize = 1;
    }
}

GeometryBVH::GeometryBVH(const Properties &props)
    : ConfigurableObject(props),
      m_maxLeafSize(props.getInteger("maxLeafSize", 4)),
      m_samplingMode(GeometryBVHSamplingMode::Primitive)
{
    if (m_maxLeafSize < 1) {
        m_maxLeafSize = 1;
    }
    
    std::string samplingModeStr = props.getString("samplingMode", "Primitive");
    if (samplingModeStr == "SphericalAABB") {
        m_samplingMode = GeometryBVHSamplingMode::SphericalAABB;
    } else if (samplingModeStr == "Primitive") {
        m_samplingMode = GeometryBVHSamplingMode::Primitive;
    } else {
        SLog(EWarn, "GeometryBVH: Unknown sampling mode '%s', defaulting to 'Primitive'", 
            samplingModeStr.c_str());
    }

    // Read BVH importance parameters (default: 1.0 = full importance, 0.0 = ignore): 
    m_sggxCrossSectionScale = props.getFloat("sggxCrossSectionScale", 1.0f);
    m_solidAngleScale = props.getFloat("solidAngleScale", 0.0f);
    m_variancePenaltyScale = props.getFloat("variancePenaltyScale", 0.0f);
    m_cosinePenaltyScale = props.getFloat("cosinePenaltyScale", 1.0f);

    m_sggxCrossSectionScale = math::clamp(m_sggxCrossSectionScale, 0.f, 1.f);
    m_solidAngleScale = math::clamp(m_solidAngleScale, 0.f, 1.f);
    m_variancePenaltyScale = math::clamp(m_variancePenaltyScale, 0.f, 1.f);
    m_cosinePenaltyScale = math::clamp(m_cosinePenaltyScale, 0.f, 1.f);

    if (m_sggxCrossSectionScale < .99f || m_solidAngleScale < .99f || m_variancePenaltyScale < .99f || m_cosinePenaltyScale < .99f) {
        SLog(EInfo, "GeometryBVH: Configured importance scales: sggxCrossSectionScale=%.2f, solidAngleScale=%.2f, variancePenaltyScale=%.2f, cosinePenaltyScale=%.2f",
            m_sggxCrossSectionScale, m_solidAngleScale, m_variancePenaltyScale, m_cosinePenaltyScale);
    }

    // more importance params:
    m_probmult = props.getFloat("importanceMultiplier", 1.0f);
    m_defensive_pdf = props.getFloat("defensivePDF", 1e-2f);
    m_probmult = std::max(0.f, m_probmult);

    if (m_probmult != 1.0f || m_defensive_pdf != 1e-2f) {
        SLog(EInfo, "GeometryBVH: Configured importance parameters: importanceMultiplier=%.2f, defensivePDF=%.4f",
            m_probmult, m_defensive_pdf);
    }
}

GeometryBVH::~GeometryBVH() {
    // vectors clean up automatically
}

GeometryBVH::GeometryBVH(Stream *stream, InstanceManager *manager)
    : ConfigurableObject(stream, manager) 
{
    m_maxLeafSize = stream->readUInt();
    uint32_t samplingModeInt = stream->readUInt();
    m_samplingMode = static_cast<GeometryBVHSamplingMode>(samplingModeInt);
    
    // Note: BVH data (nodes, primitives, etc.) are not serialized here.
    // They should be rebuilt using buildBVH() after deserialization.
}

void GeometryBVH::buildBVH(Scene *scene) {
    if (!scene) {
        SLog(EWarn, "GeometryBVH::build: scene is NULL");
        return;
    }

    m_scene = scene; // Get root scene in case this is a sub-scene
    
    // Step 1: Collect all triangles from all meshes
    std::vector<BVHPrimitive> buildPrims;
    
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    
    // Count total triangles for reserve
    size_t totalTriangles = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        totalTriangles += meshes[i]->getTriangleCount();
    }
    buildPrims.reserve(totalTriangles);
    
    SLog(EInfo, "GeometryBVH: Collecting %zu triangles from %zu meshes", 
         totalTriangles, meshes.size());
    
    for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx) {
        const TriMesh *mesh = meshes[meshIdx];
        const Point *positions = mesh->getVertexPositions();
        const Triangle *triangles = mesh->getTriangles();
        const BSDF *bsdf = mesh->getBSDF();
        
        for (size_t triIdx = 0; triIdx < mesh->getTriangleCount(); ++triIdx) {
            // Compute triangle AABB
            AABB bounds = triangles[triIdx].getAABB(positions);
            Point centroid = bounds.getCenter();
            
            buildPrims.push_back(BVHPrimitive(
                static_cast<uint32_t>(meshIdx),
                static_cast<uint32_t>(triIdx),
                bounds,
                centroid,
                bsdf
            ));
        }
    }
    
    if (buildPrims.empty()) {
        SLog(EWarn, "GeometryBVH::build: No triangles in scene");
        return;
    }
    
    // Step 2: Build tree recursively
    std::vector<BVHPrimitive> orderedPrims;
    orderedPrims.reserve(buildPrims.size());
    
    m_nodes.clear();
    m_nodes.reserve(2 * buildPrims.size()); // Upper bound on node count
    
    buildBVHRecursive(buildPrims, 0, buildPrims.size(), orderedPrims);
    
    // Step 3: Store ordered primitives
    m_primitives = std::move(orderedPrims);
    
    // Step 4: Build triangle-to-primitive mapping
    m_meshOffsets.resize(meshes.size());
    size_t currentOffset = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        m_meshOffsets[i] = static_cast<uint32_t>(currentOffset);
        currentOffset += meshes[i]->getTriangleCount();
    }
    m_triangleToPrim.assign(totalTriangles, 0xFFFFFFFF);
    for (size_t i = 0; i < m_primitives.size(); ++i) {
        const BVHPrimitive &prim = m_primitives[i];
        m_triangleToPrim[m_meshOffsets[prim.meshIndex] + prim.triangleIndex] = static_cast<uint32_t>(i);
    }
    
    // Step 5: Allocate nodeInfo array (filled later by buildAggregates)
    m_nodeInfos.resize(m_nodes.size());
    
    SLog(EInfo, "GeometryBVH built: %zu nodes, %zu primitives, maxLeafSize=%u", 
         m_nodes.size(), m_primitives.size(), m_maxLeafSize);

}

size_t GeometryBVH::buildBVHRecursive(
    std::vector<BVHPrimitive> &buildPrims,
    size_t start, size_t end,
    std::vector<BVHPrimitive> &orderedPrims)
{
    // Compute bounds of all primitives in range [start, end)
    AABB bounds;
    for (size_t i = start; i < end; ++i) {
        bounds.expandBy(buildPrims[i].bounds);
    }
    
    size_t nPrims = end - start;

    
    // Create leaf if few enough primitives
    if (nPrims <= m_maxLeafSize) {
        size_t primOffset = orderedPrims.size();
        for (size_t i = start; i < end; ++i) {
            orderedPrims.push_back(buildPrims[i]);
        }
        
        size_t nodeIndex = m_nodes.size();
        BVHNode node;
        node.bounds = bounds;
        node.primitivesOffset = static_cast<uint32_t>(primOffset);
        node.nSubtreePrimitives = static_cast<uint32_t>(nPrims);
        node.nLeafPrimitives = static_cast<uint16_t>(nPrims);
        node.splitAxis = 0;
        m_nodes.push_back(node);
        return nodeIndex;
    }
    
    // Compute centroid bounds for split decision
    // TODO(jorge): should the bvh be splitted like this or use the bbox of the primitives? or SAH?
    AABB centroidBounds;
    for (size_t i = start; i < end; ++i) {
        centroidBounds.expandBy(buildPrims[i].centroid);
    }
    
    // Choose split axis (longest dimension of centroid bounds)
    Vector diag = centroidBounds.max - centroidBounds.min;
    int splitAxis = 0;
    if (diag.y > diag.x) splitAxis = 1;
    if (diag.z > diag[splitAxis]) splitAxis = 2;
    
    // Check for degenerate case: all centroids at same position
    if (centroidBounds.max[splitAxis] == centroidBounds.min[splitAxis]) {
        // Can't split meaningfully - create leaf even if above maxLeafSize
        size_t primOffset = orderedPrims.size();
        for (size_t i = start; i < end; ++i) {
            orderedPrims.push_back(buildPrims[i]);
        }
        
        size_t nodeIndex = m_nodes.size();
        BVHNode node;
        node.bounds = bounds;
        node.primitivesOffset = static_cast<uint32_t>(primOffset);
        node.nSubtreePrimitives = static_cast<uint32_t>(nPrims);
        node.nLeafPrimitives = static_cast<uint16_t>(nPrims);
        node.splitAxis = static_cast<uint8_t>(splitAxis);
        m_nodes.push_back(node);
        return nodeIndex;
    }
    
    // Object median split: partition primitives at the median centroid
    size_t mid = (start + end) / 2;
    std::nth_element(
        buildPrims.begin() + start,
        buildPrims.begin() + mid,
        buildPrims.begin() + end,
        [splitAxis](const BVHPrimitive &a, const BVHPrimitive &b) {
            return a.centroid[splitAxis] < b.centroid[splitAxis];
        }
    );
    
    // Create interior node
    size_t nodeIndex = m_nodes.size();
    BVHNode node;
    node.bounds = bounds;
    node.nLeafPrimitives = 0;  // Interior node marker
    node.nSubtreePrimitives = static_cast<uint32_t>(nPrims);
    node.splitAxis = static_cast<uint8_t>(splitAxis);
    m_nodes.push_back(node);
    
    // Build left child (will be at nodeIndex + 1 due to depth-first order)
    buildBVHRecursive(buildPrims, start, mid, orderedPrims);
    
    // Build right child and store its index
    m_nodes[nodeIndex].secondChildOffset = static_cast<uint32_t>(
        buildBVHRecursive(buildPrims, mid, end, orderedPrims)
    );
    
    return nodeIndex;
}

void GeometryBVH::buildAggregates(const Scene *scene) {
    if (m_nodes.empty()) {
        SLog(EWarn, "GeometryBVH::buildAggregates: BVH not built yet");
        return;
    }
    
    if (!scene) {
        SLog(EWarn, "GeometryBVH::buildAggregates: scene is NULL");
        return;
    }
    
    // Compute aggregates bottom-up via post-order traversal
    buildAggregatesRecursive(0, scene);
    
#if DUMP_DEBUG_TO_CSV == 1
    print();
#endif
    SLog(EInfo, "GeometryBVH aggregates computed for %zu nodes", m_nodes.size());
}

void GeometryBVH::buildAggregatesRecursive(size_t nodeIndex, const Scene *scene) {
    const BVHNode &node = m_nodes[nodeIndex];
    BVHNodeInfo &info = m_nodeInfos[nodeIndex];
    
    if (node.isLeaf()) {
        // Leaf: accumulate from actual triangles
        info = BVHNodeInfo(); // Reset
        
        const std::vector<TriMesh*> &meshes = scene->getMeshes();
        
        for (uint32_t i = 0; i < node.nLeafPrimitives; ++i) {
            const BVHPrimitive &prim = m_primitives[node.primitivesOffset + i];
            const TriMesh *mesh = meshes[prim.meshIndex];
            
            // Compute triangle info and accumulate directly
            BVHNodeInfo triInfo = computeTriangleInfo(
                mesh, prim.triangleIndex, scene);
            
            info.accumulate(triInfo);
        }
    } else {
        // Interior: first compute children, then combine
        size_t leftChild = nodeIndex + 1;
        size_t rightChild = node.secondChildOffset;
        
        buildAggregatesRecursive(leftChild, scene);
        buildAggregatesRecursive(rightChild, scene);
        
        // Combine children aggregates
        info = BVHNodeInfo(); // Reset
        info.accumulate(m_nodeInfos[leftChild]);
        info.accumulate(m_nodeInfos[rightChild]);
    }
}

BVHNodeInfo computeTriangleInfo(
    const Shape *shape,
    size_t triangleIndex,
    const Scene *scene)
{
    BVHNodeInfo info;
    
    // Only handle TriMesh for now
    const TriMesh *mesh = dynamic_cast<const TriMesh*>(shape);
    if (!mesh) {
        SLog(EWarn, "computeTriangleInfo: shape is not a TriMesh");
        return info;
    }
    
    // Get triangle vertices
    const Point *vertexPositions = mesh->getVertexPositions();
    const Triangle *triangles = mesh->getTriangles();
    
    if (triangleIndex >= mesh->getTriangleCount()) {
        SLog(EWarn, "computeTriangleInfo: triangleIndex out of bounds");
        return info;
    }
    
    const Triangle &tri = triangles[triangleIndex];
    const Point &p0 = vertexPositions[tri.idx[0]];
    const Point &p1 = vertexPositions[tri.idx[1]];
    const Point &p2 = vertexPositions[tri.idx[2]];
    
    // Compute area: 0.5 * ||cross(e1, e2)||
    Vector e1 = p1 - p0;
    Vector e2 = p2 - p0;
    Vector crossProd = cross(e1, e2);
    info.surfaceArea = 0.5f * crossProd.length();
    
    // Compute (unit) geometric normal
    Vector gn;
    if (info.surfaceArea > 0.0f) {
        gn = normalize(crossProd);
    } else {
        gn = Vector(0, 0, 1);
    }

    // Initialize SGGX moments from this single-triangle normal
    info.normalDistribution.setFromNormal(gn);
    
    // Compute center of mass (centroid of triangle)
    info.centerOfMass = (p0 + p1 + p2) * (1.0f / 3.0f);
    
    // Get diffuse albedo from BSDF
    // Sample at triangle center for simplicity
    Point center = info.centerOfMass;
    
    const BSDF *bsdf = shape->getBSDF();
    if (bsdf) {
        // Create a dummy intersection for BSDF evaluation
        Intersection its;
        its.p = center;
        its.geoFrame = Frame(Normal(info.normalDistribution.meanDirection()));
        its.shFrame = its.geoFrame;
        its.wi = its.toLocal(Vector(0, 0, 1));
        its.shape = shape;
        
        // Get diffuse reflectance (this is an approximation)
        Spectrum albedo = bsdf->getDiffuseReflectance(its);
        
        // Convert spectrum to single float (luminance)
        info.diffuseAlbedo = albedo.getLuminance();
    } else {
        // Default albedo if no BSDF
        info.diffuseAlbedo = 0.5f;
    }
    
    // For a single triangle, second-moment variance proxy is near zero
    // (since ||E[n]|| ~= 1 and E[||n||^2] ~= 1)
    
    // Mark as valid
    info.valid = true;
    
    return info;
}

MTS_IMPLEMENT_CLASS_S(GeometryBVH, false, ConfigurableObject)

// Note: plugin export lives in src/geometrybvh/geometrybvh.cpp

MTS_NAMESPACE_END