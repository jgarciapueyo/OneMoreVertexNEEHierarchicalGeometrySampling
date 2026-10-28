#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/scene.h>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <mitsuba/core/matrix.h>

#include <mitsuba/core/vmf.h>

#if defined(MTS_OPENMP)
#include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

SamplingBVH::SamplingBVH(const Properties &props)
    : ConfigurableObject(props) {
    m_maxDepth = (uint32_t) std::max(1, props.getInteger("maxDepth", 16));

    std::string samplingModeStr = props.getString("samplingMode", "Primitive");
    if (samplingModeStr == "Primitive")
        m_samplingMode = SamplingBVHSamplingMode::Primitive;
    else if (samplingModeStr == "SphericalAABB")
        m_samplingMode = SamplingBVHSamplingMode::SphericalAABB;
    else if (samplingModeStr == "SphericalAABB_anyNode")
        m_samplingMode = SamplingBVHSamplingMode::SphericalAABB_anyNode;
    else
        Log(EError, "Unknown samplingMode \"%s\" (valid: Primitive, SphericalAABB, "
            "SphericalAABB_anyNode)", samplingModeStr.c_str());

    m_enableLightcuts = props.getBoolean("enableLightcuts", false);
    m_lightcutThreshold = props.getFloat("lightcutThreshold", 2.f);
    m_gaussianSAThreshold = props.getFloat("gaussianSAThreshold", (Float) (M_PI * 1e-2));
    m_angularSAHWeight = std::max((Float) 0.0f, props.getFloat("angularSAHWeight", 0.0f));
    m_angularSAHKappaScale = std::max((Float) 1e-4f, props.getFloat("angularSAHKappaScale", 10.0f));
    m_logAngularSAHStats = props.getBoolean("logAngularSAHStats", true);

    /* Node importance weight */
    m_uniformImportance = props.getBoolean("uniformImportance", false);
    m_specularImportance = props.getBoolean("specularImportance", true);
    std::string sigmaPoints = props.getString("sigmaPoints", "precomputed");
    if (sigmaPoints == "precomputed")
        m_sigmaPoints = EPrecomputedSigmaPoints;
    else if (sigmaPoints == "perCall")
        m_sigmaPoints = EPerCallSigmaPoints;
    else if (sigmaPoints == "directionAtMean")
        m_sigmaPoints = EDirectionAtMean;
    else
        Log(EError, "Unknown sigmaPoints \"%s\" (valid: precomputed, perCall, "
            "directionAtMean)", sigmaPoints.c_str());

    m_probmult = std::max((Float) 0.f, props.getFloat("importanceMultiplier", 1.0f));
    m_defensive_pdf = props.getFloat("defensivePDF", 1e-16f);

    /* Ablation of the factors of the node importance weight (Fig. 5) */
    auto enableAblation = [&](const char *propertyName, Ablation flag) {
        if (props.getBoolean(propertyName, false))
            m_ablationMask |= static_cast<uint32_t>(flag);
    };
    enableAblation("ablateVMF", AblateVMF);
    enableAblation("ablateGaussian", AblateGaussian);
    enableAblation("ablateCosineXs", AblateCosineXs);
    enableAblation("ablateCosineXe", AblateCosineXe);
    enableAblation("ablateDistanceXs", AblateDistanceXs);
    enableAblation("ablateDistanceXe", AblateDistanceXe);
    enableAblation("ablateAlbedo", AblateAlbedo);

    /* Debug CSV used for the importance visualization of Fig. 4 */
    m_outputDebugCSV = props.getBoolean("outputDebugCSV", false);
    m_csvEnabledLevels = (uint32_t) props.getLong("csvEnabledLevels", 0xFFFFFFFFLL);
    m_debugQueryPoint = props.getPoint("debugQueryPoint", m_debugQueryPoint);
    m_debugQueryShape = props.getInteger("debugQueryShape", m_debugQueryShape);
}

SamplingBVH::SamplingBVH(Stream *stream, InstanceManager *manager)
    : ConfigurableObject(stream, manager) {
    m_maxDepth = stream->readUInt();
    m_samplingMode = static_cast<SamplingBVHSamplingMode>(stream->readUInt());
    // Note: the remaining configuration and the BVH data are not serialized;
    // the BVH has to be rebuilt with buildBVH() after deserialization.
}

SamplingBVH::~SamplingBVH() { }

std::string SamplingBVH::toString() const {
    const char *samplingMode =
        m_samplingMode == SamplingBVHSamplingMode::Primitive ? "Primitive" :
        m_samplingMode == SamplingBVHSamplingMode::SphericalAABB ? "SphericalAABB" :
        "SphericalAABB_anyNode";
    const char *sigmaPoints =
        m_sigmaPoints == EPrecomputedSigmaPoints ? "precomputed" :
        m_sigmaPoints == EPerCallSigmaPoints ? "perCall" : "directionAtMean";

    std::ostringstream oss;
    oss << "SamplingBVH[" << endl
        << "  maxDepth = " << m_maxDepth << "," << endl
        << "  samplingMode = " << samplingMode << "," << endl
        << "  uniformImportance = " << (m_uniformImportance ? "true" : "false") << "," << endl
        << "  specularImportance = " << (m_specularImportance ? "true" : "false") << "," << endl
        << "  sigmaPoints = " << sigmaPoints << "," << endl
        << "  gaussianSAThreshold = " << m_gaussianSAThreshold << "," << endl
        << "  defensivePDF = " << m_defensive_pdf << "," << endl
        << "  importanceMultiplier = " << m_probmult << "," << endl
        << "  ablationMask = " << m_ablationMask << " [VMF=" << ((m_ablationMask & AblateVMF) != 0)
        << ", Gaussian=" << ((m_ablationMask & AblateGaussian) != 0)
        << ", CosineXs=" << ((m_ablationMask & AblateCosineXs) != 0)
        << ", CosineXe=" << ((m_ablationMask & AblateCosineXe) != 0)
        << ", DistanceXs=" << ((m_ablationMask & AblateDistanceXs) != 0)
        << ", DistanceXe=" << ((m_ablationMask & AblateDistanceXe) != 0)
        << ", Albedo=" << ((m_ablationMask & AblateAlbedo) != 0) << "]," << endl
        << "  enableLightcuts = " << (m_enableLightcuts ? "true" : "false")
        << " (threshold " << m_lightcutThreshold << ")," << endl
        << "  angularSAHWeight = " << m_angularSAHWeight
        << " (kappa scale " << m_angularSAHKappaScale << ")," << endl
        << "  outputDebugCSV = " << (m_outputDebugCSV ? "true" : "false") << endl
        << "]";
    return oss.str();
}

size_t SamplingBVH::getMemoryUsageBytes() const {
    size_t totalBytes = sizeof(SamplingBVH);
    totalBytes += m_nodes.capacity() * sizeof(BVHNode);
    totalBytes += m_primitives.capacity() * sizeof(BVHPrimitive);
    totalBytes += m_nodeInfos.capacity() * sizeof(BVHNodeInfo);
    totalBytes += m_leafAliasTableOffsets.capacity() * sizeof(uint32_t);
    totalBytes += m_leafAliasProb.capacity() * sizeof(Float);
    totalBytes += m_leafAliasAlias.capacity() * sizeof(uint32_t);
    totalBytes += m_meshOffsets.capacity() * sizeof(uint32_t);
    totalBytes += m_triangleToPrim.capacity() * sizeof(uint32_t);
    return totalBytes;
}

uint32_t SamplingBVH::getMeshIndex(const Shape *shape) const {
    std::map<const Shape *, uint32_t>::const_iterator it = m_meshIndices.find(shape);
    return it != m_meshIndices.end() ? it->second : 0xFFFFFFFFu;
}

void SamplingBVH::buildBVH(Scene *scene) {
    if (!scene) {
        SLog(EWarn, "SamplingBVH::build: scene is NULL");
        return;
    }
    Log(EInfo, "Building %s", toString().c_str());
    m_scene = scene;
    
    // Step 1: Collect all triangles from all meshes
    std::vector<BVHBuildPrimitive> buildPrims;
    
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    m_meshIndices.clear();
    for (size_t i = 0; i < meshes.size(); ++i)
        m_meshIndices[meshes[i]] = (uint32_t) i;

    // Count total triangles for reserve
    size_t totalTriangles = 0;
    for (size_t i = 0; i < meshes.size(); ++i) {
        totalTriangles += meshes[i]->getTriangleCount();
    }
    buildPrims.reserve(totalTriangles);
    size_t skippedDegenerateTriangles = 0;
    std::vector<std::vector<BVHBuildPrimitive> > meshPrims(meshes.size());
    std::vector<size_t> meshSkipped(meshes.size(), 0);
    
    SLog(EInfo, "SamplingBVH: Collecting %zu triangles from %zu meshes", 
         totalTriangles, meshes.size());
    
    auto collectMeshPrimitives = [&](size_t meshIdx) {
        const TriMesh *mesh = meshes[meshIdx];
        const Point *positions = mesh->getVertexPositions();
        const Triangle *triangles = mesh->getTriangles();
        std::vector<BVHBuildPrimitive> &localPrims = meshPrims[meshIdx];
        size_t localSkipped = 0;

        localPrims.reserve(mesh->getTriangleCount());
        
        const bool needAngularSAH = m_angularSAHWeight > 0.0f;
        for (size_t triIdx = 0; triIdx < mesh->getTriangleCount(); ++triIdx) {
            const Float area = triangles[triIdx].surfaceArea(positions);
            if (!std::isfinite(area) || area <= 0.0f) {
                ++localSkipped;
                continue;
            }

            const Triangle &tri = triangles[triIdx];
            Vector areaWeightedNormal(0.0f);
            if (needAngularSAH) {
                const Point &p0 = positions[tri.idx[0]];
                const Point &p1 = positions[tri.idx[1]];
                const Point &p2 = positions[tri.idx[2]];
                areaWeightedNormal = cross(p1 - p0, p2 - p0) * 0.5f;
            }

            // Compute triangle AABB
            AABB bounds = triangles[triIdx].getAABB(positions);
            Point centroid = bounds.getCenter();
            
            localPrims.push_back(BVHBuildPrimitive(
                static_cast<uint32_t>(meshIdx),
                static_cast<uint32_t>(triIdx),
                bounds,
                centroid,
                area,
                areaWeightedNormal
            ));
        }

        meshSkipped[meshIdx] = localSkipped;
    };

#if defined(MTS_OPENMP)
    bool parallelCollectionFailed = false;
    std::string parallelCollectionError;
    #pragma omp parallel for schedule(static)
    for (int meshIdx = 0; meshIdx < (int) meshes.size(); ++meshIdx) {
        try {
            collectMeshPrimitives((size_t) meshIdx);
        } catch (const std::exception &e) {
            #pragma omp critical
            {
                if (!parallelCollectionFailed) {
                    parallelCollectionFailed = true;
                    parallelCollectionError = e.what();
                }
            }
        } catch (...) {
            #pragma omp critical
            {
                if (!parallelCollectionFailed) {
                    parallelCollectionFailed = true;
                    parallelCollectionError = "unknown exception";
                }
            }
        }
    }

    if (parallelCollectionFailed) {
        SLog(EError, "SamplingBVH: parallel triangle collection failed: %s",
             parallelCollectionError.c_str());
        return;
    }
#else
    for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx)
        collectMeshPrimitives(meshIdx);
#endif

    // Merge in mesh order to keep build determinism stable across runs.
    for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx) {
        skippedDegenerateTriangles += meshSkipped[meshIdx];
        const std::vector<BVHBuildPrimitive> &localPrims = meshPrims[meshIdx];
        buildPrims.insert(buildPrims.end(), localPrims.begin(), localPrims.end());
    }

    if (skippedDegenerateTriangles > 0) {
        SLog(EWarn, "SamplingBVH: skipped %zu degenerate/non-finite-area triangles during build",
             skippedDegenerateTriangles);
    }

    
    if (buildPrims.empty()) {
        SLog(EWarn, "SamplingBVH::build: No valid triangles in scene after filtering degenerate triangles");
        return;
    }
    
    // Step 2: Build tree recursively
    std::vector<BVHPrimitive> orderedPrims;
    std::vector<Float> orderedAreas;
    orderedPrims.reserve(buildPrims.size());
    orderedAreas.reserve(buildPrims.size());

    m_buildAngularPenaltySum = 0.0;
    m_buildWeightedAngularPenaltySum = 0.0;
    m_buildAngularSplitCount = 0;
    
    m_nodes.clear();
    // Upper bound on node count: min(2*N-1, 2^(maxDepth+1)-1).
    // This avoids huge over-reservation when maxDepth is small.
    auto computeNodeReserve = [&](size_t primCount) -> size_t {
        if (primCount == 0)
            return 0;

        const size_t maxSize = std::numeric_limits<size_t>::max();
        const size_t maxNodesByPrim = (primCount > (maxSize - 1) / 2)
            ? maxSize
            : (2 * primCount - 1);

        const uint32_t maxShift = static_cast<uint32_t>(sizeof(size_t) * 8 - 1);
        const size_t maxNodesByDepth = (m_maxDepth >= maxShift)
            ? maxSize
            : ((static_cast<size_t>(1) << (m_maxDepth + 1)) - 1);

        return std::min(maxNodesByPrim, maxNodesByDepth);
    };

    m_nodes.reserve(computeNodeReserve(buildPrims.size()));
    
    buildBVHRecursiveSAH(buildPrims, 0, buildPrims.size(), orderedPrims, orderedAreas, 0);

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

    // Step 6: Precompute per-leaf alias tables for O(1) primitive sampling by area
    buildLeafSamplingTables(orderedAreas);
    
    SLog(EInfo, "SamplingBVH built: %zu nodes, %zu primitives, maxDepth=%u", 
         m_nodes.size(), m_primitives.size(), m_maxDepth);

    if (m_logAngularSAHStats && m_angularSAHWeight > 0.0f) {
        if (m_buildAngularSplitCount > 0) {
            const double invCount = 1.0 / (double) m_buildAngularSplitCount;
            const double avgAngularPenalty = m_buildAngularPenaltySum * invCount;
            const double avgWeightedAngularPenalty = m_buildWeightedAngularPenaltySum * invCount;
            SLog(EInfo,
                 "SamplingBVH angular SAH stats: selectedSplits=%llu, avgAngularPenalty=%.6f, avgWeightedAngularPenalty=%.6f, angularSAHWeight=%.3f, angularSAHKappaScale=%.3f",
                 (unsigned long long) m_buildAngularSplitCount,
                 avgAngularPenalty,
                 avgWeightedAngularPenalty,
                 m_angularSAHWeight,
                 m_angularSAHKappaScale);
        } else {
            SLog(EInfo,
                 "SamplingBVH angular SAH stats: no selected splits contributed (selectedSplits=0, angularSAHWeight=%.3f, angularSAHKappaScale=%.3f)",
                 m_angularSAHWeight,
                 m_angularSAHKappaScale);
        }
    }

}

void SamplingBVH::buildLeafSamplingTables(const std::vector<Float> &primitiveAreas) {
    m_leafAliasTableOffsets.assign(m_nodes.size(), 0xFFFFFFFFu);
    m_leafAliasProb.clear();
    m_leafAliasAlias.clear();

    for (size_t nodeIdx = 0; nodeIdx < m_nodes.size(); ++nodeIdx) {
        const BVHNode &node = m_nodes[nodeIdx];
        if (!node.isLeaf() || node.nLeafPrimitives == 0)
            continue;

        const uint32_t count = node.nLeafPrimitives;
        const uint32_t offset = static_cast<uint32_t>(m_leafAliasProb.size());
        m_leafAliasTableOffsets[nodeIdx] = offset;

        m_leafAliasProb.resize(offset + count);
        m_leafAliasAlias.resize(offset + count);

        if (count == 1) {
            m_leafAliasProb[offset] = 1.0f;
            m_leafAliasAlias[offset] = 0;
            continue;
        }

        std::vector<Float> scaled(count, 0.0f);
        std::vector<uint32_t> small;
        std::vector<uint32_t> large;
        small.reserve(count);
        large.reserve(count);

        Float totalArea = 0.0f;
        for (uint32_t i = 0; i < count; ++i) {
            const Float area = primitiveAreas[node.primitivesOffset + i];
            if (std::isfinite(area) && area > 0.0f)
                totalArea += area;
        }

        if (totalArea <= 0.0f) {
            for (uint32_t i = 0; i < count; ++i) {
                m_leafAliasProb[offset + i] = 1.0f;
                m_leafAliasAlias[offset + i] = i;
            }
            continue;
        }

        const Float invTotalArea = 1.0f / totalArea;
        for (uint32_t i = 0; i < count; ++i) {
            const Float rawArea = primitiveAreas[node.primitivesOffset + i];
            const Float area = (std::isfinite(rawArea) && rawArea > 0.0f)
                ? rawArea : 0.0f;

            scaled[i] = area * invTotalArea * (Float) count;
            if (scaled[i] < 1.0f)
                small.push_back(i);
            else
                large.push_back(i);
        }

        while (!small.empty() && !large.empty()) {
            const uint32_t s = small.back();
            small.pop_back();
            const uint32_t l = large.back();

            m_leafAliasProb[offset + s] = scaled[s];
            m_leafAliasAlias[offset + s] = l;

            scaled[l] = (scaled[l] + scaled[s]) - 1.0f;
            if (scaled[l] < 1.0f) {
                large.pop_back();
                small.push_back(l);
            }
        }

        while (!large.empty()) {
            const uint32_t l = large.back();
            large.pop_back();
            m_leafAliasProb[offset + l] = 1.0f;
            m_leafAliasAlias[offset + l] = l;
        }

        while (!small.empty()) {
            const uint32_t s = small.back();
            small.pop_back();
            m_leafAliasProb[offset + s] = 1.0f;
            m_leafAliasAlias[offset + s] = s;
        }
    }
    Log(EDebug, "Leaf alias tables: %zu entries (%s)", m_leafAliasProb.size(),
        memString(m_leafAliasProb.size() * (sizeof(Float) + sizeof(uint32_t))).c_str());
}

void SamplingBVH::buildAggregates(const Scene *scene) {
    if (m_nodes.empty()) {
        SLog(EWarn, "SamplingBVH::buildAggregates: BVH not built yet");
        return;
    }
    
    if (!scene) {
        SLog(EWarn, "SamplingBVH::buildAggregates: scene is NULL");
        return;
    }
    
    // Compute aggregates bottom-up via post-order traversal
    buildAggregatesRecursive(0, scene);
    
    if (m_outputDebugCSV)  {
        SLog(EInfo, "SamplingBVH: Generating debug CSV output for node aggregates...");
        print();
    }

    SLog(EInfo, "SamplingBVH aggregates computed for %zu nodes", m_nodes.size());
    SLog(EInfo, "SamplingBVH memory footprint (approx): %.3f MiB", (double) getMemoryUsageMB());
}

size_t SamplingBVH::buildBVHRecursiveSAH(
    std::vector<BVHBuildPrimitive> &buildPrims,
    size_t start, size_t end,
    std::vector<BVHPrimitive> &orderedPrims,
    std::vector<Float> &orderedAreas,
    uint32_t depth) 
{
    size_t nPrims = end - start;
    AABB bounds;
    AABB centroidBounds;
    for (size_t i = start; i < end; ++i) {
        bounds.expandBy(buildPrims[i].bounds);
        centroidBounds.expandBy(buildPrims[i].centroid);
    }

    // --- Base Case: Leaf Creation ---
    auto makeLeaf = [&]() -> size_t {
        size_t primOffset = orderedPrims.size();
        for (size_t i = start; i < end; ++i) {
            orderedPrims.push_back(BVHPrimitive(
                buildPrims[i].meshIndex,
                buildPrims[i].triangleIndex));
            orderedAreas.push_back(buildPrims[i].surfaceArea);
        }
        size_t nodeIndex = m_nodes.size();
        BVHNode node;
        node.bounds = bounds;
        node.primitivesOffset = static_cast<uint32_t>(primOffset);
        node.nSubtreePrimitives = static_cast<uint32_t>(nPrims);
        node.nLeafPrimitives = static_cast<uint32_t>(nPrims);
        node.splitAxis = 0;
        m_nodes.push_back(node);
        return nodeIndex;
    };

    if (nPrims <= 1 || depth >= m_maxDepth) {
        return makeLeaf();
    }

    // --- SAH Split Logic ---
    int bestAxis = -1;
    int bestSplitBin = -1;
    Float minCost = std::numeric_limits<Float>::infinity();
    Float bestAngularCost = 0.0f;
    
    const int nBins = 32;
    const Float traversalCost = 0.125f; // Relative cost of visiting a node
    const Float intersectionCost = 1.0f; // Relative cost of intersecting a primitive
    const bool useAngularSAH = (m_angularSAHWeight > 0.0f);

    auto computeAngularDispersion = [&](Float area, const Vector &normalSum) -> Float {
        if (!useAngularSAH || area <= 0.0f)
            return 0.0f;

        Float rBar = normalSum.length() / area;
        rBar = std::max((Float) 0.0f, std::min((Float) (1.0f - 1e-6f), rBar));
        Float kappa = VonMisesFisherDistr::forMeanLength(rBar);
        Float coherence = kappa / (kappa + m_angularSAHKappaScale);
        coherence = std::max((Float) 0.0f, std::min((Float) 1.0f, coherence));
        return 1.0f - coherence;
    };

    Vector diag = centroidBounds.max - centroidBounds.min;

    for (int axis = 0; axis < 3; ++axis) {
        if (diag[axis] <= 1e-6f) continue; // Skip flat axes

        struct Bin {
            AABB bounds;
            uint32_t count = 0;
            Float area = 0.0f;
            Vector normalSum = Vector(0.0f);
        } bins[nBins];

        // 1. Fill Bins
        for (size_t i = start; i < end; ++i) {
            Float offset = (buildPrims[i].centroid[axis] - centroidBounds.min[axis]) / diag[axis];
            int b = static_cast<int>(nBins * offset);
            if (b == nBins) b = nBins - 1;
            bins[b].count++;
            bins[b].bounds.expandBy(buildPrims[i].bounds);
            bins[b].area += buildPrims[i].surfaceArea;
            if (useAngularSAH)
                bins[b].normalSum += buildPrims[i].areaWeightedNormal;
        }

        // 2. Evaluate split costs between bins
        // We precompute the "Right" side bounds using a backward sweep
        AABB rightBounds[nBins];
        uint32_t rightCount[nBins];
        Float rightArea[nBins];
        Vector rightNormalSum[nBins];
        AABB currentRight;
        uint32_t currentRightCount = 0;
        Float currentRightArea = 0.0f;
        Vector currentRightNormalSum(0.0f);

        for (int i = nBins - 1; i >= 0; --i) {
            currentRight.expandBy(bins[i].bounds);
            currentRightCount += bins[i].count;
            currentRightArea += bins[i].area;
            if (useAngularSAH)
                currentRightNormalSum += bins[i].normalSum;
            rightBounds[i] = currentRight;
            rightCount[i] = currentRightCount;
            rightArea[i] = currentRightArea;
            if (useAngularSAH)
                rightNormalSum[i] = currentRightNormalSum;
        }

        AABB leftBounds;
        uint32_t leftCount = 0;
        Float leftArea = 0.0f;
        Vector leftNormalSum(0.0f);
        for (int i = 0; i < nBins - 1; ++i) {
            leftBounds.expandBy(bins[i].bounds);
            leftCount += bins[i].count;
            leftArea += bins[i].area;
            if (useAngularSAH)
                leftNormalSum += bins[i].normalSum;

            if (leftCount == 0 || rightCount[i + 1] == 0) continue;

            // SAH Cost: C = Ct + (Al * Nl + Ar * Nr) / Atotal
            Float spatialCost = traversalCost + (leftBounds.getSurfaceArea() * leftCount + 
                                                 rightBounds[i + 1].getSurfaceArea() * rightCount[i + 1]) 
                                                / bounds.getSurfaceArea();

            Float cost = spatialCost;
            Float angularCost = 0.0f;
            if (useAngularSAH) {
                const Float leftDispersion = computeAngularDispersion(leftArea, leftNormalSum);
                const Float rightDispersion = computeAngularDispersion(rightArea[i + 1], rightNormalSum[i + 1]);
                angularCost = (leftDispersion * leftCount + rightDispersion * rightCount[i + 1])
                            / (Float) nPrims;
                cost += m_angularSAHWeight * angularCost;
            }
            
            if (cost < minCost) {
                minCost = cost;
                bestAxis = axis;
                bestSplitBin = i;
                bestAngularCost = angularCost;
            }
        }
    }

    // --- Termination Decision ---
    Float leafCost = static_cast<Float>(nPrims) * intersectionCost;
    
    // If splitting doesn't actually reduce the cost (or we failed to find a split), make a leaf
    if (minCost >= leafCost || bestAxis == -1) {
        return makeLeaf();
    }

    // --- Partition and Recurse ---
    size_t mid;
    auto midPtr = std::partition(buildPrims.begin() + start, buildPrims.begin() + end,
        [&](const BVHBuildPrimitive &p) {
            Float offset = (p.centroid[bestAxis] - centroidBounds.min[bestAxis]) / diag[bestAxis];
            int b = static_cast<int>(nBins * offset);
            if (b == nBins) b = nBins - 1;
            return b <= bestSplitBin;
        });
    mid = std::distance(buildPrims.begin(), midPtr);

    // Guard against degenerate SAH partitions that place all primitives on one side.
    if (mid == start || mid == end) {
        return makeLeaf();
    }

    if (useAngularSAH) {
        m_buildAngularPenaltySum += (double) bestAngularCost;
        m_buildWeightedAngularPenaltySum += (double) (m_angularSAHWeight * bestAngularCost);
        ++m_buildAngularSplitCount;
    }

    // Create interior node
    size_t nodeIndex = m_nodes.size();
    BVHNode node;
    node.bounds = bounds;
    node.nLeafPrimitives = 0; 
    node.nSubtreePrimitives = static_cast<uint32_t>(nPrims);
    node.splitAxis = static_cast<uint8_t>(bestAxis);
    m_nodes.push_back(node);
    
    buildBVHRecursiveSAH(buildPrims, start, mid, orderedPrims, orderedAreas, depth + 1);
    
    m_nodes[nodeIndex].secondChildOffset = static_cast<uint32_t>(
        buildBVHRecursiveSAH(buildPrims, mid, end, orderedPrims, orderedAreas, depth + 1)
    );
    
    return nodeIndex;
}

void SamplingBVH::buildAggregatesRecursive(size_t nodeIndex, const Scene *scene) {
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
                mesh, prim.triangleIndex, scene
            );
            
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

    // Precompute derived values (kappa_n, mu_n, kappa_m, kappa_conv) from the finalized sums
    #ifndef USE_SGGX_AGGREGATES
    info.finalize();
    #endif
}

 void BVHNodeInfo::setFromTriangle(const Point& v1, const Point& v2, const Point& v3, const Shape* shape) {
    Vector e1 = v2 - v1;
    Vector e2 = v3 - v1;
    Vector crossProd = cross(e1, e2);
    surfaceArea = 0.5f * crossProd.length();

    if (!std::isfinite(surfaceArea) || surfaceArea <= 0.0f) {
        // Mark invalid so accumulate() ignores this contribution.
        valid = false;
        surfaceArea = 0.0f;
        s_diffuse_albedo = Spectrum(0.0f);
        s_metallic_albedo = Spectrum(0.0f);
        s_specular = 0.0f;
        s_metallic_specular = 0.0f;
        s_alpha_sq = 0.0f;
        return;
    }
    
    // Compute (unit) geometric normal
    Vector gn = normalize(crossProd);

    #ifdef USE_SGGX_AGGREGATES
    // Initialize SGGX moments from this single-triangle normal
    normalDistribution.setFromNormal(gn);
    #else 
    gaussianAggregate.setFromTriangle(v1, v2, v3, surfaceArea); // For a single triangle, the Gaussian aggregate is just the triangle's contribution
    normalDistribution.s_normals = gn * surfaceArea; // For a single triangle, the sum of weighted normals is just the normal times the area
    #endif

    const BSDF *bsdf = shape->getBSDF();

    if (bsdf) {
        // Create a dummy intersection for BSDF evaluation
        Intersection its;
        its.p = (v1 + v2 + v3) / 3.0f; // centroid
        its.geoFrame = Frame(Normal(gn));
        its.shFrame = its.geoFrame;
        its.wi = its.toLocal(Vector(0, 0, 1));
        its.shape = shape;

        Spectrum diffuseRefl = bsdf->getDiffuseReflectance(its);
        Spectrum rawRefl    = bsdf->getRawReflectance(its);
        Float    metallic   = bsdf->getMetallic(its);
        Float    specularF0 = bsdf->getSpecularF0(its);

        Float roughness = 1.0f; // default: fully rough (for non-glossy BSDFs)
        if (bsdf->getType() & BSDF::EGlossyReflection)
            roughness = bsdf->getRoughness(its, 0);
        roughness = std::max((Float) 1e-4f, std::min((Float) 1.0f, roughness));

        // Store as area-weighted sums (single triangle: multiply by surfaceArea)
        s_diffuse_albedo    = surfaceArea * diffuseRefl;
        s_metallic_albedo   = surfaceArea * rawRefl * metallic;
        s_specular          = surfaceArea * specularF0;
        s_metallic_specular = surfaceArea * metallic * specularF0;
        s_alpha_sq          = surfaceArea * roughness * roughness;
    } else {
        s_diffuse_albedo    = surfaceArea * Spectrum(0.5f); // conservative gray fallback
        s_metallic_albedo   = Spectrum(0.0f);
        s_specular          = 0.0f;
        s_metallic_specular = 0.0f;
        s_alpha_sq          = surfaceArea * 1.0f; // fully rough
    }
    
    valid = true;
}
BVHNodeInfo computeTriangleInfo(
    const Shape *shape,
    size_t triangleIndex,
    const Scene *scene)
{
    BVHNodeInfo info;
    
    const TriMesh *mesh = dynamic_cast<const TriMesh*>(shape);
    if (!mesh) {
        SLog(EWarn, "computeTriangleInfo: shape is not a TriMesh");
        return info;
    }
    
    const Point *vertexPositions = mesh->getVertexPositions();
    const Triangle *triangles = mesh->getTriangles();
    
    if (triangleIndex >= mesh->getTriangleCount()) {
        SLog(EWarn, "computeTriangleInfo: triangleIndex out of bounds");
        return info;
    }
    
    const Triangle &tri = triangles[triangleIndex];
    Point p0 = vertexPositions[tri.idx[0]];
    Point p1 = vertexPositions[tri.idx[1]];
    Point p2 = vertexPositions[tri.idx[2]];

    
    // ---------------------------------------------------------
    
    // Now the Gaussian aggregates and normals will be built in World Space
    info.setFromTriangle(p0, p1, p2, shape);

    return info;
}

void BVHNodeInfo::accumulate(const BVHNodeInfo &other) {
    if (!other.valid) return;
    
    if (!valid) {
        // First valid node
        *this = other;
        return;
    }
    
    Float totalAreaNew = surfaceArea + other.surfaceArea;
    if (totalAreaNew <= 0.0f) {
        return; // Avoid division by zero
    }
    
    surfaceArea = totalAreaNew;

    #ifdef USE_SGGX_AGGREGATES
    // Moment-preserving aggregation of normal distribution
    Float w1 = surfaceArea / totalAreaNew;
    Float w2 = other.surfaceArea / totalAreaNew;
    normalDistribution.m1 = normalDistribution.m1 * w1 + other.normalDistribution.m1 * w2;
    normalDistribution.m2 = normalDistribution.m2 * w1 + other.normalDistribution.m2 * w2;
    #else
    normalDistribution.s_normals = normalDistribution.s_normals + other.normalDistribution.s_normals;
    gaussianAggregate.combine(other.gaussianAggregate);
    #endif

    // All material statistics are raw area-weighted sums — just add them directly
    s_diffuse_albedo    += other.s_diffuse_albedo;
    s_metallic_albedo   += other.s_metallic_albedo;
    s_specular          += other.s_specular;
    s_metallic_specular += other.s_metallic_specular;
    s_alpha_sq          += other.s_alpha_sq;

    valid = true;
}

MTS_IMPLEMENT_CLASS_S(SamplingBVH, false, ConfigurableObject)

// Note: plugin export lives in src/geometrybvh/geometrybvh.cpp

MTS_NAMESPACE_END