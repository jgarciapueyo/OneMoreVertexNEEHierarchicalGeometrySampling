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

SamplingBVH::SamplingBVH(uint32_t maxDepth, SamplingBVHSamplingMode samplingMode)
    : ConfigurableObject(Properties()), m_maxDepth(maxDepth), m_samplingMode(samplingMode), m_ablationMask(AblateNone) {
    if (m_maxDepth < 1) {
        m_maxDepth = 1;
    }
    m_rrStrength = 1.0f;
}

SamplingBVH::SamplingBVH(const Properties &props)
    : ConfigurableObject(props),
      m_maxDepth(props.getInteger("maxDepth", 16)), // Default to depth 16 if not provided
      m_samplingMode(SamplingBVHSamplingMode::Primitive),
    m_vmfLUT(), // Load VMF LUT during construction
    m_ablationMask(AblateNone)
{
    if (m_maxDepth < 1) {
        m_maxDepth = 1;
    }
    
    std::string samplingModeStr = props.getString("samplingMode", "Primitive");
    if (samplingModeStr == "SphericalAABB") {
        m_samplingMode = SamplingBVHSamplingMode::SphericalAABB;
    } else if (samplingModeStr == "Primitive") {
        m_samplingMode = SamplingBVHSamplingMode::Primitive;
    } else if (samplingModeStr == "Primitive_3_retries") {
        m_samplingMode = SamplingBVHSamplingMode::Primitive_3_retries;
    } else if (samplingModeStr == "Primitive_5_retries") {
        m_samplingMode = SamplingBVHSamplingMode::Primitive_5_retries;
    } else if (samplingModeStr == "SphericalAABB_anyNode") {
        m_samplingMode = SamplingBVHSamplingMode::SphericalAABB_anyNode;
    }
    else {
        SLog(EWarn, "SamplingBVH: Unknown sampling mode '%s', defaulting to 'Primitive'", 
            samplingModeStr.c_str());
    }

    m_enableLightcuts = props.getBoolean("enableLightcuts", false);
    m_lightcutThreshold = props.getFloat("lightcutThreshold", 2.f);
    m_gaussianSAThreshold = props.getFloat("gaussianSAThreshold", M_PI * 1e-2f);
    m_kappaThreshold = props.getFloat("kappaThreshold", 10.f);
    m_angularSAHWeight = std::max(0.0f, props.getFloat("angularSAHWeight", 0.0f));
    m_angularSAHKappaScale = std::max(1e-4f, props.getFloat("angularSAHKappaScale", 10.0f));
    m_logAngularSAHStats = props.getBoolean("logAngularSAHStats", true);

    // Read BVH importance parameters (default: 1.0 = full importance, 0.0 = ignore): 
    m_debug1 = props.getFloat("debug1", 1.0f);
    m_debug2 = props.getFloat("debug2", 1.0f);
    m_debug3 = props.getFloat("debug3", 1.0f);
    m_debug4 = props.getFloat("debug4", 1.0f);
    m_debug5 = props.getFloat("debug5", 1.0f);

    if (m_debug1 < .99f || m_debug2 < .99f || m_debug3 < .99f || m_debug4 < .99f || m_debug5 < .99f) {
        SLog(EWarn, "SamplingBVH: Configured debug values: debug1=%.2f, debug2=%.2f, debug3=%.2f, debug4=%.2f, debug5=%.2f\n\n",
            m_debug1, m_debug2, m_debug3, m_debug4, m_debug5);
    }

    // more importance params:
    m_probmult = props.getFloat("importanceMultiplier", 1.0f);
    m_defensive_pdf = props.getFloat("defensivePDF", 1e-16f);
    m_probmult = std::max(0.f, m_probmult);
    m_rrStrength = std::max(0.f, props.getFloat("rrStrength", 0.0f));

    // Intuitive ablation controls: each component can be toggled independently.
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

    if (m_probmult != 1.0f || m_defensive_pdf != 1e-16f) {
        SLog(EInfo, "SamplingBVH: Configured importance parameters: importanceMultiplier=%.2f, defensivePDF=%.4f",
            m_probmult, m_defensive_pdf);
    }

    if (m_ablationMask != AblateNone) {
        SLog(EWarn,
            "SamplingBVH: Ablation enabled: mask=%u [VMF=%s, Gaussian=%s, CosineXs=%s, CosineXe=%s, DistanceXs=%s, DistanceXe=%s, Albedo=%s]",
            m_ablationMask,
            (m_ablationMask & AblateVMF) ? "true" : "false",
            (m_ablationMask & AblateGaussian) ? "true" : "false",
            (m_ablationMask & AblateCosineXs) ? "true" : "false",
            (m_ablationMask & AblateCosineXe) ? "true" : "false",
            (m_ablationMask & AblateDistanceXs) ? "true" : "false",
            (m_ablationMask & AblateDistanceXe) ? "true" : "false",
            (m_ablationMask & AblateAlbedo) ? "true" : "false");
    }

    m_outputDebugCSV = props.getBoolean("outputDebugCSV", false);

    const uint32_t defaultEnabledLevels = 0xffffffff; // 1<<3 | 1<<5; // | 1<<4 | 1<<5; // by default enable levels 3, 4 and 5 (but not deeper levels to avoid huge CSV files)

    m_csvEnabledLevels = props.getInteger("csvEnabledLevels", defaultEnabledLevels);

    SLog(EInfo, "SamplingBVH. Summary of the configuration: maxDepth=%u, samplingMode=%s, enableLightcuts=%s, lightcutThreshold=%.2f, gaussianSAThreshold=%.4f, kappaThreshold=%.2f, angularSAHWeight=%.3f, angularSAHKappaScale=%.3f, importanceMultiplier=%.2f, defensivePDF=%.4f, rrStrength=%.3f, outputDebugCSV=%s, csvEnabledLevels=%u",
        m_maxDepth,
        (m_samplingMode == SamplingBVHSamplingMode::Primitive) ? "Primitive" : 
        (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB) ? "SphericalAABB" :
        (m_samplingMode == SamplingBVHSamplingMode::Primitive_3_retries) ? "Primitive_3_retries" :
        (m_samplingMode == SamplingBVHSamplingMode::Primitive_5_retries) ? "Primitive_5_retries" :
        (m_samplingMode == SamplingBVHSamplingMode::SphericalAABB_anyNode) ? "SphericalAABB_anyNode" : "Unknown",
        m_enableLightcuts ? "true" : "false",
        m_lightcutThreshold,
        m_gaussianSAThreshold,
        m_kappaThreshold,
        m_angularSAHWeight,
        m_angularSAHKappaScale,
        m_probmult,
        m_defensive_pdf,
        m_rrStrength,
        m_outputDebugCSV ? "true" : "false",
        m_csvEnabledLevels
    );
}

SamplingBVH::~SamplingBVH() {
    // vectors clean up automatically
}

size_t SamplingBVH::getMemoryUsageBytes() const {
    size_t totalBytes = sizeof(SamplingBVH);

    totalBytes += m_nodes.capacity() * sizeof(BVHNode);
    totalBytes += m_primitives.capacity() * sizeof(BVHPrimitive);
    totalBytes += m_nodeInfos.capacity() * sizeof(BVHNodeInfo);

    totalBytes += m_leafAliasTableOffsets.capacity() * sizeof(uint32_t);
    totalBytes += m_leafAliasProb.capacity() * sizeof(Float);
    totalBytes += m_leafAliasAlias.capacity() * sizeof(uint32_t);

    totalBytes += m_primitiveLevels.capacity() * sizeof(uint32_t);
    totalBytes += m_meshOffsets.capacity() * sizeof(uint32_t);
    totalBytes += m_triangleToPrim.capacity() * sizeof(uint32_t);

    std::cout << "Capacities of each substructure: " << std::endl;
    std::cout << "  Nodes: " << m_nodes.size() << " / " << m_nodes.capacity() << " (" << m_nodes.capacity() * sizeof(BVHNode) << "B, "<< (m_nodes.capacity() * sizeof(BVHNode)) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  Primitives: " << m_primitives.size() << " / " << m_primitives.capacity() << " (" << m_primitives.capacity() * sizeof(BVHPrimitive) << "B, "<< (m_primitives.capacity() * sizeof(BVHPrimitive)) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  NodeInfos: " << m_nodeInfos.size() << " / " << m_nodeInfos.capacity() << " (" << m_nodeInfos.capacity() * sizeof(BVHNodeInfo) << "B, "<< (m_nodeInfos.capacity() * sizeof(BVHNodeInfo)) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  LeafAliasTableOffsets: " << m_leafAliasTableOffsets.size() << " / " << m_leafAliasTableOffsets.capacity() << " (" << m_leafAliasTableOffsets.capacity() * sizeof(uint32_t) << "B, "<< (m_leafAliasTableOffsets.capacity() * sizeof(uint32_t)) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  LeafAliasProb: " << m_leafAliasProb.size() << " / " << m_leafAliasProb.capacity() << " (" << m_leafAliasProb.capacity() * sizeof(Float   ) << "B, "<< (m_leafAliasProb.capacity() * sizeof(Float   )) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  LeafAliasAlias: " << m_leafAliasAlias.size() << " / " << m_leafAliasAlias.capacity() << " (" << m_leafAliasAlias.capacity() * sizeof(uint32_t) << "B, "<< (m_leafAliasAlias.capacity() * sizeof(uint32_t)) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  PrimitiveLevels: " << m_primitiveLevels.size() << " / " << m_primitiveLevels.capacity() << " (" << m_primitiveLevels.capacity() * sizeof(uint32_t) << "B, "<< (m_primitiveLevels.capacity() * sizeof(uint32_t)) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  MeshOffsets: " << m_meshOffsets.size() << " / " << m_meshOffsets.capacity() << " (" << m_meshOffsets.capacity() * sizeof(uint32_t) << "B, "<< (m_meshOffsets.capacity() * sizeof(uint32_t)) / (1024 * 1024) << "MB)" << std::endl;
    std::cout << "  TriangleToPrim: " << m_triangleToPrim.size() << " / " << m_triangleToPrim.capacity() << " (" << m_triangleToPrim.capacity() * sizeof(uint32_t) << "B, "<< (m_triangleToPrim.capacity() * sizeof(uint32_t)) / (1024 * 1024) << "MB)" << std::endl;

    return totalBytes;
}

SamplingBVH::SamplingBVH(Stream *stream, InstanceManager *manager)
    : ConfigurableObject(stream, manager) 
{
    m_maxDepth = stream->readUInt();
    uint32_t samplingModeInt = stream->readUInt();
    m_samplingMode = static_cast<SamplingBVHSamplingMode>(samplingModeInt);
    m_ablationMask = AblateNone;
    m_rrStrength = 1.0f;
    
    // Note: BVH data (nodes, primitives, etc.) are not serialized here.
    // They should be rebuilt using buildBVH() after deserialization.
}

void SamplingBVH::buildBVH(Scene *scene) {
    if (!scene) {
        SLog(EWarn, "SamplingBVH::build: scene is NULL");
        return;
    }
    const Integrator *integrator = scene->getIntegrator();

    if (integrator) {
        bool supportedIntegrator = false;
        const Integrator *it = integrator;

        // Walk through potential wrapper chains (e.g., adaptive -> subintegrator)
        while (it) {
            const std::string className  = it->getClass()->getName();
            const std::string pluginName = it->getProperties().getPluginName();

            if (className == "DoublestepIntegrator" ||
                className == "AdditionalVertexIntegrator" ||
                className == "GlintTracer" ||
                pluginName == "doublestep" ||
                pluginName == "additionalvertex" ||
                pluginName == "fastdoublestep" ||
                pluginName == "glinttracer" ||
                pluginName == "glinttracermis" ) {
                supportedIntegrator = true;
                break;
            }

            it = it->getSubIntegrator(0);
        }

        if (!supportedIntegrator) {
            SLog(EInfo,
                "SamplingBVH::build: integrator '%s' (plugin '%s') is not supported, disabling SamplingBVH",
                integrator->getClass()->getName().c_str(),
                integrator->getProperties().getPluginName().c_str());
            return;
        }
    }
    m_scene = scene; // Get root scene in case this is a sub-scene
    
    // Step 1: Collect all triangles from all meshes
    std::vector<BVHBuildPrimitive> buildPrims;
    
    const std::vector<TriMesh*> &meshes = scene->getMeshes();
    
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
        
        const bool needAngularSAH = m_enableSAH && (m_angularSAHWeight > 0.0f);
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
    
    // Pass initial depth of 0
    if (m_enableSAH) {
        SLog(EInfo, "SamplingBVH: Building aggregates using SAH-based traversal...");
        buildBVHRecursiveSAH(buildPrims, 0, buildPrims.size(), orderedPrims, orderedAreas, 0);
    }
    else {
        SLog(EInfo, "SamplingBVH: Building aggregates using old traversal...");
        buildBVHRecursive(buildPrims, 0, buildPrims.size(), orderedPrims, orderedAreas, 0);
    }
    
    
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
    buildLeafSamplingTables(scene, orderedAreas);
    
    SLog(EInfo, "SamplingBVH built: %zu nodes, %zu primitives, maxDepth=%u", 
         m_nodes.size(), m_primitives.size(), m_maxDepth);

    if (m_logAngularSAHStats && m_enableSAH && m_angularSAHWeight > 0.0f) {
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

void SamplingBVH::buildLeafSamplingTables(const Scene *scene, const std::vector<Float> &primitiveAreas) {
    m_leafAliasTableOffsets.assign(m_nodes.size(), 0xFFFFFFFFu);
    m_leafAliasProb.clear();
    m_leafAliasAlias.clear();

    if (!scene)
        return;

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
    // Print the total size of the alias tables for debugging
    SLog(EInfo, "SamplingBVH: Built leaf sampling tables: total entries = %zu (prob (%zu) + alias (%zu)), total memory = %.2f KB", 
         m_leafAliasProb.size() + m_leafAliasAlias.size(), 
         m_leafAliasProb.size(),
         m_leafAliasAlias.size(),
         (m_leafAliasProb.size() + m_leafAliasAlias.size()) * sizeof(float) / 1024.f);
}

size_t SamplingBVH::buildBVHRecursive(
    std::vector<BVHBuildPrimitive> &buildPrims,
    size_t start, size_t end,
    std::vector<BVHPrimitive> &orderedPrims,
    std::vector<Float> &orderedAreas,
    uint32_t depth) // Added depth parameter
{
    // Compute bounds of all primitives in range [start, end)
    AABB bounds;
    for (size_t i = start; i < end; ++i) {
        bounds.expandBy(buildPrims[i].bounds);
    }
    
    size_t nPrims = end - start;

    // Create leaf if max depth is reached OR if only 1 primitive remains
    if (depth >= m_maxDepth || nPrims <= 1) {
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
        node.nLeafPrimitives = static_cast<uint32_t>(nPrims); // NOTE: Ensure this doesn't overflow if maxDepth is very low!
        node.splitAxis = 0;
        m_nodes.push_back(node);
        return nodeIndex;
    }
    
    // Compute centroid bounds for split decision
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
        // Can't split meaningfully - create leaf
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
        [splitAxis](const BVHBuildPrimitive &a, const BVHBuildPrimitive &b) {
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
    buildBVHRecursive(buildPrims, start, mid, orderedPrims, orderedAreas, depth + 1);
    
    // Build right child and store its index
    m_nodes[nodeIndex].secondChildOffset = static_cast<uint32_t>(
        buildBVHRecursive(buildPrims, mid, end, orderedPrims, orderedAreas, depth + 1)
    );
    
    return nodeIndex;
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