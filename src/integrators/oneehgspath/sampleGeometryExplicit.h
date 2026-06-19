#pragma once

#include <mitsuba/render/scene.h>
#include <mitsuba/render/hgs/bvh.h>

MTS_NAMESPACE_BEGIN

/**
 * Brute-force stochastic geometry sampling.
 * Iterates over all triangles in the scene, computes importance for each,
 * and samples one proportionally to its importance.
 */
extern bool sampleGeometryExplicit(
    const Scene *scene,
    Sampler *sampler,
    const Intersection &its_xs,
    const PositionSamplingRecord &pRec_xe,
    Intersection &its_xp,
    Float &pdf,
    bool checkVisibility = true
);

MTS_NAMESPACE_END
