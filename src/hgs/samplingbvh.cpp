#include <mitsuba/render/hgs/bvh.h>
#include <mitsuba/core/plugin.h>

MTS_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------
// The main files are in include/mitsuba/render/hgs/ and src/librender/hgs/ 
// This file is just a thin wrapper to export the plugin and define the main class.
// ---------------------------------------------------------------------------

MTS_EXPORT_PLUGIN(SamplingBVH, "BVH with aggregates for importance sampling vertices on the geometry of a scene");

MTS_NAMESPACE_END
