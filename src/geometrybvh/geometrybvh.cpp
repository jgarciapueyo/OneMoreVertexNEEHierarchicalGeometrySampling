#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/core/plugin.h>

MTS_NAMESPACE_BEGIN

// ---------------------------------------------------------------------------
// ------- Nestor notes -------
// Note: the main files are in include/mitsuba/render/bvh/ and src/librender/bvh/ 
// this file is just a thin wrapper to export the plugin and define the main class...

// ---------------------------------------------------------------------------
// Note2: why does this have to be here? Bc it seems like plugins cannot be inside librender.
//        (PluginManager gave an error that it couldnt find the plugin)
//    |-> it would be a nice refactor to move the librender/bvh/ files here but i got annoyed yesterday
// ---------------------------------------------------------------------------

MTS_EXPORT_PLUGIN(GeometryBVH, "Geometry BVH with aggregates for importance sampling");

MTS_NAMESPACE_END
