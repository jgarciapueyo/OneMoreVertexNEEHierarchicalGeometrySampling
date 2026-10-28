/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#if !defined(__ONEEHGSPTRACERMIS_H)
#define __ONEEHGSPTRACERMIS_H

#include <mitsuba/mitsuba.h>
#include "../oneehgspath/onemorevertex_li.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

/// Configuration of the OneMoreVertexPTracerMIS integrator
struct OneMoreVertexPTracerMISConfiguration {
    int blockSize;
    /// Enable the camera-side BVH sampling technique (stored in the light image)
    bool lightImage;
    size_t sampleCount;
    Vector2i cropSize;
    /// Options of the camera-path (one-more-vertex path tracing) technique
    OneMoreVertexPathOptions path;

    inline OneMoreVertexPTracerMISConfiguration() { }

    inline OneMoreVertexPTracerMISConfiguration(Stream *stream) {
        blockSize = stream->readInt();
        lightImage = stream->readBool();
        sampleCount = stream->readSize();
        cropSize = Vector2i(stream);
        path.unserialize(stream);
    }

    inline void serialize(Stream *stream) const {
        stream->writeInt(blockSize);
        stream->writeBool(lightImage);
        stream->writeSize(sampleCount);
        cropSize.serialize(stream);
        path.serialize(stream);
    }

    void dump() const {
        SLog(EDebug, "OneMoreVertexPTracerMIS configuration:");
        SLog(EDebug, "   Maximum path depth          : %i", path.maxDepth);
        SLog(EDebug, "   Image size                  : %ix%i", cropSize.x, cropSize.y);
        SLog(EDebug, "   Camera BVH technique        : %s", lightImage ? "yes" : "no");
        SLog(EDebug, "   Russian roulette depth      : %i", path.rrDepth);
        SLog(EDebug, "   Block size                  : %i", blockSize);
        SLog(EDebug, "   Number of samples           : " SIZE_T_FMT, sampleCount);
        SLog(EDebug, "   Hide emitters               : %s", path.hideEmitters ? "yes" : "no");
        SLog(EDebug, "   Strict normals              : %s", path.strictNormals ? "yes" : "no");
        SLog(EDebug, "   One-more-vertex NEE         : %s", path.doAdditionalVertex ? "yes" : "no");
        SLog(EDebug, "   BVH strategy for it         : %s", path.useBVH ? "yes" : "no");
        SLog(EDebug, "   Disable NEE                 : %s", path.disableNee ? "yes" : "no");
        SLog(EDebug, "   Disable first bounce NEE    : %s", path.disableFirstBounceNEE ? "yes" : "no");
    }
};

MTS_NAMESPACE_END

#endif /* __ONEEHGSPTRACERMIS_H */
