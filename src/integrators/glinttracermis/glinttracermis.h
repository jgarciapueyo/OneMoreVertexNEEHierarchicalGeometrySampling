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

#if !defined(__GLINTTRACERMIS_H)
#define __GLINTTRACERMIS_H

#include <mitsuba/mitsuba.h>

#define ENABLE_GLINTTRACER_DOUBLESTEP 1

/**
 * When the following is set to "1", the GlintTracerMIS integrator
 * will generate a series of debugging images that split up the final
 * rendering into the weighted contributions of the individual sampling
 * strategies.
 */
//#define GLINTTRACERMIS_DEBUG 1

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                         Configuration storage                        */
/* ==================================================================== */

/**
 * \brief Stores all configuration parameters of the
 * GlintTracerMIS integrator
 */
struct GlintTracerMISConfiguration {
    int maxDepth, blockSize, borderSize;
    bool lightImage;
    bool sampleDirect;
    bool showWeighted;
    size_t sampleCount;
    Vector2i cropSize;
    int rrDepth;
    bool hideEmitters;
    bool strictNormals;
    #if ENABLE_GLINTTRACER_DOUBLESTEP == 1
        bool m_doAdditionalVertex;
        bool m_useBVH;
        bool m_disableNee;
        bool m_disableFirstBounceNEE;
        float m_debug1, m_debug2, m_debug3, m_debug4; // debug flags for bias isolation
    #endif

    inline GlintTracerMISConfiguration() { }

    inline GlintTracerMISConfiguration(Stream *stream) {
        maxDepth = stream->readInt();
        blockSize = stream->readInt();
        lightImage = stream->readBool();
        sampleDirect = stream->readBool();
        showWeighted = stream->readBool();
        sampleCount = stream->readSize();
        cropSize = Vector2i(stream);
        rrDepth = stream->readInt();
        hideEmitters = stream->readBool();
        strictNormals = stream->readBool();
    }

    inline void serialize(Stream *stream) const {
        stream->writeInt(maxDepth);
        stream->writeInt(blockSize);
        stream->writeBool(lightImage);
        stream->writeBool(sampleDirect);
        stream->writeBool(showWeighted);
        stream->writeSize(sampleCount);
        cropSize.serialize(stream);
        stream->writeInt(rrDepth);
        stream->writeBool(hideEmitters);
        stream->writeBool(strictNormals);
    }

    void dump() const {
        SLog(EDebug, "GlintTracerMIS configuration:");
        SLog(EDebug, "   Maximum path depth          : %i", maxDepth);
        SLog(EDebug, "   Image size                  : %ix%i",
            cropSize.x, cropSize.y);
        SLog(EDebug, "   Direct sampling strategies  : %s",
            sampleDirect ? "yes" : "no");
        SLog(EDebug, "   Generate light image        : %s",
            lightImage ? "yes" : "no");
        SLog(EDebug, "   Russian roulette depth      : %i", rrDepth);
        SLog(EDebug, "   Block size                  : %i", blockSize);
        SLog(EDebug, "   Number of samples           : " SIZE_T_FMT, sampleCount);
        #if GLINTTRACERMIS_DEBUG == 1
            SLog(EDebug, "   Show weighted contributions : %s", showWeighted ? "yes" : "no");
        #endif
        SLog(EDebug, "   Hide emitters               : %s", hideEmitters ? "yes" : "no");
        SLog(EDebug, "   Strict normals              : %s", strictNormals ? "yes" : "no");
        #if ENABLE_GLINTTRACER_DOUBLESTEP == 1
            SLog(EDebug, "   Additional vertex           : %s", m_doAdditionalVertex ? "yes" : "no");
            SLog(EDebug, "   Use BVH for additional vert : %s", m_useBVH ? "yes" : "no");
            SLog(EDebug, "   Disable NEE for additional vert : %s", m_disableNee ? "yes" : "no");
            SLog(EDebug, "   Disable first bounce NEE    : %s", m_disableFirstBounceNEE ? "yes" : "no");
            SLog(EDebug, "   Debug flag 1 (disable L2 block) : %f", m_debug1);
            SLog(EDebug, "   Debug flag 2 (disable BVH/2NEE) : %f", m_debug2);
            SLog(EDebug, "   Debug flag 3 (disable BSDF/NEE) : %f", m_debug3);
            SLog(EDebug, "   Debug flag 4 (force MIS weights=1) : %f", m_debug4);
        #endif
    }
};

MTS_NAMESPACE_END

#endif /* __GLINTTRACERMIS_H */
