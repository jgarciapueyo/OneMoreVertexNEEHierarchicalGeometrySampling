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

#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include "oneehgsptracermis_wr.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                             Work result                              */
/* ==================================================================== */

OneMoreVertexPTracerMISWorkResult::OneMoreVertexPTracerMISWorkResult(
        const OneMoreVertexPTracerMISConfiguration &conf,
        const ReconstructionFilter *rfilter, Vector2i blockSize) {
    /* Stores the 'camera image' -- this can be blocked when
       spreading out work to multiple workers */
    if (blockSize == Vector2i(-1, -1))
        blockSize = Vector2i(conf.blockSize, conf.blockSize);

    m_block = new ImageBlock(Bitmap::ESpectrumAlphaWeight, blockSize, rfilter);
    m_block->setOffset(Point2i(0, 0));
    m_block->setSize(blockSize);

    if (conf.lightImage) {
        /* Stores the 'light image' -- every worker requires a
           full-resolution version, since camera-technique contributions
           can affect any pixel of this bitmap */
        m_lightImage = new ImageBlock(Bitmap::ESpectrum,
                conf.cropSize, rfilter);
        m_lightImage->setSize(conf.cropSize);
        m_lightImage->setOffset(Point2i(0, 0));
    }

}

OneMoreVertexPTracerMISWorkResult::~OneMoreVertexPTracerMISWorkResult() { }

void OneMoreVertexPTracerMISWorkResult::put(const OneMoreVertexPTracerMISWorkResult *workResult) {
    m_block->put(workResult->m_block.get());
    if (m_lightImage)
        m_lightImage->put(workResult->m_lightImage.get());
}

void OneMoreVertexPTracerMISWorkResult::clear() {
    if (m_lightImage)
        m_lightImage->clear();
    m_block->clear();
}


void OneMoreVertexPTracerMISWorkResult::load(Stream *stream) {
    if (m_lightImage)
        m_lightImage->load(stream);
    m_block->load(stream);
}

void OneMoreVertexPTracerMISWorkResult::save(Stream *stream) const {
    if (m_lightImage.get())
        m_lightImage->save(stream);
    m_block->save(stream);
}

std::string OneMoreVertexPTracerMISWorkResult::toString() const {
    return m_block->toString();
}

MTS_IMPLEMENT_CLASS(OneMoreVertexPTracerMISWorkResult, false, WorkResult)
MTS_NAMESPACE_END
