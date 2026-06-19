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

GlintTracerMISWorkResult::GlintTracerMISWorkResult(
        const GlintTracerMISConfiguration &conf,
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
           full-resolution version, since glint contributions
           can affect any pixel of this bitmap */
        m_lightImage = new ImageBlock(Bitmap::ESpectrum,
                conf.cropSize, rfilter);
        m_lightImage->setSize(conf.cropSize);
        m_lightImage->setOffset(Point2i(0, 0));
    }

#if GLINTTRACERMIS_DEBUG == 1
    m_debugBlocks.resize(
        conf.maxDepth*(5+conf.maxDepth)/2);

    for (size_t i=0; i<m_debugBlocks.size(); ++i) {
        m_debugBlocks[i] = new ImageBlock(
                Bitmap::ESpectrum, conf.cropSize, rfilter);
        m_debugBlocks[i]->setOffset(Point2i(0,0));
        m_debugBlocks[i]->setSize(conf.cropSize);
    }
#endif
}

GlintTracerMISWorkResult::~GlintTracerMISWorkResult() { }

void GlintTracerMISWorkResult::put(const GlintTracerMISWorkResult *workResult) {
#if GLINTTRACERMIS_DEBUG == 1
    for (size_t i=0; i<m_debugBlocks.size(); ++i)
        m_debugBlocks[i]->put(workResult->m_debugBlocks[i].get());
#endif
    m_block->put(workResult->m_block.get());
    if (m_lightImage)
        m_lightImage->put(workResult->m_lightImage.get());
}

void GlintTracerMISWorkResult::clear() {
#if GLINTTRACERMIS_DEBUG == 1
    for (size_t i=0; i<m_debugBlocks.size(); ++i)
        m_debugBlocks[i]->clear();
#endif
    if (m_lightImage)
        m_lightImage->clear();
    m_block->clear();
}

#if GLINTTRACERMIS_DEBUG == 1
void GlintTracerMISWorkResult::dump(const GlintTracerMISConfiguration &conf,
        const fs::path &prefix, const fs::path &stem) const {
    Float weight = (Float) 1.0f / (Float) conf.sampleCount;
    for (int k = 1; k<=conf.maxDepth; ++k) {
        for (int t=0; t<=k+1; ++t) {
            size_t s = k+1-t;
            Bitmap *bitmap = const_cast<Bitmap *>(m_debugBlocks[strategyIndex(s, t)]->getBitmap());
            ref<Bitmap> ldrBitmap = bitmap->convert(Bitmap::ERGB, Bitmap::EUInt8, -1, weight);
            fs::path filename =
                prefix / fs::path(formatString("%s_k%02i_s%02i_t%02i.png", stem.filename().string().c_str(), k, s, t));
            ref<FileStream> targetFile = new FileStream(filename,
                FileStream::ETruncReadWrite);
            ldrBitmap->write(Bitmap::EPNG, targetFile, 1);
        }
    }
}
#endif

void GlintTracerMISWorkResult::load(Stream *stream) {
#if GLINTTRACERMIS_DEBUG == 1
    for (size_t i=0; i<m_debugBlocks.size(); ++i)
        m_debugBlocks[i]->load(stream);
#endif
    if (m_lightImage)
        m_lightImage->load(stream);
    m_block->load(stream);
}

void GlintTracerMISWorkResult::save(Stream *stream) const {
#if GLINTTRACERMIS_DEBUG == 1
    for (size_t i=0; i<m_debugBlocks.size(); ++i)
        m_debugBlocks[i]->save(stream);
#endif
    if (m_lightImage.get())
        m_lightImage->save(stream);
    m_block->save(stream);
}

std::string GlintTracerMISWorkResult::toString() const {
    return m_block->toString();
}

MTS_IMPLEMENT_CLASS(GlintTracerMISWorkResult, false, WorkResult)
MTS_NAMESPACE_END
