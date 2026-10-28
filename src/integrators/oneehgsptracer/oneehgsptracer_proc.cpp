#include "oneehgsptracer_proc.h"

MTS_NAMESPACE_BEGIN

/* ==================================================================== */
/*                           Work result impl.                          */
/* ==================================================================== */

void OneMoreVertexPTracerWorkResult::load(Stream *stream) {
    size_t nEntries = (size_t) m_size.x * (size_t) m_size.y;
    stream->readFloatArray(reinterpret_cast<Float *>(m_bitmap->getFloatData()),
        nEntries * SPECTRUM_SAMPLES);
    m_range->load(stream);
}

void OneMoreVertexPTracerWorkResult::save(Stream *stream) const {
    size_t n = (size_t) m_size.x * (size_t) m_size.y;
    stream->writeFloatArray(reinterpret_cast<const Float *>(m_bitmap->getFloatData()),
                            n * SPECTRUM_SAMPLES);
    m_range->save(stream);
}

/* ==================================================================== */
/*                         Work processor impl.                         */
/* ==================================================================== */

OneMoreVertexPTracerWorker::OneMoreVertexPTracerWorker(Stream *stream, InstanceManager *manager)
    : WorkProcessor(stream, manager) {}

void OneMoreVertexPTracerWorker::serialize(Stream *stream, InstanceManager *manager) const {
    WorkProcessor::serialize(stream, manager);
}

void OneMoreVertexPTracerWorker::prepare() {
    m_scene   = static_cast<Scene *>(getResource("scene"));

    Sampler *baseSampler = static_cast<Sampler *>(getResource("sampler"));
    m_sampler = baseSampler->clone(); // per-worker sampler

    m_sensor  = static_cast<Sensor *>(getResource("sensor"));
    m_rfilter = m_sensor->getFilm()->getReconstructionFilter();

    const AnimatedTransform *trafo = m_sensor->getWorldTransform();
    m_cameraOrigin = trafo->eval(0.f)(Point(0.f, 0.f, 0.f));
}

ref<WorkProcessor> OneMoreVertexPTracerWorker::clone() const {
    return new OneMoreVertexPTracerWorker();
}

ref<WorkResult> OneMoreVertexPTracerWorker::createWorkResult() const {
    const Film *film = m_sensor->getFilm();
    return new OneMoreVertexPTracerWorkResult(film->getCropSize(), m_rfilter.get());
}

ref<WorkUnit> OneMoreVertexPTracerWorker::createWorkUnit() const {
    return new RangeWorkUnit();
}

void OneMoreVertexPTracerWorker::process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop) {
    const RangeWorkUnit *range = static_cast<const RangeWorkUnit *>(workUnit);
    m_workResult = static_cast<OneMoreVertexPTracerWorkResult *>(workResult);
    m_workResult->setRangeWorkUnit(range);
    m_workResult->clear();

    // Similar to ParticleTracer::process but with custom sampling strategy and no path tracing (single vertex x_n)
    const SamplingBVH *bvh = m_scene->getSamplingBVH();
    if (!bvh || !bvh->isBuilt()) {
        Log(EWarn, "OneMoreVertexPTracer: SamplingBVH not available or not built!");
        return;
    }
    for (size_t index = range->getRangeStart(); index <= range->getRangeEnd() && !stop; ++index) {
        m_sampler->setSampleIndex(index);
        traceSample(bvh);
    }

    m_workResult = NULL;
}

void OneMoreVertexPTracerWorker::traceSample(const SamplingBVH *bvh) {
    // ----------------------------------------------------------------
    // 1. Sample emitter position x_e
    // ----------------------------------------------------------------
    PositionSamplingRecord pRec_xe(0.f);
    m_scene->sampleEmitterPosition(pRec_xe, m_sampler->next2D());
    if (pRec_xe.pdf <= 0.f || !pRec_xe.object) {
        return;
    }


    // ----------------------------------------------------------------
    // 2. BVH camera sampling: find intermediate vertex x_n
    //    The traversal uses computeNodeImportanceGlint (camera field of view,
    //    emitter and diffuse VMF term at x_n).
    // ----------------------------------------------------------------
    Intersection its_xn;
    Float pdf_xn = 0.f;
    if (!bvh->sampleGeometryCamera(m_scene, m_sampler, m_cameraOrigin,
                                    m_sensor.get(), pRec_xe, its_xn, pdf_xn)) {
        return;
    }

    if (pdf_xn <= 0.f || !its_xn.isValid()) {
        return;
    }

    // ----------------------------------------------------------------
    // 3. Sensor connection: determine which pixel x_n maps to and W_e.
    //    sampleAttenuatedSensorDirect also checks visibility x_n → x_o.
    // ----------------------------------------------------------------
    DirectSamplingRecord dRec(its_xn);
    Spectrum W_e = m_scene->sampleSensorDirect(dRec, m_sampler->next2D(), true);
    if (W_e.isZero()) {
        return;
    }

    // dRec.d  = direction from x_n toward the sensor (camera) — normalized
    // dRec.uv = film coordinates of the pixel that sees this direction

    // Set local incoming direction at x_n so BSDF eval is well-defined
    its_xn.wi = its_xn.toLocal(dRec.d);

    // ----------------------------------------------------------------
    // 4. Evaluate x_n → x_e contribution: f_xn * G_ne * Le
    //    (does NOT include pRec_xe.pdf — divided below)
    // ----------------------------------------------------------------
    Spectrum contrib = evalGlintContribution(
        m_scene, its_xn, pRec_xe, dRec.d, /* checkVisibility */ true);
    if (contrib.isZero())
        return;

    // ----------------------------------------------------------------
    // 5. Full estimator weight and splat to pixel
    //    value = W_e * f_xn * G_ne * Le / (pdf_xn * pRec_xe.pdf)
    // ----------------------------------------------------------------
    Spectrum value = W_e * contrib / (pdf_xn * pRec_xe.pdf);

    if (!value.isValid()) {
        Log(EWarn, "OneMoreVertexPTracer: Invalid sample value found! %s", value.toString().c_str());
        return;
    }
    
    m_workResult->put(dRec.uv, (Float *) &value[0]);
}

/* ==================================================================== */
/*                        Parallel process impl.                        */
/* ==================================================================== */

void OneMoreVertexPTracerProcess::develop() {
    Float weight = (m_accum->getWidth() * m_accum->getHeight())
                    / (Float) m_receivedResultCount;
    m_film->setBitmap(m_accum->getBitmap(), weight);
    m_queue->signalRefresh(m_job);
}

void OneMoreVertexPTracerProcess::processResult(const WorkResult *wr, bool cancelled) {
    const OneMoreVertexPTracerWorkResult *result = static_cast<const OneMoreVertexPTracerWorkResult *>(wr);
    const RangeWorkUnit *range = result->getRangeWorkUnit();
    if (cancelled)
        return;
    
    LockGuard lock(m_resultMutex);
    increaseResultCount(range->getSize());
    m_accum->put(result);
    if (m_job->isInteractive() || m_receivedResultCount == m_workCount)
        develop();
}

void OneMoreVertexPTracerProcess::bindResource(const std::string &name, int id) {
    if (name == "sensor") {
        Sensor *sensor = static_cast<Sensor *>(
            Scheduler::getInstance()->getResource(id));
        m_film  = sensor->getFilm();
        m_accum = new ImageBlock(Bitmap::ESpectrum,
                                    m_film->getCropSize(), nullptr);
        m_accum->clear();
    }
    ParticleProcess::bindResource(name, id);
}

ref<WorkProcessor> OneMoreVertexPTracerProcess::createWorkProcessor() const {
    return new OneMoreVertexPTracerWorker();
}

MTS_IMPLEMENT_CLASS(OneMoreVertexPTracerWorker, false, WorkProcessor)
MTS_IMPLEMENT_CLASS(OneMoreVertexPTracerProcess, false, ParticleProcess)
MTS_IMPLEMENT_CLASS(OneMoreVertexPTracerWorkResult, false, WorkResult)
MTS_NAMESPACE_END