#if !defined(__GLINTTRACER_PROC_H)
#define __GLINTTRACER_PROC_H

#include <mitsuba/render/particleproc.h>
#include <mitsuba/render/range.h>
#include <mitsuba/render/renderjob.h>
#include <mitsuba/core/bitmap.h>

MTS_NAMESPACE_BEGIN

/* ======================================================================== */
/*  GlintWorkResult                                                          */
/*  Mirrors CaptureParticleWorkResult from ptracer_proc.h/.cpp              */
/* ======================================================================== */

class GlintWorkResult : public ImageBlock {
public:
    inline GlintWorkResult(const Vector2i &res, const ReconstructionFilter *filter)
        : ImageBlock(Bitmap::ESpectrum, res, filter) {
        setOffset(Point2i(0, 0));
        setSize(res);
        m_range = new RangeWorkUnit();
    }

    inline const RangeWorkUnit *getRangeWorkUnit() const {
        return m_range.get();
    }
    
    inline void setRangeWorkUnit(const RangeWorkUnit *range) {
        m_range->set(range);
    }

    /* Work unit implementation */
    void load(Stream *stream);
    void save(Stream *stream) const;

    MTS_DECLARE_CLASS()
protected:
    virtual ~GlintWorkResult() { }
protected:
    ref<RangeWorkUnit> m_range;
};

/* ==================================================================== */
/*  GlintWorker                                                              */
/*  Custom WorkProcessor: BVH camera-origin sampling + sensor splatting.    */
/* ==================================================================== */

class GlintWorker : public WorkProcessor {
public:
    inline GlintWorker() : WorkProcessor() { }

    GlintWorker(Stream *stream, InstanceManager *manager);

    void serialize(Stream *stream, InstanceManager *manager) const;
    void prepare();
    ref<WorkProcessor> clone() const;
    ref<WorkResult> createWorkResult() const;
    ref<WorkUnit> createWorkUnit() const;

    void process(const WorkUnit *workUnit, WorkResult *workResult, const bool &stop);

    MTS_DECLARE_CLASS()
protected:
    /// Virtual destructor
    virtual ~GlintWorker() { }
private:
    void traceSample(GeometryBVH *bvh);

    ref<Scene>    m_scene;
    ref<Sampler>  m_sampler;
    ref<Sensor>   m_sensor;
    ref<const ReconstructionFilter> m_rfilter;
    ref<GlintWorkResult> m_workResult;
    Point  m_cameraOrigin;
};

/* ======================================================================== */
/*  GlintProcess                                                             */
/*  Mirrors CaptureParticleProcess from ptracer_proc.h/.cpp                 */
/* ======================================================================== */

class GlintProcess : public ParticleProcess {
public:
    GlintProcess(const RenderJob *job, RenderQueue *queue,
                 size_t sampleCount, size_t granularity)
        : ParticleProcess(ParticleProcess::ETrace, sampleCount,
                          granularity, "Rendering (glints)", job),
          m_job(job), m_queue(queue) { }

    void develop();

    /* ParallelProcess implementation */
    void processResult(const WorkResult *wr, bool cancelled);
    void bindResource(const std::string &name, int id);
    ref<WorkProcessor> createWorkProcessor() const;

    MTS_DECLARE_CLASS()
protected:
    virtual ~GlintProcess() { }
private:
    ref<const RenderJob> m_job;
    ref<RenderQueue>     m_queue;
    ref<Film>            m_film;
    ref<ImageBlock>      m_accum;
};

MTS_NAMESPACE_END

#endif /* __GLINTTRACER_PROC_H */