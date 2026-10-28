#include "oneehgsptracer_proc.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{oneehgsptracer}{One-more-vertex NEE from the camera}
 * \parameters{
 *     \parameter{granularity}{\Integer}{Number of samples per work unit
 *         \default{200000}}
 * }
 * Particle-tracer counterpart of the camera technique of
 * \pluginref{oneehgsptracermis}: an emitter position x_e is sampled, the
 * intermediate vertex x_n is sampled from the SamplingBVH as seen from the
 * camera and connected to the sensor. It only renders this direct
 * illumination of the visible geometry (no camera paths, no MIS), and is
 * used by the scenes in scenes/7_glints.
 */
class OneMoreVertexPTracerIntegrator : public Integrator {
public:
    OneMoreVertexPTracerIntegrator(const Properties &props): Integrator(props) {
        /* Granularity of the work units used in parallelizing
        the particle tracing task (default: 200K samples).
        Should be high enough so that sending and accumulating
        the partially exposed films is not the bottleneck. */
        m_granularity = props.getSize("granularity", 200000);
    }

    OneMoreVertexPTracerIntegrator(Stream *stream, InstanceManager *manager)
        : Integrator(stream, manager) {
        m_granularity  = stream->readSize();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
        stream->writeSize(m_granularity);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
                    int sceneResID, int sensorResID, int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);

        Scheduler *sched = Scheduler::getInstance();
        const Sensor *sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        Vector2i size = sensor->getFilm()->getCropSize();

        m_sampleCount = (size_t) scene->getSampler()->getSampleCount()
                        * (size_t) size.x * (size_t) size.y;
        return true;
    }

    void cancel() {
        Scheduler::getInstance()->cancel(m_process);
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> scheduler = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        const Film *film = sensor->getFilm();
        size_t sampleCount = scene->getSampler()->getSampleCount();
        size_t nCores = scheduler->getCoreCount();
        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " samples, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, nCores, nCores == 1 ? "core" : "cores");

        // The scene builds the SamplingBVH because needsSamplingBVH() returns true
        const SamplingBVH *bvh = scene->getSamplingBVH();
        if (!bvh || !bvh->isBuilt())
            Log(EError, "SamplingBVH was not built. Make sure a <geometrybvh> "
                "block is present in the scene XML and the scene contains geometry.");
        
        ref<ParallelProcess> process = new OneMoreVertexPTracerProcess(
            job, queue, m_sampleCount, m_granularity);

        process->bindResource("scene",   sceneResID);
        process->bindResource("sensor",  sensorResID);
        process->bindResource("sampler", samplerResID);
        scheduler->schedule(process);
        m_process = process;
        scheduler->wait(process);
        m_process = NULL;

        return process->getReturnStatus() == ParallelProcess::ESuccess;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "OneMoreVertexPTracerIntegrator[" << endl
            << "  granularity = " << m_granularity << endl
            << "]";
        return oss.str();
    }

    /// Intermediate vertices are sampled from the scene's SamplingBVH
    bool needsSamplingBVH() const { return true; }

    MTS_DECLARE_CLASS()
protected:
    ref<ParallelProcess> m_process;
    size_t m_sampleCount, m_granularity;
};

MTS_IMPLEMENT_CLASS_S(OneMoreVertexPTracerIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(OneMoreVertexPTracerIntegrator, "One-more-vertex particle tracer");
MTS_NAMESPACE_END