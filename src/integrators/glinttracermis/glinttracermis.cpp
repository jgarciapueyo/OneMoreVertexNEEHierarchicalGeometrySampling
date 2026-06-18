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

#include <mitsuba/bidir/vertex.h>
#include <mitsuba/bidir/edge.h>
#include "glinttracermis_proc.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{glinttracermis}{GlintTracerMIS integrator}
 * \order{6}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *         A value of \code{1} will only render directly visible light sources.
 *         \code{2} will lead to single-bounce (direct-only) illumination,
 *         and so on. \default{\code{-1}}
 *     }
 *     \parameter{lightImage}{\Boolean}{Include glint sampling strategies that connect
 *        paths traced via geometry BVH directly to the camera? This improves the
 *        effectiveness of the integrator for specular/glint materials but requires
 *        a full-resolution light image buffer per worker.
 *        \default{include these strategies, i.e. \code{true}}
 *     }
 *     \parameter{sampleDirect}{\Boolean}{Enable direct sampling strategies?
 *        \default{use direct sampling, i.e. \code{true}}}
 *     \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *        which the implementation will start to use the ``russian roulette''
 *        path termination criterion. \default{\code{5}}
 *     }
 * }
 *
 * This integrator combines standard path tracing with a geometry-BVH glint
 * sampling technique using multiple importance sampling (MIS). The glint
 * technique importance-samples intermediate specular/glint surfaces from the
 * camera origin, generating length-2 paths that are MIS-weighted against the
 * camera-ray NEE contributions from the path tracer.
 *
 * Glint contributions may land on any pixel (determined by connecting the
 * sampled surface vertex back to the sensor), so they are accumulated in a
 * separate full-resolution light image and combined with the camera image at
 * the end of rendering, analogous to BDPT's light image handling.
 */
class GlintTracerMISIntegrator : public Integrator {
public:
    GlintTracerMISIntegrator(const Properties &props) : Integrator(props) {
        /* Load the parameters / defaults */
        m_config.maxDepth = props.getInteger("maxDepth", 5);
        m_config.rrDepth = props.getInteger("rrDepth", 5);
        m_config.hideEmitters = props.getBoolean("hideEmitters", false);
        m_config.strictNormals = props.getBoolean("strictNormals", true);
        m_config.lightImage = props.getBoolean("lightImage", true);
        m_config.sampleDirect = props.getBoolean("sampleDirect", true);
        m_config.showWeighted = props.getBoolean("showWeighted", false);

        if (m_config.rrDepth <= 0)
            Log(EError, "'rrDepth' must be set to a value greater than zero!");

        if (m_config.maxDepth <= 0 && m_config.maxDepth != -1)
            Log(EError, "'maxDepth' must be set to -1 (infinite) or a value greater than zero!");


        #if ENABLE_GLINTTRACER_DOUBLESTEP == 1
        
        m_config.m_doAdditionalVertex = props.getBoolean("doAdditionalVertex", true);
        m_config.m_useBVH = props.getBoolean("useBVH", true);
        m_config.m_disableNee = props.getBoolean("disableNEE", false);

        m_config.m_disableFirstBounceNEE = props.getBoolean("disableFirstBounceNEE", false); 

        // Debug flags for bias isolation:
        //   debug1 = 0.  ->  disable entire L2 additional-vertex block (plain PT+NEE only)
        //   debug2 = 0.  ->  disable Strategy A (BVH / 2NEE contribution)
        //   debug3 = 0.  ->  disable Strategy B (BSDF+NEE L2 contribution)
        //   debug4 = 0.  ->  force all L2 MIS weights to 1 (diagnose double-counting/PDF errors)
        m_config.m_debug1 = props.getFloat("debug1", 1.f);
        m_config.m_debug2 = props.getFloat("debug2", 1.f);
        m_config.m_debug3 = props.getFloat("debug3", 1.f);
        m_config.m_debug4 = props.getFloat("debug4", 1.f);
        #endif

    }

    /// Unserialize from a binary data stream
    GlintTracerMISIntegrator(Stream *stream, InstanceManager *manager)
     : Integrator(stream, manager) {
        m_config = GlintTracerMISConfiguration(stream);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
        m_config.serialize(stream);
    }

    bool preprocess(const Scene *scene, RenderQueue *queue,
            const RenderJob *job, int sceneResID, int sensorResID,
            int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID,
                sensorResID, samplerResID);

        if (scene->getSubsurfaceIntegrators().size() > 0)
            Log(EError, "Subsurface integrators are not supported "
                "by the GlintTracerMIS integrator!");

        return true;
    }

    void cancel() {
        Scheduler::getInstance()->cancel(m_process);
    }

    void configureSampler(const Scene *scene, Sampler *sampler) {
        /* Prepare the sampler for tile-based rendering */
        sampler->setFilmResolution(scene->getFilm()->getCropSize(), true);
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> scheduler = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        const Film *film = sensor->getFilm();
        size_t sampleCount = scene->getSampler()->getSampleCount();
        size_t nCores = scheduler->getCoreCount();

        Log(EDebug, "Size of data structures: PathVertex=%i bytes, PathEdge=%i bytes",
            (int) sizeof(PathVertex), (int) sizeof(PathEdge));

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " samples, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, nCores, nCores == 1 ? "core" : "cores");

        m_config.blockSize = scene->getBlockSize();
        m_config.cropSize = film->getCropSize();
        m_config.sampleCount = sampleCount;
        m_config.dump();

        ref<GlintTracerMISProcess> process = new GlintTracerMISProcess(job, queue, m_config);
        m_process = process;

        process->bindResource("scene", sceneResID);
        process->bindResource("sensor", sensorResID);
        process->bindResource("sampler", samplerResID);
        scheduler->schedule(process);

        scheduler->wait(process);
        m_process = NULL;
        process->develop();

        #if GLINTTRACERMIS_DEBUG == 1
            fs::path path = scene->getDestinationFile();
            if (m_config.lightImage)
                process->getResult()->dump(m_config, path.parent_path(), path.stem());
        #endif

        return process->getReturnStatus() == ParallelProcess::ESuccess;
    }

    MTS_DECLARE_CLASS()
private:
    ref<ParallelProcess> m_process;
    GlintTracerMISConfiguration m_config;
};

MTS_IMPLEMENT_CLASS_S(GlintTracerMISIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(GlintTracerMISIntegrator, "GlintTracerMIS integrator");
MTS_NAMESPACE_END
