#include <mitsuba/core/statistics.h>
#include "onemorevertex_li.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

/*!\plugin{oneehgspath_2step}{One-more-vertex path tracer}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Longest path depth (-1 = infinite) \default{-1}}
 *     \parameter{rrDepth}{\Integer}{Depth at which Russian roulette starts \default{5}}
 *     \parameter{strictNormals}{\Boolean}{See the \pluginref{path} plugin \default{false}}
 *     \parameter{hideEmitters}{\Boolean}{See the \pluginref{path} plugin \default{false}}
 *     \parameter{doAdditionalVertex}{\Boolean}{Enable one-more-vertex NEE \default{true}}
 *     \parameter{useBVH}{\Boolean}{Sample the intermediate vertex with the SamplingBVH
 *         (strategy A). When false, only the BSDF strategy B estimates the length-2
 *         connection \default{true}}
 *     \parameter{disableNEE}{\Boolean}{Disable standard (direct) next-event estimation
 *         \default{false}}
 *     \parameter{disableFirstBounceNEE}{\Boolean}{Skip direct illumination at the first
 *         path vertex, e.g. to render indirect illumination only \default{false}}
 * }
 *
 * Unidirectional path tracer extended with one-more-vertex next-event estimation
 * (Sec. 5 of the paper); see onemorevertex::Li() for the estimator. This is
 * the integrator used for "ours" in the path-tracing results of the paper.
 */
class OneMoreVertexPath2StepIntegrator : public MonteCarloIntegrator {
public:
    OneMoreVertexPath2StepIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) {
        m_options.doAdditionalVertex = props.getBoolean("doAdditionalVertex", true);
        m_options.useBVH = props.getBoolean("useBVH", true);
        m_options.disableNee = props.getBoolean("disableNEE", false);
        m_options.disableFirstBounceNEE = props.getBoolean("disableFirstBounceNEE", false);
        configureOptions();
    }

    /// Unserialize from a binary data stream
    OneMoreVertexPath2StepIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) {
        m_options.unserialize(stream);
        configureOptions();
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        Spectrum result = onemorevertex::Li(m_options, r, rRec);
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        return result;
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
        m_options.serialize(stream);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "OneMoreVertexPath2StepIntegrator[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << "," << endl
            << "  doAdditionalVertex = " << m_options.doAdditionalVertex << "," << endl
            << "  useBVH = " << m_options.useBVH << "," << endl
            << "  disableNEE = " << m_options.disableNee << "," << endl
            << "  disableFirstBounceNEE = " << m_options.disableFirstBounceNEE << endl
            << "]";
        return oss.str();
    }

    /// Intermediate vertices are sampled from the scene's SamplingBVH
    bool needsSamplingBVH() const { return true; }

    MTS_DECLARE_CLASS()
private:
    /// Copy the options handled by MonteCarloIntegrator
    void configureOptions() {
        m_options.maxDepth = m_maxDepth;
        m_options.rrDepth = m_rrDepth;
        m_options.strictNormals = m_strictNormals;
        m_options.hideEmitters = m_hideEmitters;
    }

    OneMoreVertexPathOptions m_options;
};

MTS_IMPLEMENT_CLASS_S(OneMoreVertexPath2StepIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(OneMoreVertexPath2StepIntegrator, "One-more-vertex path tracer");
MTS_NAMESPACE_END
