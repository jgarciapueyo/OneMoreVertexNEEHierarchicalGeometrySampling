#include <mitsuba/render/trimesh.h> 
#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include "sampleGeometryExplicit.h"

#include <mitsuba/core/timer.h>


MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);


#define L2_NEE_MULT 1
#define NEE_MULT 1


#define DO_TIMING 0

class DoublestepIntegrator : public MonteCarloIntegrator {
public:
    DoublestepIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) { 
        m_doAdditionalVertex = props.getBoolean("doAdditionalVertex", true);
        m_useBVH = props.getBoolean("useBVH", true);
        m_disableNee = props.getBoolean("disableNEE", false);

        m_disableFirstBounceNEE = props.getBoolean("disableFirstBounceNEE", false); 

        // Debug flags for bias isolation:
        //   debug1 = 0.  ->  disable entire L2 additional-vertex block (plain PT+NEE only)
        //   debug2 = 0.  ->  disable Strategy A (BVH / 2NEE contribution)
        //   debug3 = 0.  ->  disable Strategy B (BSDF+NEE L2 contribution)
        //   debug4 = 0.  ->  force all L2 MIS weights to 1 (diagnose double-counting/PDF errors)
        m_debug1 = props.getFloat("debug1", 1.f);
        m_debug2 = props.getFloat("debug2", 1.f);
        m_debug3 = props.getFloat("debug3", 1.f);
        m_debug4 = props.getFloat("debug4", 1.f);
    }

    /// Unserialize from a binary data stream
    DoublestepIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }
            
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        // std::cout << "Starting path with ray: " << ray.toString() << std::endl;
#if DO_TIMING == 1
        // Log(EInfo, "Timing enabled for DoublestepIntegrator. This may impact performance.");
        ref<Timer> timer = new Timer(false);
        timer->start(); // start a single timer that accumulates total path time

        float time_2NEE = 0.0f, time_BSDF_NEE = 0.0f;
        float time_bvh_sample = 0.0f, time_2neecontrib = 0.0f, time_bsdf2neecontrib = 0.0f;
        float time_bvh_pdf = 0.0f;
        float time_total = 0.0f;
#endif 

        /* Perform the first ray intersection */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        bool prev_evaluated_l2 = false; // avoid double counting if both strategies account for the same L2 contribution

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            // std::cout << "[Doublestep] depth=" << rRec.depth
            //           << " itsValid=" << its.isValid()
            //           << " doAdditionalVertex=" << m_doAdditionalVertex
            //           << " useBVH=" << m_useBVH
            //           << " disableNEE=" << m_disableNee
            //           << " debug1=" << m_debug1
            //           << " debug2=" << m_debug2
            //           << " debug3=" << m_debug3
            //           << " debug4=" << m_debug4
            //           << std::endl;

            /* Emission (Le) from current intersection */
            bool do_nee = m_disableFirstBounceNEE ? (rRec.depth > 1) : true; // optionally disable NEE at the first bounce to isolate its contribution

            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered) && (do_nee))
                Li += throughput * its.Le(-ray.d);

            /* Max depth and normal checks */
            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0) || 
                (m_strictNormals && dot(ray.d, its.geoFrame.n) * Frame::cosTheta(its.wi) >= 0)) {
                break;
            }

            /* ==================================================================== */
            /* STRATEGY 1: Standard Direct Lighting (Length-1 NEE)                  */
            /* ==================================================================== */
            bool run_standard_nee = !m_disableNee && !prev_evaluated_l2; // disable nee if last bounce already accounted for the L2 contribution
            DirectSamplingRecord dRec(its);
            if (run_standard_nee && (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) && (bsdf->getType() & BSDF::ESmooth) && do_nee) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    if (!bsdfVal.isZero() && (!m_strictNormals || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {
                        // Standard MIS between BSDF and Direct Light
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle) ? bsdf->pdf(bRec) : 0;
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        Li += NEE_MULT * throughput * value * bsdfVal * weight;
                    }
                }
            }

            /* ==================================================================== */
            /* PREPARE SHARED SAMPLES FOR LENGTH-2 INDIRECT MIS                     */
            /* ==================================================================== */
            
            // 1. Sample the Emitter (xe) ONCE. This effectively fixes the endpoint for both strategies.
            PositionSamplingRecord pRec_L2;
            Spectrum xe_val = scene->sampleEmitterPosition(pRec_L2, rRec.nextSample2D());

            bool l2_possible = !xe_val.isZero() && m_doAdditionalVertex && (rRec.depth + 1 < m_maxDepth)
                 && (m_debug1 > 0.5f);

            // std::cout << "[Doublestep] L2 gate: xe_val_zero=" << xe_val.isZero()
            //           << " doAdditionalVertex=" << m_doAdditionalVertex
            //           << " depth+1<maxDepth=" << (rRec.depth + 1 < m_maxDepth)
            //           << " debug1>0.5=" << (m_debug1 > 0.5f)
            //           << " => l2_possible=" << l2_possible << std::endl;

            prev_evaluated_l2 = l2_possible;

            // Store results of Strategy A (BVH)
            Intersection its_y; 
            Spectrum contrib_2NEE(0.0f);
            Float pdf_2NEE_solidAngle = 0.0f;

            /* ==================================================================== */
            /* STRATEGY A: Length-2 NEE (The BVH Shortcut)                          */
            /* Path: xs -> y -> xe                                                  */
            /* ==================================================================== */



            if (l2_possible && m_useBVH) {
                // std::cout << "[Doublestep] Strategy A entered (BVH)" << std::endl;
                #if DO_TIMING == 1
                uint64_t t0_bvh_sample = timer->getNanoseconds();
                #endif
                // Note: We assume sampleGeometry returns PDF in Solid Angle (matching your pdfGeometry) 
                // or we might need to convert it here if sampleGeometry returns Area measure.
                // For this snippet, assuming consistent Solid Angle usage as per your comment.
                bool success = scene->getGeometryBVH()->sampleGeometry(
                    scene,
                    rRec.sampler,
                    its,
                    pRec_L2,
                    its_y,
                    pdf_2NEE_solidAngle
                );

                // std::cout << "[Doublestep] Strategy A sampleGeometry: success=" << success
                //           << " pdf_2NEE_solidAngle=" << pdf_2NEE_solidAngle << std::endl;

                #if DO_TIMING == 1
                uint64_t t1_bvh_sample = timer->getNanoseconds();
                time_bvh_sample += (float) (t1_bvh_sample - t0_bvh_sample);
                #endif
                
                if (success && pdf_2NEE_solidAngle > 0) {
                    its_y.wi = its_y.toLocal(normalize(its.p - its_y.p));
#if 1
                    {// Check visibility for both rays: xs->y and y->xe
                        // 1. Check visibility from xs to y

                        #if DO_TIMING == 1
                        uint64_t t0_2nee = timer->getNanoseconds();
                        #endif
                        Vector d_xy = its_y.p - its.p;
                        Float dist_xy = d_xy.length();
                        d_xy /= dist_xy;
                        Ray shadowRay_xy(its.p, d_xy, Epsilon, dist_xy * (1.0f - ShadowEpsilon), ray.time);
                        
                        // 2. Check visibility from y to xe
                        Vector d_ye = pRec_L2.p - its_y.p;
                        Float dist_ye = d_ye.length();
                        d_ye /= dist_ye;
                        Ray shadowRay_ye(its_y.p, d_ye, Epsilon, dist_ye * (1.0f - ShadowEpsilon), ray.time);


                        // Only evaluate the contribution if BOTH segments are completely unoccluded
                        bool occ_xy = scene->rayIntersect(shadowRay_xy);
                        bool occ_ye = scene->rayIntersect(shadowRay_ye);

                        // std::cout << "[Doublestep] Strategy A visibility: occ_xy=" << occ_xy
                        //           << " occ_ye=" << occ_ye << std::endl;

                        if (!occ_xy && !occ_ye) {
                            // std::cout << "[Doublestep] Calling evaluatePathContribution from Strategy A" << std::endl;
                            contrib_2NEE = L2_NEE_MULT * evaluatePathContribution(
                                scene, its, its_y, pRec_L2, throughput, bsdf
                            ) / pdf_2NEE_solidAngle;
                        }
                        //     std::cout << "[Doublestep] Strategy A contrib_2NEE=" << contrib_2NEE.toString() << std::endl;
                        // } else {
                        //     std::cout << "[Doublestep] Strategy A skipped: blocked visibility" << std::endl;
                        // }
                    }
#else
                    // old: no visibility, enable if we do vis in BVH.sampleGeometry

                    // Pass pdf_2NEE_solidAngle to the function
                    contrib_2NEE = L2_NEE_MULT * evaluatePathContribution(
                        scene, its, its_y, pRec_L2, throughput, bsdf
                    ) / pdf_2NEE_solidAngle;
#endif
                    

                    // if (contrib_2NEE.getLuminance() > 10000.f) {
                    //     std::cout << "Debug: Large contrib from 2NEE path: " << contrib_2NEE.toString() << " with pdf_2NEE_solidAngle = " << pdf_2NEE_solidAngle << std::endl;
                    // }
                }
            }

            /* ==================================================================== */
            /* STRATEGY B: BSDF Sampling (The Standard Path)                        */
            /* Path: xs -> z ...                                                    */
            /* ==================================================================== */
            Float bsdfPdf;
            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
            Spectrum bsdfWeight = bsdf->sample(bRec, bsdfPdf, rRec.nextSample2D());
            
            if (bsdfWeight.isZero()) break;

            scattered |= bRec.sampledType != BSDF::ENull;
            const Vector wo = its.toWorld(bRec.wo);
            
            if (m_strictNormals && dot(its.geoFrame.n, wo) * Frame::cosTheta(bRec.wo) <= 0) break;

            // Trace the ray xs -> z
            ray = Ray(its.p, wo, ray.time);
            Intersection its_z;
            bool hit_z = scene->rayIntersect(ray, its_z);

            /* ==================================================================== */
            /* MIS: COMBINE L2 ESTIMATES (BVH vs BSDF+NEE)                          */
            /* ==================================================================== */
            if (l2_possible) {
                
                // --- 1. Weight for Strategy A (BVH) ---
                // Competing PDF: Probability that BSDF sampling would have generated direction to 'y'
                // debug2 > 0: skip Strategy A (BVH) contribution entirely
                if (!contrib_2NEE.isZero() && m_debug2 > 0.5f) {
                    Vector dir_xs_y = normalize(its_y.p - its.p);
                    BSDFSamplingRecord bRec_check(its, its.toLocal(dir_xs_y));
                    Float pdf_bsdf_at_y = bsdf->pdf(bRec_check); // Solid Angle

                    // debug4 > 0: force weight=1 to diagnose double-counting / PDF errors
                    Float weight_A = (m_debug4 > 0.5f)
                        ? miWeight(pdf_2NEE_solidAngle, pdf_bsdf_at_y)
                        : 1.0f;
                    
                    Li += weight_A * contrib_2NEE;
                }

                // --- 2. Evaluate Strategy B (BSDF + NEE) ---
                // We are at 'z' (its_z). We try to connect z -> xe.
                // Only possible if z is a valid surface and not an emitter (emitters handled by L1)
                // debug3 > 0: skip Strategy B (BSDF+NEE L2) contribution entirely
                if (hit_z && its_z.isValid() && !its_z.isEmitter() && m_debug3 > 0.5f) {
                    
                    
                    // Check visibility z -> xe
                    Ray shadowRay(its_z.p, normalize(pRec_L2.p - its_z.p), Epsilon, distance(pRec_L2.p, its_z.p)-Epsilon, ray.time);
                    if (!scene->rayIntersect(shadowRay)) {
                        #if DO_TIMING == 1
                        uint64_t t0_bsdf2nee = timer->getNanoseconds();
                        #endif

                        Spectrum contrib_BSDF_NEE = evaluatePathContribution(scene, its, its_z, pRec_L2, throughput, bsdf) / bsdfPdf;


                        #if DO_TIMING == 1
                        uint64_t t1_bsdf2nee = timer->getNanoseconds();
                        time_bsdf2neecontrib += (float) (t1_bsdf2nee - t0_bsdf2nee);
                        #endif
                        
                        if (!contrib_BSDF_NEE.isZero()) {
                            
                            // Competing PDF: Probability that BVH would have generated 'z'
                            Float pdf_2NEE_at_z = 0.0f;

                            #if DO_TIMING == 1
                            uint64_t t0_bvh_pdf = timer->getNanoseconds();
                            #endif
                            
                            if (its_z.shape && its_z.shape->getClass()->derivesFrom(MTS_CLASS(TriMesh))) {
                                const TriMesh *mesh = static_cast<const TriMesh *>(its_z.shape);
                                // NOTE: pdfGeometry must return Solid Angle.
                                pdf_2NEE_at_z = scene->getGeometryBVH()->pdfGeometry(
                                    scene, 
                                    its, 
                                    pRec_L2,
                                    its_z
                                );
                            }

                            #if DO_TIMING == 1
                            uint64_t t1_bvh_pdf = timer->getNanoseconds();
                            time_bvh_pdf += (float) (t1_bvh_pdf - t0_bvh_pdf);
                            #endif
                            
                            // debug4 > 0: force weight=1 to diagnose double-counting / PDF errors
                            Float weight_B = (m_debug4 > 0.5f)
                                ? miWeight(bsdfPdf, pdf_2NEE_at_z)
                                : 1.0f;

                            Li += weight_B * contrib_BSDF_NEE;
                        }
                    }
                }
            }
            //         std::cout << "[Doublestep] Strategy B visibility z->xe occ_ze=" << occ_ze << std::endl;

            //         if (!occ_ze) {
            //             std::cout << "[Doublestep] Calling evaluatePathContribution from Strategy B (bsdfPdf="
            //                       << bsdfPdf << ")" << std::endl;
            //             Spectrum contrib_BSDF_NEE = evaluatePathContribution(scene, its, its_z, pRec_L2, throughput, bsdf) / bsdfPdf;
            //             std::cout << "[Doublestep] Strategy B contrib_BSDF_NEE=" << contrib_BSDF_NEE.toString() << std::endl;
            //         } else {
            //             std::cout << "[Doublestep] Strategy B skipped: blocked z->xe" << std::endl;
            //         }
            //     } else {
            //         std::cout << "[Doublestep] Strategy B not entered: hit_z=" << hit_z
            //                   << " its_z_valid=" << its_z.isValid()
            //                   << " its_z_isEmitter=" << (hit_z ? its_z.isEmitter() : false)
            //                   << " debug3>0.5=" << (m_debug3 > 0.5f) << std::endl;
            //     }
            // } else {
            //     std::cout << "[Doublestep] L2 block skipped entirely (l2_possible=false)" << std::endl;
            // }

            /* ==================================================================== */
            /* CONTINUE RECURSION                                                   */
            /* ==================================================================== */
            
            throughput *= bsdfWeight; // update throughput for the sampled BSDF direction
            if (!hit_z) {
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance) && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            } else if (its_z.isEmitter()) {
                // Hit light directly (standard PT). 
                // Note: This is part of the L1 calculation (BSDF strategy for direct light)
                dRec.setQuery(ray, its_z);
                Spectrum value = its_z.Le(-ray.d);
                Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                Li += throughput * value * miWeight(bsdfPdf, lumPdf);
                break; // terminate path — emitter already accounted for above
            }

            its = its_z;
            eta *= bRec.eta;

            if (rRec.depth++ >= m_rrDepth) {
                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q) break;
                throughput /= q;
            }
        }

        #if DO_TIMING == 1
        if (time_bvh_pdf>0) {
            // ONLY print timing info if we have data for the pdf
            time_total = (float) timer->getNanoseconds();

            float unaccounted_time = time_total - (time_bvh_sample + time_2neecontrib + time_bsdf2neecontrib + time_bvh_pdf);
                        
            // Table-style timing report
            float total_time = time_bvh_sample + time_2neecontrib + time_bsdf2neecontrib + time_bvh_pdf;
            int effective_bounces = std::max(1, (int)rRec.depth);
    
            Log(EInfo, "Timing Report (ns):");
            Log(EInfo, "%-15s | %10s | %10s | %15s | %10s | %10s | %10s", 
                "Metric", "BVH Sample", "2NEE Contrib", "BSDF2NEE Contrib", "BVH PDF", "Other", "Total Path");
            Log(EInfo, "%s", std::string(125, '-').c_str());
            Log(EInfo, "%-15s | %10.0f | %10.0f | %15.0f | %10.0f | %10.0f | %10.0f", 
                "Total Time (ns)", time_bvh_sample, time_2neecontrib, time_bsdf2neecontrib, time_bvh_pdf, unaccounted_time, time_total);
            Log(EInfo, "%-15s | %10.0f | %10.0f | %15.0f | %10.0f | %10.0f | %10.0f", 
                "Per Bounce (ns)", time_bvh_sample / effective_bounces, time_2neecontrib / effective_bounces, 
                time_bsdf2neecontrib / effective_bounces, time_bvh_pdf / effective_bounces, unaccounted_time / effective_bounces, 
                time_total / effective_bounces);
            
            if (time_total > 0) {
                Log(EInfo, "%-15s | %10.2f%% | %10.2f%% | %15.2f%% | %10.2f%% | %10.2f%% | %5s", 
                    "Percentage (%)", (time_bvh_sample / time_total) * 100, 
                    (time_2neecontrib / time_total) * 100, 
                    (time_bsdf2neecontrib / time_total) * 100, 
                    (time_bvh_pdf / time_total) * 100, (unaccounted_time / time_total) * 100, "100%");
            }

            // Log the overhead of our approach, relative to the unaccounted time:
            if (unaccounted_time > 0) {
                float overhead = ((time_bvh_sample + time_2neecontrib + time_bsdf2neecontrib + time_bvh_pdf) / unaccounted_time) * 100;
                Log(EInfo, "Overhead of L2 strategies relative to unaccounted time: %.2f%%", overhead);
            }

        }
        
        #endif

        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;
        return Li;
    }

// Helper to evaluate raw throughput of a 2-segment path (xs -> p -> xe)
    // Used for both the 2NEE path and the BSDF+NEE path
    // Spectrum evaluatePathContribution(const Scene *scene, const Intersection &xs, const Intersection &p, 
    //                                   const PositionSamplingRecord &pRec, const Spectrum &pathThroughput, 
    //                                   const BSDF *bsdf_xs) const {
        
    //     Vector d_sp = p.p - xs.p;
    //     Float dist_sp = d_sp.length();
    //     d_sp /= dist_sp;

    //     Vector d_pe = pRec.p - p.p;
    //     Float dist_pe = d_pe.length();
    //     d_pe /= dist_pe;

    //     // BSDF at xs (connection to p)
    //     BSDFSamplingRecord bRec_xs(xs, xs.toLocal(d_sp));
    //     Spectrum f_xs = bsdf_xs->eval(bRec_xs);
        
    //     // BSDF at p (connection from xs to xe)
    //     const BSDF *bsdf_p = p.getBSDF();
    //     BSDFSamplingRecord bRec_p(p, p.toLocal(-d_sp), p.toLocal(d_pe));
    //     Spectrum f_p = bsdf_p->eval(bRec_p);

    //     if (f_xs.isZero() || f_p.isZero()) return Spectrum(0.0f);

    //     // Geometry terms
    //     Float cos_theta_p_in = std::abs(dot(p.shFrame.n, -d_sp));
    //     Float cos_theta_p_out = std::abs(dot(p.shFrame.n, d_pe));
    //     Float G_sp = cos_theta_p_in;// / (dist_sp * dist_sp);
    //     Float G_pe = 1.0f / (dist_pe * dist_pe); // Note: cos_theta at Light is handled by Emitter::eval usually, or added here

    //     // Emitter Radiance
    //     const Emitter *emitter = static_cast<const Emitter *>(pRec.object);
    //     DirectionSamplingRecord dRec_direction(-d_pe);
    //     dRec_direction.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
    //     Spectrum Le = emitter->evalPosition(pRec) * emitter->evalDirection(dRec_direction, pRec);

    //     // Combined contribution
    //     // Note: We divide by pRec.pdf (light selection pdf) here or in the main loop? 
    //     // In the main loop we sum (Weight * Contrib), so Contrib should include the 1/pdf_light term.
        
    //     return pathThroughput * f_xs * G_sp * f_p * G_pe * Le * (1.0f / pRec.pdf);
    // }
    Spectrum evaluatePathContribution(const Scene *scene, const Intersection &xs, const Intersection &y, 
                                  const PositionSamplingRecord &pRec, const Spectrum &pathThroughput, 
                                  const BSDF *bsdf_xs) const { 
    
        // std::cout << "[Doublestep] >>> evaluatePathContribution ENTERED"
        //       << " xs=" << xs.p.toString()
        //       << " y=" << y.p.toString()
        //       << " light=" << pRec.p.toString()
        //       << " pRec.pdf=" << pRec.pdf
        //       << std::endl;

        Vector d_sp = y.p - xs.p;
        d_sp = normalize(d_sp); // SA measure, no distance squared needed for this leg

        Vector d_pe = pRec.p - y.p;
        Float dist_pe = d_pe.length();
        d_pe /= dist_pe;

        // 1. BSDF at xs (Includes outgoing cos_theta_xs)
        BSDFSamplingRecord bRec_xs(xs, xs.toLocal(d_sp));
        Spectrum f_xs = bsdf_xs->eval(bRec_xs);

        // 2. BSDF at y (Includes outgoing cos_theta_y)
        const BSDF *bsdf_y = y.getBSDF();
        BSDFSamplingRecord bRec_y(y, y.toLocal(-d_sp), y.toLocal(d_pe));
        Spectrum f_p = bsdf_y->eval(bRec_y);

        if (f_xs.isZero() || f_p.isZero()) return Spectrum(0.0f);

        // 3. Geometry Term for Light Connection ONLY (No cos_theta_light, it's in Le!)
        Float G_pe = 1.0f / (dist_pe * dist_pe);

        // 4. Emitter Radiance (Includes cos_theta_light implicitly via evalDirection)
        const Emitter *emitter = static_cast<const Emitter *>(pRec.object);
        DirectionSamplingRecord dRec_direction(-d_pe);
        dRec_direction.measure = emitter->needsDirectionSample() ? ESolidAngle : EDiscrete;
        Spectrum Le = emitter->evalPosition(pRec) * emitter->evalDirection(dRec_direction, pRec);

        // std::cout << "evaluatePathContribution Debug Info:\n";
        // std::cout << "d_sp: (" << d_sp.x << ", " << d_sp.y << ", " << d_sp.z << "), length = " << (y.p - xs.p).length() << "\n";
        // std::cout << "d_pe: (" << d_pe.x << ", " << d_pe.y << ", " << d_pe.z << "), length = " << dist_pe << "\n";
        // std::cout << "f_xs: " << f_xs.toString() << "\n";
        

        // 5. Final Combine
        // We divide by pRec.pdf (Light Area PDF). pdf_2NEE (Solid Angle) is handled in the MIS loop.
        return pathThroughput * f_xs * f_p * G_pe * Le * (1.0f / pRec.pdf);
        }



    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "DoublestepIntegrator[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << "," << endl
            << "  doAdditionalVertex = " << m_doAdditionalVertex << "," << endl
            << "  useBVH = " << m_useBVH << endl
            << "  disableNEE = " << m_disableNee << endl
            << "  disableFirstBounceNEE = " << m_disableFirstBounceNEE << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_doAdditionalVertex;
    bool m_useBVH;
    bool m_disableNee, m_disableFirstBounceNEE;
    float m_debug1, m_debug2, m_debug3, m_debug4;
};

MTS_IMPLEMENT_CLASS_S(DoublestepIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(DoublestepIntegrator, "Double step integrator");
MTS_NAMESPACE_END