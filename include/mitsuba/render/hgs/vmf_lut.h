#pragma once

#include <iostream>
#include <cmath>
#include <algorithm>

#include <mitsuba/core/point.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/frame.h>

// Include the auto-generated polynomial coefficients and powers
#include "vmf_poly_unrolled_deg4.h"

MTS_NAMESPACE_BEGIN

struct KappaQuery {
    float exp_k; // Cached exponential for the polynomial transformation
};

class VMFLUT {
public:
    // Constructor no longer needs file paths!
    VMFLUT(const std::string& filepath="", bool verbose = false) {
        if (verbose) {
            SLog(EInfo, "VMFLUT initialized using embedded Polynomial Degree 4 weights.");
        }
    }
    
    inline void init(bool verbose=false) { /* No-op */ }

    // Evaluates based on absolute angles (Legacy format)
    inline float eval(float theta1, float theta2, float phi1, float phi2, float vmf_kappa) const {
        vmf_kappa = std::max(vmf_kappa, 0.f);
        float exp_k = std::exp(-vmf_kappa / 10.0f);

        // Normalize azimuths
        phi1 = std::fmod(phi1, 2.0f * (float)M_PI);
        if (phi1 < 0.0f) phi1 += 2.0f * (float)M_PI;
        phi2 = std::fmod(phi2, 2.0f * (float)M_PI);
        if (phi2 < 0.0f) phi2 += 2.0f * (float)M_PI;

        float dphi = std::abs(phi1 - phi2);
        if (dphi > M_PI) dphi = 2.0f * (float)M_PI - dphi;

        return evaluate_vmf_poly(exp_k, std::cos(theta1), std::cos(theta2), std::cos(dphi));
    }

    // Eval for v1 and v2 in LOCAL coordinates
    inline float eval(const Vector& v1, const Vector& v2, float vmf_kappa) const {
        KappaQuery kq = prepareKappa(vmf_kappa);
        return evalFast(v1, v2, kq);
    }

    // Eval for v1 and v2 in GLOBAL coordinates
    inline float eval(const Vector& v1, const Vector& v2, const Vector& vmf_mu, float vmf_kappa) const {
        Vector mu = vmf_mu;
        if (mu.lengthSquared() <= Epsilon)
            mu = Vector(0.0f, 0.0f, 1.0f);
        mu = normalize(mu);

        Frame vmfFrame(mu);
        Vector v1_local = vmfFrame.toLocal(v1);
        Vector v2_local = vmfFrame.toLocal(v2);
        
        KappaQuery kq = prepareKappa(vmf_kappa);
        return evalFast(v1_local, v2_local, kq); 
    }

    // ---------------------------------------------------------------------------------
    // Fast Eval
    
    // Caches the transformed kappa term
    inline KappaQuery prepareKappa(float vmf_kappa) const {
        vmf_kappa = std::max(vmf_kappa, 0.f);
        return { std::exp(-vmf_kappa / 10.0f) };
    }

    // Extremely fast eval: Bypasses acos() and evaluates the polynomial
    inline float evalFast(const Vector& v1, const Vector& v2, const KappaQuery& kq) const {
        float cos_th1 = math::clamp((float)v1.z, -1.0f, 1.0f);
        float cos_th2 = math::clamp((float)v2.z, -1.0f, 1.0f);

        float cos_dphi = 1.0f; // Default if vectors are strictly parallel to Z
        float len2_1 = v1.x * v1.x + v1.y * v1.y;
        float len2_2 = v2.x * v2.x + v2.y * v2.y;
        
        if (len2_1 > 1e-8f && len2_2 > 1e-8f) {
            float dot_xy = v1.x * v2.x + v1.y * v2.y;
            cos_dphi = math::clamp(dot_xy / std::sqrt(len2_1 * len2_2), -1.0f, 1.0f);
        }

        return evaluate_vmf_poly(kq.exp_k, cos_th1, cos_th2, cos_dphi);
    }
}; 

MTS_NAMESPACE_END