
#define VMF_LUT_VERSION 4
// 0-> Original version: loads the LUT and uses is directly
// 1-> MLP fitted to the LUT
// 2-> polynomial fitted to the LUT (currently degree 4, but can be changed in fit_LUT.py and re-exported)
// 3-> tensor decomposition fitted to the LUT (currently rank 8, but can be changed in fit_LUT.py and re-exported)
// 4-> unrolled polynomial fitted to the LUT (currently degree 3, but can be changed in fit_LUT.py and re-exported)

// Summary: TLDR: 4 is best.  
//    0,2,3 are very very similar. 1 is much slower, too many weights. 
//    4 is the fastest, using an unrolled polynomial of deg 4. Did not notice a significant variance increase vs 0
//    using degree 3 (by including vmf_poly_unrolled.h instead of vmf_poly_unrolled_deg4.h) is faster, but variance is visible


#if VMF_LUT_VERSION == 0

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>

#include <algorithm>

#include <mitsuba/core/point.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/frame.h>

MTS_NAMESPACE_BEGIN

struct KappaQuery {
    int ik; // kappa index
    float tk; // Weight 
};

class IntensityGrid4D {
private:
    std::vector<float> m_grid;
    
    uint64_t m_num_rows;
    // 4. Allocate vectors for the columns
    std::vector<uint16_t> m_kappa_idx;
    std::vector<uint16_t> m_th1_idx;
    std::vector<uint16_t> m_th2_idx;
    std::vector<uint16_t> m_dphi_idx;
    std::vector<double> m_I_vals;
    
    // Grid dimensions (Must match the LUT)
    static constexpr int N_K = 100;
    static constexpr int N_T1 = 40;
    static constexpr int N_T2 = 40;
    static constexpr int N_DP = 10; 

    // Helper for linear interpolation
    inline float lerp(float a, float b, float t) const {
        return a + t * (b - a);
    }
    // Precomputed strides 
    static constexpr int stride_dp = 1;
    static constexpr int stride_t2 = N_DP;
    static constexpr int stride_t1 = N_T2 * N_DP;
    static constexpr int stride_k  = N_T1 * N_T2 * N_DP;

    inline int get_flat_index(int ik, int it1, int it2, int idp) const {
        return ik * stride_k + it1 * stride_t1 + it2 * stride_t2 + idp;
    }

public:
    inline IntensityGrid4D(const std::string& filepath, bool verbose = false) {
        load(filepath, verbose);
    }
    IntensityGrid4D() : m_num_rows(0) {}

    void load(const std::string& filepath, bool verbose = false) {
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + filepath);
        
        // 1. Read number of rows
        uint64_t num_rows;
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(uint64_t));

        // 2. Allocate temporary vectors for the columns
        std::vector<uint16_t> kappa_idx(num_rows);
        std::vector<uint16_t> th1_idx(num_rows);
        std::vector<uint16_t> th2_idx(num_rows);
        std::vector<uint16_t> dphi_idx(num_rows);
        std::vector<double> I_vals(num_rows);

        // 3. Read the data directly into temporary memory
        file.read(reinterpret_cast<char*>(kappa_idx.data()), num_rows * sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(th1_idx.data()), num_rows * sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(th2_idx.data()), num_rows * sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(dphi_idx.data()), num_rows * sizeof(uint16_t));
        file.read(reinterpret_cast<char*>(I_vals.data()), num_rows * sizeof(double));

        // 4. Allocate the dense 4D grid (initialized to 0.0)
        int total_size = N_K * N_T1 * N_T2 * N_DP;
        m_grid.assign(total_size, 0.0f);

        // 5. Populate the dense grid
        for (uint64_t i = 0; i < num_rows; ++i) {
            int flat_idx = get_flat_index(kappa_idx[i], th1_idx[i], th2_idx[i], dphi_idx[i]);
            // Convert double to float to save memory in the permanent grid
            if (flat_idx >= total_size) {
                throw std::runtime_error("Flat index out of bounds: " + std::to_string(flat_idx));
            }
            m_grid[flat_idx] = static_cast<float>(I_vals[i]); 
        }

        // Print the first few rows:
        if (verbose) {
            std::cout << "Loaded " << num_rows << " rows into a dense grid of size " << total_size << "." << std::endl;
            for (size_t i = 0; i < std::min(num_rows, uint64_t(5)); ++i) {
                std::cout << "Row " << i << ": kappa_idx=" << kappa_idx[i] 
                          << ", th1_idx=" << th1_idx[i] 
                          << ", th2_idx=" << th2_idx[i] 
                          << ", dphi_idx=" << dphi_idx[i] 
                          << ", I=" << I_vals[i] << std::endl;
            }
        }
    }

    inline float get_intensity(float theta1, float theta2, float phi1, float phi2, float vmf_kappa) const {
        // Ensure kappa non-negative
        vmf_kappa = std::max(vmf_kappa, 0.f);

        // Theta is a polar angle in [0, pi] -> clamp (do NOT modulo)
        theta1 = mitsuba::math::clamp(theta1, 0.0f, (float)M_PI);
        theta2 = mitsuba::math::clamp(theta2, 0.0f, (float)M_PI);

        // Normalize azimuths to [0, 2*pi)
        phi1 = std::fmod(phi1, 2.0f * M_PI);
        if (phi1 < 0.0f) phi1 += 2.0f * M_PI;
        phi2 = std::fmod(phi2, 2.0f * M_PI);
        if (phi2 < 0.0f) phi2 += 2.0f * M_PI;

        // 1. Calculate dphi (shortest distance around circle) in [0, pi]
        float dphi = std::abs(phi1 - phi2);
        if (dphi > M_PI) dphi = 2.0f * M_PI - dphi;

        // 2. Map physical values to fractional indices [0, N-1]
        // Kappa: inverse mapping of generation: fk in [0, N_K-1]
        float fk = (std::log10(vmf_kappa + 1.0f)) / 2.0f * (N_K - 1);
        fk = mitsuba::math::clamp(fk, 0.0f, (float)(N_K - 1));
        int ik = (int) std::floor(fk);

        // Reconstruct physical kappa bounds and compute a safe interpolation weight
        float tk;
        if (ik >= N_K - 1) {
            ik = N_K - 2;
            tk = 1.0f;
        } else {
            float k0 = -1.0f + std::pow(10.0f, (float)ik * 2.0f / (N_K - 1));
            float k1 = -1.0f + std::pow(10.0f, (float)(ik + 1) * 2.0f / (N_K - 1));
            tk = (vmf_kappa - k0) / (k1 - k0);
            tk = mitsuba::math::clamp(tk, 0.0f, 1.0f);
        }

        float ft1 = (theta1 / M_PI) * (N_T1 - 1);
        float ft2 = (theta2 / M_PI) * (N_T2 - 1);
        float fdp = (dphi / M_PI) * (N_DP - 1);

        // Clamp fractional indices to valid ranges and compute integer bases + weights
        ft1 = mitsuba::math::clamp(ft1, 0.0f, (float)(N_T1 - 1));
        ft2 = mitsuba::math::clamp(ft2, 0.0f, (float)(N_T2 - 1));
        fdp = mitsuba::math::clamp(fdp, 0.0f, (float)(N_DP - 1));

        int it1 = (int) std::floor(ft1); float tt1;
        if (it1 >= N_T1 - 1) { it1 = N_T1 - 2; tt1 = 1.0f; } else tt1 = ft1 - it1;

        int it2 = (int) std::floor(ft2); float tt2;
        if (it2 >= N_T2 - 1) { it2 = N_T2 - 2; tt2 = 1.0f; } else tt2 = ft2 - it2;

        int idp = (int) std::floor(fdp); float tdp;
        if (idp >= N_DP - 1) { idp = N_DP - 2; tdp = 1.0f; } else tdp = fdp - idp;

        // Now all base indices (ik,it1,it2,idp) are guaranteed to have +1 valid
        // Interpolate across the 4th dimension (dphi) for all 8 corners of the 3D cube
        float c000 = lerp(m_grid[get_flat_index(ik,   it1,   it2,   idp)],     m_grid[get_flat_index(ik,   it1,   it2,   idp+1)], tdp);
        float c001 = lerp(m_grid[get_flat_index(ik,   it1,   it2+1, idp)],     m_grid[get_flat_index(ik,   it1,   it2+1, idp+1)], tdp);
        float c010 = lerp(m_grid[get_flat_index(ik,   it1+1, it2,   idp)],     m_grid[get_flat_index(ik,   it1+1, it2,   idp+1)], tdp);
        float c011 = lerp(m_grid[get_flat_index(ik,   it1+1, it2+1, idp)],     m_grid[get_flat_index(ik,   it1+1, it2+1, idp+1)], tdp);
        float c100 = lerp(m_grid[get_flat_index(ik+1, it1,   it2,   idp)],     m_grid[get_flat_index(ik+1, it1,   it2,   idp+1)], tdp);
        float c101 = lerp(m_grid[get_flat_index(ik+1, it1,   it2+1, idp)],     m_grid[get_flat_index(ik+1, it1,   it2+1, idp+1)], tdp);
        float c110 = lerp(m_grid[get_flat_index(ik+1, it1+1, it2,   idp)],     m_grid[get_flat_index(ik+1, it1+1, it2,   idp+1)], tdp);
        float c111 = lerp(m_grid[get_flat_index(ik+1, it1+1, it2+1, idp)],     m_grid[get_flat_index(ik+1, it1+1, it2+1, idp+1)], tdp);

        // Interpolate across 3rd dimension (th2)
        float c00 = lerp(c000, c001, tt2);
        float c01 = lerp(c010, c011, tt2);
        float c10 = lerp(c100, c101, tt2);
        float c11 = lerp(c110, c111, tt2);

        // Interpolate across 2nd dimension (th1)
        float c0 = lerp(c00, c01, tt1);
        float c1 = lerp(c10, c11, tt1);

        // Interpolate across 1st dimension (kappa)
        return lerp(c0, c1, tk);
    }

    // ------------------------------------------------------------------------------------
    // Faster version:
    // Precompute the kappa index and weight (must be called once before evaluating multiple intensities for the same kappa)
    inline KappaQuery prepare_kappa(float vmf_kappa) const {
        vmf_kappa = std::max(vmf_kappa, 0.f);

        float fk = (std::log10(vmf_kappa + 1.0f)) / 2.0f * (N_K - 1);
        fk = math::clamp(fk, 0.0f, (float)(N_K - 1));
        int ik = (int) std::floor(fk);

        float tk;
        if (ik >= N_K - 1) {
            ik = N_K - 2;
            tk = 1.0f;
        } else {
            float k0 = -1.0f + std::pow(10.0f, (float)ik * 2.0f / (N_K - 1));
            float k1 = -1.0f + std::pow(10.0f, (float)(ik + 1) * 2.0f / (N_K - 1));
            tk = math::clamp((vmf_kappa - k0) / (k1 - k0), 0.0f, 1.0f);
        }
        return { ik, tk };
    }

    inline float get_intensity_fast(float theta1, float theta2, float dphi, const KappaQuery& kq) const {
        float ft1 = (theta1 / (float)M_PI) * (N_T1 - 1);
        float ft2 = (theta2 / (float)M_PI) * (N_T2 - 1);
        float fdp = (dphi   / (float)M_PI) * (N_DP - 1);

        // Branchless clamp and separate integer/fractional parts
        int it1 = std::min(std::max((int)ft1, 0), N_T1 - 2);
        int it2 = std::min(std::max((int)ft2, 0), N_T2 - 2);
        int idp = std::min(std::max((int)fdp, 0), N_DP - 2);

        float tt1 = std::min(std::max(ft1 - it1, 0.0f), 1.0f);
        float tt2 = std::min(std::max(ft2 - it2, 0.0f), 1.0f);
        float tdp = std::min(std::max(fdp - idp, 0.0f), 1.0f);

        // Calculate base pointer once
        int base_idx = kq.ik * stride_k + it1 * stride_t1 + it2 * stride_t2 + idp;
        const float* p0 = &m_grid[base_idx];
        const float* p1 = p0 + stride_k;

        // Lambda for fast 3D lerp using pointer offsets
        auto lerp3D = [tdp, tt2, tt1](const float* p) {
            float c00 = p[0]         + tdp * (p[1]         - p[0]);
            float c01 = p[stride_t2] + tdp * (p[stride_t2+1] - p[stride_t2]);
            float c10 = p[stride_t1] + tdp * (p[stride_t1+1] - p[stride_t1]);
            float c11 = p[stride_t1 + stride_t2] + tdp * (p[stride_t1 + stride_t2 + 1] - p[stride_t1 + stride_t2]);
            
            float c0 = c00 + tt2 * (c01 - c00);
            float c1 = c10 + tt2 * (c11 - c10);
            return c0 + tt1 * (c1 - c0);
        };

        float val0 = lerp3D(p0);
        float val1 = lerp3D(p1);

        return val0 + kq.tk * (val1 - val0);
    }
};

class VMFLUT {
private:
    IntensityGrid4D m_intensityGrid; // To ensure the grid is loaded and available
    bool m_initialized = false;
    std::string m_filepath; // Path to the LUT file (should match the one used during generation)


public:
    VMFLUT(const std::string& filepath="data/vmf_lut.bin", bool verbose = true) : m_filepath(filepath) {
        init(verbose);
    }
    inline void init(bool verbose=false) {
        // fs::path resolvedPath(m_filepath);
        fs::path resolvedPath;

        Thread *thread = Thread::getThread();
        if (thread) {
            FileResolver *fr = thread->getFileResolver();
            if (fr)
                resolvedPath = fr->resolve(m_filepath); // supports relative paths via resolver
        }

        if (!fs::exists(resolvedPath)) {
            SLog(EError, "VMF LUT file not found: '%s' (resolved to '%s')",
                 m_filepath.c_str(), resolvedPath.string().c_str());
            return; // EError already raises in Mitsuba
        }
        

        if (verbose)
            SLog(EInfo, "Loading VMF LUT from: %s", resolvedPath.string().c_str());

        // m_intensityGrid = new IntensityGrid4D(resolvedPath.string(), verbose);
        m_intensityGrid.load(resolvedPath.string(), verbose);
        m_initialized = true;
    }


    inline float eval(float theta1, float theta2, float phi1, float phi2, float vmf_kappa) const {
        if (!m_initialized)
            throw std::runtime_error("VMFLUT not initialized. Call init() before eval() and make sure the vmf_lut.bin file is available!!");
        return m_intensityGrid.get_intensity(theta1, theta2, phi1, phi2, vmf_kappa);
    }

    // Eval for v1 and v2 in LOCAL coordinates (where the mean direction is (0, 0, 1))
    // Returns int_Omega dot(v1, w) * dot(v2, w) * VMF(w; mu, kappa) dw, where w is the normal direction integrated over the sphere
    // and VMF(w; mu, kappa) is the von Mises-Fisher distribution with mean direction mu and concentration kappa.
    inline float eval(const Vector& v1, const Vector& v2, float vmf_kappa) const {
        const float z1 = math::clamp((float) v1.z, -1.0f, 1.0f);
        const float z2 = math::clamp((float) v2.z, -1.0f, 1.0f);

        float theta1 = std::acos(z1);
        float phi1   = std::atan2((float) v1.y, (float) v1.x);

        float theta2 = std::acos(z2);
        float phi2   = std::atan2((float) v2.y, (float) v2.x);

        return eval(theta1, theta2, phi1, phi2, vmf_kappa);
    }

    // Eval for v1 and v2 in GLOBAL coordinates, given the VMF mean direction (mu) in global coordinates.
    // We will need to rotate v1 and v2 to the local frame where mu is (0, 0, 1) before calling the eval above.
    inline float eval(const Vector& v1, const Vector& v2, const Vector& vmf_mu, float vmf_kappa) const {
        Vector mu = vmf_mu;
        if (mu.lengthSquared() <= Epsilon)
            mu = Vector(0.0f, 0.0f, 1.0f);
        mu = normalize(mu);

        Frame vmfFrame(mu);
        Vector v1_local = vmfFrame.toLocal(v1);
        Vector v2_local = vmfFrame.toLocal(v2);
        return eval(v1_local, v2_local, vmf_kappa); 
    }

    // ---------------------------------------------------------------------------------
    // Faster eval version if we already have the kappa query prepared:
    inline KappaQuery prepareKappa(float vmf_kappa) const {
        return m_intensityGrid.prepare_kappa(vmf_kappa);
    }

    // Fast eval taking local vectors and a precomputed kappa query
    inline float evalFast(const Vector& v1, const Vector& v2, const KappaQuery& kq) const {
        if (!m_initialized)
            throw std::runtime_error("VMFLUT not initialized!");

        float z1 = math::clamp((float)v1.z, -1.0f, 1.0f);
        float z2 = math::clamp((float)v2.z, -1.0f, 1.0f);

        float theta1 = std::acos(z1);
        float theta2 = std::acos(z2);

        // Fast dphi calculation without atan2 or fmod
        float dphi = 0.0f;
        float len2_1 = v1.x * v1.x + v1.y * v1.y;
        float len2_2 = v2.x * v2.x + v2.y * v2.y;
        
        if (len2_1 > 1e-8f && len2_2 > 1e-8f) {
            float dot_xy = v1.x * v2.x + v1.y * v2.y;
            float cos_dphi = math::clamp(dot_xy / std::sqrt(len2_1 * len2_2), -1.0f, 1.0f);
            dphi = std::acos(cos_dphi);
        }

        return m_intensityGrid.get_intensity_fast(theta1, theta2, dphi, kq);
    }
};  

MTS_NAMESPACE_END

#elif VMF_LUT_VERSION == 1
#pragma once

#include <iostream>
#include <cmath>
#include <algorithm>

#include <mitsuba/core/point.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/frame.h>

// Include the auto-generated weights
#include "vmf_mlp_weights.h"

MTS_NAMESPACE_BEGIN

struct KappaQuery {
    float exp_k; // Cached exponential for the MLP
};

class VMFLUT {
private:
    // Core MLP evaluation routine
    inline float evaluate_mlp(float exp_k, float cos_th1, float cos_th2, float cos_dphi) const {
        using namespace VMF_MLP;

        float inputs[4] = { exp_k, cos_th1, cos_th2, cos_dphi };
        float act[32];
        float next_act[32];

        // Layer 0
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sum = B0[j];
            for (int i = 0; i < 4; ++i) sum += inputs[i] * W0[i][j];
            act[j] = std::max(0.0f, sum);
        }

        // Layer 1
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sum = B1[j];
            for (int i = 0; i < 32; ++i) sum += act[i] * W1[i][j];
            next_act[j] = std::max(0.0f, sum);
        }

        // Layer 2
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sum = B2[j];
            for (int i = 0; i < 32; ++i) sum += next_act[i] * W2[i][j];
            act[j] = std::max(0.0f, sum);
        }

        // Output Layer
        float output = B3[0];
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            output += act[i] * W3[i][0];
        }

        return std::max(0.0f, output);
    }

public:
    // Constructor no longer needs file paths or initialization!
    VMFLUT(const std::string& filepath="", bool verbose = false) {
        if (verbose) {
            SLog(EInfo, "VMFLUT initialized using embedded MLP weights.");
        }
    }
    
    inline void init(bool verbose=false) { /* No-op */ }

    // Evaluates based on absolute angles (legacy format, keeps your old API intact)
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

        return evaluate_mlp(exp_k, std::cos(theta1), std::cos(theta2), std::cos(dphi));
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
    
    // Caches the exponential so we don't recalculate it inside integration loops
    inline KappaQuery prepareKappa(float vmf_kappa) const {
        vmf_kappa = std::max(vmf_kappa, 0.f);
        return { std::exp(-vmf_kappa / 10.0f) };
    }

    // Extremely fast eval: No acos() calls, pure SIMD-friendly math
    inline float evalFast(const Vector& v1, const Vector& v2, const KappaQuery& kq) const {
        float cos_th1 = math::clamp((float)v1.z, -1.0f, 1.0f);
        float cos_th2 = math::clamp((float)v2.z, -1.0f, 1.0f);

        float cos_dphi = 1.0f; // Default if vectors are parallel to Z
        float len2_1 = v1.x * v1.x + v1.y * v1.y;
        float len2_2 = v2.x * v2.x + v2.y * v2.y;
        
        if (len2_1 > 1e-8f && len2_2 > 1e-8f) {
            float dot_xy = v1.x * v2.x + v1.y * v2.y;
            cos_dphi = math::clamp(dot_xy / std::sqrt(len2_1 * len2_2), -1.0f, 1.0f);
        }

        return evaluate_mlp(kq.exp_k, cos_th1, cos_th2, cos_dphi);
    }
}; 

MTS_NAMESPACE_END

#elif VMF_LUT_VERSION == 2

#pragma once

#include <iostream>
#include <cmath>
#include <algorithm>

#include <mitsuba/core/point.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/frame.h>

// Include the auto-generated polynomial coefficients and powers
#include "vmf_poly_weights.h"

MTS_NAMESPACE_BEGIN

struct KappaQuery {
    float exp_k; // Cached exponential for the polynomial transformation
};

class VMFLUT {
private:
    // Core Polynomial evaluation routine
    inline float evaluate_poly(float exp_k, float cos_th1, float cos_th2, float cos_dphi) const {
        using namespace VMF_POLY;

        float inputs[4] = { exp_k, cos_th1, cos_th2, cos_dphi };
        
        // Start with the intercept (usually the first term with all powers = 0)
        float result = COEFFS[0];

        // Evaluate the remaining cross-terms
        #pragma unroll
        for (int i = 1; i < NUM_TERMS; ++i) {
            float term = COEFFS[i];
            
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int p = POWERS[i][j];
                if (p == 1)      term *= inputs[j];
                else if (p == 2) term *= (inputs[j] * inputs[j]);
                else if (p == 3) term *= (inputs[j] * inputs[j] * inputs[j]);
                else if (p == 4) term *= (inputs[j] * inputs[j] * inputs[j] * inputs[j]);
            }
            result += term;
        }

        // Polynomials can dip below zero at the domain boundaries.
        // We must clamp it to physically valid values (>= 0).
        return std::max(0.0f, result);
    }

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

        return evaluate_poly(exp_k, std::cos(theta1), std::cos(theta2), std::cos(dphi));
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

        return evaluate_poly(kq.exp_k, cos_th1, cos_th2, cos_dphi);
    }
}; 

MTS_NAMESPACE_END

#elif VMF_LUT_VERSION == 3
// Version 3: Tensor decomposition of the lut

#pragma once

#include <iostream>
#include <cmath>
#include <algorithm>

#include <mitsuba/core/point.h>
#include <mitsuba/core/matrix.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/frame.h>

// Include the auto-generated 1D tensor factors
#include "vmf_tensor_weights.h"

MTS_NAMESPACE_BEGIN

struct KappaQuery {
    // We pre-multiply the tensor weight and the interpolated kappa value!
    // This removes an entire dimension from the inner-loop evaluation.
    float precomputed_k[VMF_TENSOR::RANK]; 
};

class VMFLUT {
private:
    static constexpr int N_K  = 100;
    static constexpr int N_T1 = 40;
    static constexpr int N_T2 = 40;
    static constexpr int N_DP = 10;

public:
    // Constructor no longer needs file paths or grid allocations
    VMFLUT(const std::string& filepath="", bool verbose = true) {
        if (verbose) {
            SLog(EInfo, "VMFLUT initialized using Tensor CP Decomposition (Rank %d).", VMF_TENSOR::RANK);
        }
    }
    
    inline void init(bool verbose=false) { /* No-op */ }

    // Evaluates based on absolute angles (Legacy format)
    inline float eval(float theta1, float theta2, float phi1, float phi2, float vmf_kappa) const {
        KappaQuery kq = prepareKappa(vmf_kappa);

        // Normalize azimuths
        phi1 = std::fmod(phi1, 2.0f * (float)M_PI);
        if (phi1 < 0.0f) phi1 += 2.0f * (float)M_PI;
        phi2 = std::fmod(phi2, 2.0f * (float)M_PI);
        if (phi2 < 0.0f) phi2 += 2.0f * (float)M_PI;

        float dphi = std::abs(phi1 - phi2);
        if (dphi > M_PI) dphi = 2.0f * (float)M_PI - dphi;

        // Route to a modified fast-eval that takes absolute angles directly
        return evalAnglesFast(theta1, theta2, dphi, kq);
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
    
    // Caches the 1D interpolation for Kappa AND the tensor weighting factor
    inline KappaQuery prepareKappa(float vmf_kappa) const {
        vmf_kappa = std::max(vmf_kappa, 0.f);

        float fk = (std::log10(vmf_kappa + 1.0f)) / 2.0f * (N_K - 1);
        int ik = std::min(std::max((int)fk, 0), N_K - 2);
        
        float k0 = -1.0f + std::pow(10.0f, (float)ik * 2.0f / (N_K - 1));
        float k1 = -1.0f + std::pow(10.0f, (float)(ik + 1) * 2.0f / (N_K - 1));
        float tk = math::clamp((vmf_kappa - k0) / (k1 - k0), 0.0f, 1.0f);

        KappaQuery kq;
        #pragma unroll
        for (int r = 0; r < VMF_TENSOR::RANK; ++r) {
            float k_interp = VMF_TENSOR::KAPPA[ik][r] + tk * (VMF_TENSOR::KAPPA[ik+1][r] - VMF_TENSOR::KAPPA[ik][r]);
            kq.precomputed_k[r] = VMF_TENSOR::WEIGHTS[r] * k_interp;
        }
        return kq;
    }

    // Internal helper that evaluates using absolute angles
    inline float evalAnglesFast(float theta1, float theta2, float dphi, const KappaQuery& kq) const {
        float ft1 = (theta1 / (float)M_PI) * (N_T1 - 1);
        float ft2 = (theta2 / (float)M_PI) * (N_T2 - 1);
        float fdp = (dphi   / (float)M_PI) * (N_DP - 1);

        int it1 = std::min(std::max((int)ft1, 0), N_T1 - 2);
        int it2 = std::min(std::max((int)ft2, 0), N_T2 - 2);
        int idp = std::min(std::max((int)fdp, 0), N_DP - 2);

        float tt1 = std::min(std::max(ft1 - it1, 0.0f), 1.0f);
        float tt2 = std::min(std::max(ft2 - it2, 0.0f), 1.0f);
        float tdp = std::min(std::max(fdp - idp, 0.0f), 1.0f);

        float result = 0.0f;

        // Because VMF_TENSOR::RANK is a constexpr, the compiler will completely unroll this loop!
        #pragma unroll
        for (int r = 0; r < VMF_TENSOR::RANK; ++r) {
            float t1_val = VMF_TENSOR::TH1[it1][r] + tt1 * (VMF_TENSOR::TH1[it1+1][r] - VMF_TENSOR::TH1[it1][r]);
            float t2_val = VMF_TENSOR::TH2[it2][r] + tt2 * (VMF_TENSOR::TH2[it2+1][r] - VMF_TENSOR::TH2[it2][r]);
            float dp_val = VMF_TENSOR::DPHI[idp][r] + tdp * (VMF_TENSOR::DPHI[idp+1][r] - VMF_TENSOR::DPHI[idp][r]);
            
            // Multiply the interpolated dimensions with our pre-multiplied Kappa/Weight term
            result += kq.precomputed_k[r] * t1_val * t2_val * dp_val;
        }

        return std::max(0.0f, result);
    }

    // Fast Vector Eval
    inline float evalFast(const Vector& v1, const Vector& v2, const KappaQuery& kq) const {
        float z1 = math::clamp((float)v1.z, -1.0f, 1.0f);
        float z2 = math::clamp((float)v2.z, -1.0f, 1.0f);

        float theta1 = std::acos(z1);
        float theta2 = std::acos(z2);

        float dphi = 0.0f;
        float len2_1 = v1.x * v1.x + v1.y * v1.y;
        float len2_2 = v2.x * v2.x + v2.y * v2.y;
        
        if (len2_1 > 1e-8f && len2_2 > 1e-8f) {
            float dot_xy = v1.x * v2.x + v1.y * v2.y;
            float cos_dphi = math::clamp(dot_xy / std::sqrt(len2_1 * len2_2), -1.0f, 1.0f);
            dphi = std::acos(cos_dphi);
        }

        return evalAnglesFast(theta1, theta2, dphi, kq);
    }
}; 

MTS_NAMESPACE_END

#elif VMF_LUT_VERSION == 4

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


#endif


// Example usage:
// int main() {
//     IntensityGrid4D model("vmf_lut.bin");
    
//     // Example query
//     float intensity = model.get_intensity(0.0f, 0.0f, 0.0f, 0.0f, 100.0f);
//     std::vector<float> test_kappas = {0.0f, 1.0f, 10.0f, 100.0f, 200.f, 200000.f};
//     for (float k : test_kappas) {
//         intensity = model.get_intensity(0.0f, 0.0f, 0.0f, 0.0f, k);
//         std::cout << "Interpolated Intensity(kappa=" << k << "): " << intensity << std::endl;
//     }

//     intensity = model.get_intensity(10.0f, 10.0f, 10.0f, 10.0f, 20.f);
//     std::cout << "Interpolated Intensity(th1=10, th2=10, dphi=10, kappa=20): " << intensity << std::endl;

    
//     return 0;
// }
