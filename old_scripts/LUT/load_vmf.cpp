#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>

#include <algorithm>

#define M_PI 3.14159265358979323846

#if 0
// --- Python Math Equivalents ---

// Equivalent to np.linspace(start, stop, num)
std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> res(num);
    double step = (stop - start) / (num - 1);
    for(int i = 0; i < num; ++i) {
        res[i] = start + i * step;
    }
    return res;
}

// Equivalent to -1 + np.logspace(start, stop, num) (base 10)
std::vector<double> logspace_offset(double start, double stop, int num, double offset) {
    std::vector<double> res(num);
    double step = (stop - start) / (num - 1);
    for(int i = 0; i < num; ++i) {
        // std::pow(base, exponent)
        res[i] = offset + std::pow(10.0, start + i * step);
    }
    return res;
}


// --- Main Data Loader ---

int main() {
    // 1. Reconstruct the parameter grids once
    std::vector<double> kappa_vals = logspace_offset(0.0, 2.0, 100, -1.0);
    std::vector<double> th1_vals   = linspace(0.0, M_PI, 40);
    std::vector<double> th2_vals   = linspace(0.0, M_PI, 40);
    // Assuming dphi was linspace(0.0, M_PI, 10) based on your sample
    std::vector<double> dphi_vals  = linspace(0.0, M_PI, 10); 

    // 2. Open the binary file
    std::ifstream file("./vmf_lut.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    // 3. Read number of rows
    uint64_t num_rows;
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(uint64_t));

    // 4. Allocate vectors for the columns
    std::vector<uint16_t> kappa_idx(num_rows);
    std::vector<uint16_t> th1_idx(num_rows);
    std::vector<uint16_t> th2_idx(num_rows);
    std::vector<uint16_t> dphi_idx(num_rows);
    std::vector<double> I_vals(num_rows);

    // 5. Read the data directly into memory
    file.read(reinterpret_cast<char*>(kappa_idx.data()), num_rows * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(th1_idx.data()), num_rows * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(th2_idx.data()), num_rows * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(dphi_idx.data()), num_rows * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(I_vals.data()), num_rows * sizeof(double));

    // --- Example Usage ---
    std::cout << "Loaded " << num_rows << " rows." << std::endl;
    
    // Look up the actual float values for row 819999
    size_t test_row = 0;//num_rows - 1;
    std::cout << "Row " << test_row << " data:" << std::endl;
    std::cout << "kappa: " << kappa_vals[kappa_idx[test_row]] << std::endl;
    std::cout << "th1: "   << th1_vals[th1_idx[test_row]] << std::endl;
    std::cout << "th2: "   << th2_vals[th2_idx[test_row]] << std::endl;
    std::cout << "dphi: "  << dphi_vals[dphi_idx[test_row]] << std::endl;
    std::cout << "I: "     << I_vals[test_row] << std::endl;

    return 0;
}

#else


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
    
    // Grid dimensions (Adjust these to match your exact Python generation)
    const int N_K = 100;
    const int N_T1 = 40;
    const int N_T2 = 40;
    const int N_DP = 10; 

    // Helper for linear interpolation
    inline float lerp(float a, float b, float t) const {
        return a + t * (b - a);
    }

    // Helper to map 4D indices to the flat 1D array index
    inline int get_flat_index(int ik, int it1, int it2, int idp) const {
        return ik * (N_T1 * N_T2 * N_DP) + 
               it1 * (N_T2 * N_DP) + 
               it2 * N_DP + 
               idp;
    }

public:
    IntensityGrid4D(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file");
        
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
            m_grid[flat_idx] = static_cast<float>(I_vals[i]); 
        }

        // Print the first few rows:
        std::cout << "Loaded " << num_rows << " rows into a dense grid of size " << total_size << "." << std::endl;
        for (size_t i = 0; i < std::min(num_rows, uint64_t(5)); ++i) {
            std::cout << "Row " << i << ": kappa_idx=" << kappa_idx[i] 
                      << ", th1_idx=" << th1_idx[i] 
                      << ", th2_idx=" << th2_idx[i] 
                      << ", dphi_idx=" << dphi_idx[i] 
                      << ", I=" << I_vals[i] << std::endl;
        }
    }

    float get_intensity(float theta1, float theta2, float phi1, float phi2, float vmf_kappa) const {
        // apply modulos to angles to ensure they are within [0, 2*pi]
        theta1 = std::fmod(theta1, 2.0f * M_PI);
        theta2 = std::fmod(theta2, 2.0f * M_PI);
        phi1 = std::fmod(phi1, 2.0f * M_PI);
        phi2 = std::fmod(phi2, 2.0f * M_PI);
        
        // 1. Calculate dphi (assuming shortest distance around circle)
        float dphi = std::abs(phi1 - phi2);
        if (dphi > M_PI) dphi = 2.0f * M_PI - dphi; 

        // 2. Map physical values to fractional indices [0, N-1]
        // Kappa: reverse of -1 + 10^x from 0 to 2
        // 1. Find the integer index (ik) using the inverse log function (O(1) lookup)
        float fk = (std::log10(vmf_kappa + 1.0f)) / 2.0f * (N_K - 1);

        // Clamp to prevent out-of-bounds reading
        fk = std::clamp(fk, 0.0f, (float)N_K - 1.001f);
        int ik = std::floor(fk);

        // 2. Reconstruct the physical kappa values at index ik and ik+1
        // Formula: kappa = -1 + 10^(i * 2.0 / 99)
        float k0 = -1.0f + std::pow(10.0f, (float)ik * 2.0f / (N_K - 1));
        float k1 = -1.0f + std::pow(10.0f, (float)(ik + 1) * 2.0f / (N_K - 1));

        // 3. Calculate the true linear interpolation weight in physical space
        // t = (value - min) / (max - min)
        float tk = (vmf_kappa - k0) / (k1 - k0);
        tk = std::clamp(tk, 0.0f, 1.0f); // Ensure tk is in [0, 1]

        float ft1 = (theta1 / M_PI) * (N_T1 - 1);
        float ft2 = (theta2 / M_PI) * (N_T2 - 1);
        float fdp = (dphi / M_PI) * (N_DP - 1); // Assuming dphi max is PI

        // Clamp fractional indices to valid ranges to prevent out-of-bounds
        fk  = std::clamp(fk,  0.0f, (float)N_K - 1.001f);
        ft1 = std::clamp(ft1, 0.0f, (float)N_T1 - 1.001f);
        ft2 = std::clamp(ft2, 0.0f, (float)N_T2 - 1.001f);
        fdp = std::clamp(fdp, 0.0f, (float)N_DP - 1.001f);

        // 3. Get integer base indices and interpolation weights (t)
        int it1 = std::floor(ft1); float tt1 = ft1 - it1;
        int it2 = std::floor(ft2); float tt2 = ft2 - it2;
        int idp = std::floor(fdp); float tdp = fdp - idp;

        // 4. Interpolate across the 4th dimension (dphi) for all 8 corners of the 3D cube
        float c000 = lerp(m_grid[get_flat_index(ik,   it1,   it2,   idp)], m_grid[get_flat_index(ik,   it1,   it2,   idp+1)], tdp);
        float c001 = lerp(m_grid[get_flat_index(ik,   it1,   it2+1, idp)], m_grid[get_flat_index(ik,   it1,   it2+1, idp+1)], tdp);
        float c010 = lerp(m_grid[get_flat_index(ik,   it1+1, it2,   idp)], m_grid[get_flat_index(ik,   it1+1, it2,   idp+1)], tdp);
        float c011 = lerp(m_grid[get_flat_index(ik,   it1+1, it2+1, idp)], m_grid[get_flat_index(ik,   it1+1, it2+1, idp+1)], tdp);
        float c100 = lerp(m_grid[get_flat_index(ik+1, it1,   it2,   idp)], m_grid[get_flat_index(ik+1, it1,   it2,   idp+1)], tdp);
        float c101 = lerp(m_grid[get_flat_index(ik+1, it1,   it2+1, idp)], m_grid[get_flat_index(ik+1, it1,   it2+1, idp+1)], tdp);
        float c110 = lerp(m_grid[get_flat_index(ik+1, it1+1, it2,   idp)], m_grid[get_flat_index(ik+1, it1+1, it2,   idp+1)], tdp);
        float c111 = lerp(m_grid[get_flat_index(ik+1, it1+1, it2+1, idp)], m_grid[get_flat_index(ik+1, it1+1, it2+1, idp+1)], tdp);

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
};

int main() {
    IntensityGrid4D model("vmf_lut.bin");
    
    // Example query
    float intensity = model.get_intensity(0.0f, 0.0f, 0.0f, 0.0f, 100.0f);
    std::vector<float> test_kappas = {0.0f, 1.0f, 10.0f, 100.0f, 200.f, 200000.f};
    for (float k : test_kappas) {
        intensity = model.get_intensity(0.0f, 0.0f, 0.0f, 0.0f, k);
        std::cout << "Interpolated Intensity(kappa=" << k << "): " << intensity << std::endl;
    }

    intensity = model.get_intensity(10.0f, 10.0f, 10.0f, 10.0f, 20.f);
    std::cout << "Interpolated Intensity(th1=10, th2=10, dphi=10, kappa=20): " << intensity << std::endl;

    
    return 0;
}

#endif