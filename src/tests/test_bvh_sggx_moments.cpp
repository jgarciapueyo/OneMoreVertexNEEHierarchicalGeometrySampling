/*
    Test: area-weighted dominance in normal second-moment aggregation (SGGX-ready)

    Scenario:
      - One huge triangle facing +Z
      - One tiny triangle facing +X

    Expectation:
      - Aggregated second moment m2 should be dominated by the +Z triangle:
          m2_zz ≈ A_big / (A_big + A_small)
          m2_xx ≈ A_small / (A_big + A_small)
      - Trace(m2) ≈ 1 (since m2 is an expectation of ||n||^2 for unit normals)
      - Mean direction should be close to +Z
*/

#include <mitsuba/render/bvh/bvh.h>
#include <cassert>
#include <cmath>
#include <iostream>

MTS_NAMESPACE_BEGIN

static bool approx(Float a, Float b, Float eps = (Float) 1e-5f) {
    return std::abs(a - b) <= eps;
}

int run_sggx_moments_test() {
    const Float A_big   = (Float) 100.0f;
    const Float A_small = (Float) 1.0f;
    const Float A_total = A_big + A_small;

    const Float w_big   = A_big / A_total;
    const Float w_small = A_small / A_total;

    BVHNodeInfo big;
    big.surfaceArea = A_big;
    big.s_diffuse_albedo = Spectrum(1.0f) * A_big;
    big.centerOfMass = Point(0.0f, 0.0f, 0.0f);
    big.normalDistribution.setFromNormal(Vector(0.0f, 0.0f, 1.0f));
    big.valid = true;

    BVHNodeInfo small;
    small.surfaceArea = A_small;
    small.s_diffuse_albedo = Spectrum(1.0f) * A_small;
    small.centerOfMass = Point(1.0f, 0.0f, 0.0f);
    small.normalDistribution.setFromNormal(Vector(1.0f, 0.0f, 0.0f));
    small.valid = true;

    BVHNodeInfo merged;
    merged.valid = true;
    merged.surfaceArea = A_total;
    merged.s_diffuse_albedo = big.s_diffuse_albedo + small.s_diffuse_albedo;
    merged.centerOfMass  = w_big * big.centerOfMass  + w_small * small.centerOfMass;
    merged.normalDistribution.m1 = w_big * big.normalDistribution.m1 + w_small * small.normalDistribution.m1;
    merged.normalDistribution.m2 = w_big * big.normalDistribution.m2 + w_small * small.normalDistribution.m2;

    const Matrix3x3 &m2 = merged.normalDistribution.m2;

    assert(approx(m2.m[2][2], w_big,   (Float) 1e-5f));
    assert(approx(m2.m[0][0], w_small, (Float) 1e-5f));

    const Float tr = m2.m[0][0] + m2.m[1][1] + m2.m[2][2];
    assert(approx(tr, (Float) 1.0f, (Float) 1e-5f));

    const Vector meanN = merged.normalDistribution.meanDirection();
    assert(dot(meanN, Vector(0.0f, 0.0f, 1.0f)) > (Float) 0.99f);

    std::cout << "test_bvh_sggx_moments: OK\n";
    return 0;
}

MTS_NAMESPACE_END

// Global main required by the linker
int main(int argc, char **argv) {
    return mitsuba::run_sggx_moments_test();
}