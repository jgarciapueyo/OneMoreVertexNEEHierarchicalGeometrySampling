#pragma once
#include <algorithm>

inline float evaluate_vmf_poly(float x, float y, float z, float w) {
    float out = -0.035488f;
    out += 0.109271f * x;
    out += 0.088874f * y;
    out += 0.088875f * z;
    out += 0.021495f * w;
    out += -0.022808f * x * x;
    out += 0.055722f * x * y;
    out += 0.055777f * x * z;
    out += 0.010910f * x * w;
    out += 0.134595f * y * y;
    out += 0.248984f * y * z;
    out += 0.003491f * y * w;
    out += 0.134606f * z * z;
    out += 0.003486f * z * w;
    out += -0.000756f * w * w;
    out += -0.004882f * x * x * x;
    out += -0.194871f * x * x * y;
    out += -0.194954f * x * x * z;
    out += 0.021082f * x * x * w;
    out += -0.122624f * x * y * y;
    out += -0.158948f * x * y * z;
    out += 0.005397f * x * y * w;
    out += -0.122648f * x * z * z;
    out += 0.005415f * x * z * w;
    out += 0.006748f * x * w * w;
    out += 0.000013f * y * y * y;
    out += 0.124099f * y * y * z;
    out += -0.019071f * y * y * w;
    out += 0.124090f * y * z * z;
    out += 0.006286f * y * z * w;
    out += 0.000001f * y * w * w;
    out += 0.000003f * z * z * z;
    out += -0.019069f * z * z * w;
    out += 0.000002f * z * w * w;
    out += -0.000012f * w * w * w;
    return std::max(0.0f, out);
}
