"""
Fit the degree-4 polynomial that approximates the VMF cosine integral

    I(kappa, theta1, theta2, dphi) = int_S VMF(n | z, kappa) <n, w1>_+ <n, w2>_+ dn

tabulated by create_VMF_LUT.py (vmf_lut.feather), and export it as the C++
function used by the SamplingBVH (Eq. 14 of the paper, diffuse term).

The polynomial is fitted in the variables (exp(-kappa / 10), cos theta1,
cos theta2, cos dphi) with ridge regression, and written fully unrolled.

Usage (from the repository root):
    pip install -r scripts/LUT/requirements.txt
    python scripts/LUT/fit_LUT.py [--output include/mitsuba/render/hgs/vmf_poly_unrolled_deg4.h]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent.parent


def export_unrolled_polynomial(lut_path, output_path, degree=4):
    print(f"Loading {lut_path} ...")
    df = pd.read_feather(lut_path)

    # Fit in a space where the integral is close to polynomial; cos() matches
    # passing the z coordinate of the local directions in C++
    X = np.column_stack([
        np.exp(-df['kappa'].values / 10.0),
        np.cos(df['th1'].values),
        np.cos(df['th2'].values),
        np.cos(df['dphi'].values)
    ])
    y = df['I'].values

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)

    model = Ridge(alpha=1e-5)
    model.fit(X_poly, y)

    mse = np.mean((y - model.predict(X_poly)) ** 2)
    print(f"Degree {degree} polynomial | MSE: {mse:.6f} | coefficients: {len(model.coef_)}")

    # Fully unrolled C++ code
    variables = ["x", "y", "z", "w"]
    cpp_code = "#pragma once\n#include <algorithm>\n\n"
    cpp_code += "inline float evaluate_vmf_poly(float x, float y, float z, float w) {\n"
    cpp_code += f"    float out = {model.intercept_:.6f}f;\n"

    for i in range(1, len(model.coef_)):
        term = f"{model.coef_[i]:.6f}f"
        for j in range(4):
            p = poly.powers_[i][j]
            if p > 0:
                term += "".join([f" * {variables[j]}"] * p)
        cpp_code += f"    out += {term};\n"

    cpp_code += "    return std::max(0.0f, out);\n"
    cpp_code += "}\n"

    with open(output_path, "w") as f:
        f.write(cpp_code)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lut", default=str(THIS_DIR / "vmf_lut.feather"), help="table written by create_VMF_LUT.py")
    ap.add_argument("--output", default=str(REPO_ROOT / "include/mitsuba/render/hgs/vmf_poly_unrolled_deg4.h"))
    ap.add_argument("--degree", type=int, default=4)
    args = ap.parse_args()
    export_unrolled_polynomial(args.lut, args.output, args.degree)
