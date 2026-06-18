"""
Requirements:
    pip install pandas pyarrow numpy tensorly
"""
import numpy as np
import pandas as pd
import tensorly as tl
from tensorly.decomposition import parafac

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


from pathlib import Path
import os

def train_and_export_mlp():
    print("Loading data...")
    this_dir = Path(__file__).parent
    df = pd.read_feather(this_dir / "vmf_lut.feather")
    

    print("Preparing features...")
    # Transform to linear-friendly space.
    # Note: We use cos() here, which matches passing z directly in C++
    X = np.column_stack([
        np.exp(-df['kappa'].values / 10.0),
        np.cos(df['th1'].values),
        np.cos(df['th2'].values),
        np.cos(df['dphi'].values)
    ])
    y = df['I'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 3 layers of 32 neurons is a good balance of speed and accuracy
    print("Training MLP (this might take a few minutes)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 32, 32), 
        activation='relu', 
        max_iter=500, 
        random_state=42,
        tol=1e-5
    )
    mlp.fit(X_train, y_train)
    
    score = mlp.score(X_test, y_test)
    mse = np.mean((y_test - mlp.predict(X_test))**2)
    print(f"MLP Test R^2: {score:.4f} | Test MSE: {mse:.6f}")
    
    # Export to C++ Header
    filename = this_dir / "vmf_mlp_weights.h"
    print(f"Exporting weights to {filename}...")
    with open(filename, 'w') as f:
        f.write("#pragma once\n\n")
        f.write("namespace VMF_MLP {\n")
        
        for layer_idx, (weights, biases) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            f.write(f"    const float W{layer_idx}[{weights.shape[0]}][{weights.shape[1]}] = {{\n")
            for row in weights:
                f.write("        {" + ", ".join(f"{val}f" for val in row) + "},\n")
            f.write("    };\n")
            
            f.write(f"    const float B{layer_idx}[{biases.shape[0]}] = {{")
            f.write(", ".join(f"{val}f" for val in biases) + "};\n\n")
            
        f.write("}\n")
    print("Done!")

def fit_polynomial(degree=4):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge

    print("Loading data...")
    this_dir = Path(__file__).parent
    df = pd.read_feather(this_dir / "vmf_lut.feather")
    filename = this_dir / "vmf_poly_weights.h"
    

    print("Preparing features...")
    # Transform to linear-friendly space.
    # Note: We use cos() here, which matches passing z directly in C++
    X = np.column_stack([
        np.exp(-df['kappa'].values / 10.0),
        np.cos(df['th1'].values),
        np.cos(df['th2'].values),
        np.cos(df['dphi'].values)
    ])
    y = df['I'].values
    

    # 1. Generate the 70 polynomial cross-terms
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)

    # 2. Fit the model (Ridge regression helps prevent wild coefficients)
    model = Ridge(alpha=1e-5)
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    mse = np.mean((y - y_pred)**2)
    print(f"Polynomial Degree {degree} | MSE: {mse:.6f} | Coeffs: {len(model.coef_)}")

    # 3. Export to C++
    with open(filename, "w") as f:
        f.write("#pragma once\n\nnamespace VMF_POLY {\n")
        f.write(f"    const int NUM_TERMS = {len(model.coef_)};\n")
        
        # Export coefficients
        f.write("    const float COEFFS[] = {\n")
        for c in model.coef_:
            f.write(f"        {c}f,\n")
        f.write("    };\n\n")
        
        # Export the power combinations for each term (e.g., x^2 * y^1 * z^0 * w^0)
        f.write("    const int POWERS[][4] = {\n")
        for powers in poly.powers_:
            f.write(f"        {{{powers[0]}, {powers[1]}, {powers[2]}, {powers[3]}}},\n")
        f.write("    };\n")
        f.write("}\n")

def export_unrolled_polynomial(degree=4):
    print("Loading data...")
    this_dir = Path(__file__).parent
    df = pd.read_feather(this_dir / "vmf_lut.feather")
    filename = this_dir / "vmf_poly_unrolled.h"
    

    print("Preparing features...")
    # Transform to linear-friendly space.
    # Note: We use cos() here, which matches passing z directly in C++
    X = np.column_stack([
        np.exp(-df['kappa'].values / 10.0),
        np.cos(df['th1'].values),
        np.cos(df['th2'].values),
        np.cos(df['dphi'].values)
    ])
    y = df['I'].values
    
    # 2. Fit Degree 3 (35 terms - incredibly fast)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X)
    
    model = Ridge(alpha=1e-5)
    model.fit(X_poly, y)

    # Evaluate MSE on the training data (just to get a sense of fit quality)
    y_pred = model.predict(X_poly)
    mse = np.mean((y - y_pred)**2)
    print(f"Degree {degree} Polynomial | MSE: {mse:.6f} | Coeffs: {len(model.coef_)}")
    
    # 3. Generate fully unrolled C++ code
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
    
    with open(filename, "w") as f:
        f.write(cpp_code)
        
        
    print("Unrolled polynomial exported to vmf_poly_unrolled.h")


def export_tensor_decomposition(rank=8):
    print("Loading data...")
    this_dir = Path(__file__).parent
    df = pd.read_feather(this_dir / "vmf_lut.feather")
    
    
    # Ensure the data is sorted exactly as the loops generated it
    df = df.sort_values(by=['kappa', 'th1', 'th2', 'dphi'])
    
    # Reshape back into the dense 4D grid
    print("Reshaping to dense grid...")
    grid = df['I'].values.reshape((100, 40, 40, 10))
    
    print(f"Running CP Decomposition (Rank {rank})...")
    # Using 'svd' init is deterministic and usually finds a better fit
    weights, factors = parafac(grid, rank=rank, init='svd', tol=1e-6)
    
    # Check accuracy
    reconstructed = tl.cp_to_tensor((weights, factors))
    mse = np.mean((grid - reconstructed)**2)
    print(f"Tensor MSE: {mse:.6f}")
    
    # Export to C++
    filename = this_dir / "vmf_tensor_weights.h"
    names = ["KAPPA", "TH1", "TH2", "DPHI"]
    
    print(f"Exporting to {filename}...")
    with open(filename, 'w') as f:
        f.write("#pragma once\n\nnamespace VMF_TENSOR {\n")
        f.write(f"    constexpr int RANK = {rank};\n\n")
        
        # Weights
        f.write(f"    const float WEIGHTS[{rank}] = {{\n        ")
        f.write(", ".join(f"{w:.6f}f" for w in weights))
        f.write("\n    };\n\n")
        
        # 1D Factor Matrices
        for i, factor_mat in enumerate(factors):
            name = names[i]
            N = factor_mat.shape[0]
            f.write(f"    const float {name}[{N}][RANK] = {{\n")
            for row in factor_mat:
                f.write("        {" + ", ".join(f"{val:.6f}f" for val in row) + "},\n")
            f.write("    };\n\n")
            
        f.write("}\n")
    print("Done!")

if __name__ == "__main__":
    # train_and_export_mlp()
    # fit_polynomial()
    # export_tensor_decomposition()
    export_unrolled_polynomial()

