"""
Given the integral:
    I(kappa, cos1, cos2) = int_{Omega} VMF(kappa) * cos(th1) * cos(th2) / pi dOmega

Where VMF(kappa) is the VMF distribution centered around the normal direction, and cos(th1), cos(th2) are the cosines of the angles between the sampled direction and the two given directions (e.g., light and view).

"""

import scipy
from scipy.stats import vonmises_fisher

from tqdm import tqdm
import numpy as np
import pandas as pd



def compute_vmf_lut(kappa_values, cos1_values, cos2_values, n_samples=1000):
    lut = np.zeros((len(kappa_values), len(cos1_values), len(cos2_values)), dtype=np.float32)
    
    for i, kappa in enumerate(tqdm(kappa_values, desc="Kappa")):
        # Sample directions from the VMF distribution
        if kappa == 0: # if 0, scipy complains
            vmf_samples = scipy.stats.uniform_direction.rvs(dim=3, size=n_samples)  # Uniformly sample directions
        else:
            vmf_samples = vonmises_fisher.rvs(mu=[0, 0, 1], kappa=kappa, size=n_samples)
        
        for j, cos1 in enumerate(cos1_values):
            for k, cos2 in enumerate(cos2_values):
                # Compute the contribution for each sample
                contributions = (vmf_samples[:, 2] * cos1 * cos2) / np.pi  # cos(th) is the z-component
                lut[i, j, k] = contributions.mean()
    
    return lut
def visualize_lut(df):
    import polyscope as ps
    # 1. Determine the dimensions (how many unique values in each axis)
    # This assumes your CSV is a tidy 'tile/repeat' format
    dim_k = len(df['kappa'].unique())
    dim_c1 = len(df['cos1'].unique())
    dim_c2 = len(df['cos2'].unique())
    
    node_dims = (dim_k, dim_c1, dim_c2)
    
    # 2. Define the bounds [min_x, min_y, min_z] to [max_x, max_y, max_z]
    # bound_low = (df['kappa'].min(), df['cos1'].min(), df['cos2'].min())
    # bound_high = (df['kappa'].max(), df['cos1'].max(), df['cos2'].max())
    bound_low = (0, df['cos1'].min(), df['cos2'].min())
    bound_high = (1, df['cos1'].max(), df['cos2'].max())

    ps.init()

    # 3. Register the grid structure
    # Note: We pass the SHAPE of the grid, not the data points
    ps_grid = ps.register_volume_grid("VMF LUT Grid", node_dims, bound_low, bound_high)
    
    # 4. Add the values
    values = df["I"].values.astype(np.float32).reshape(node_dims)
    print("values shape: ", values.shape)
    ps_grid.add_scalar_quantity("Intensity", values, defined_on='nodes', enabled=True)

    ps.set_ground_plane_mode("shadow_only")
    ps.set_build_gui(True)
    ps.show()
    
    
if __name__ == "__main__":
    kappa_values = -1+np.logspace(0, 2, 100)  # Example kappa values
    # cos1_values = np.linspace(0, 1, 10)     # Example cos(th1) values
    # cos2_values = np.linspace(0, 1, 10)     # Example cos(th2) values
    cos1_values = np.linspace(0, 1, 10)     # Example cos(th1) values
    cos2_values = np.linspace(0, 1, 10)     # Example cos(th2) values
    
    lut = compute_vmf_lut(kappa_values, cos1_values, cos2_values)
    
    # Save the LUT to a CSV file
    df = pd.DataFrame(lut.reshape(-1), columns=["I"])
    df["kappa"] = np.repeat(kappa_values, len(cos1_values) * len(cos2_values))
    df["cos1"] = np.tile(np.repeat(cos1_values, len(cos2_values)), len(kappa_values))
    df["cos2"] = np.tile(cos2_values, len(kappa_values) * len(cos1_values))
    
    df.to_csv("vmf_lut.csv", index=False)

    visualize_lut(df)
