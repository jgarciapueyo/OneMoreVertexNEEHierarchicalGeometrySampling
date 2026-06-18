"""
Requirements:
    pip install pandas pyarrow polyscope scipy tqdm numpy 

Given the integral:
    I(kappa, cos1, cos2) = int_{Omega} VMF(kappa) * cos(th1) * cos(th2) dOmega

Where VMF(kappa) is the VMF distribution centered around the normal direction, and cos(th1), cos(th2) are the cosines of the angles between the sampled direction and the two given directions (e.g., light and view).

"""

import sys

import scipy
from scipy.stats import vonmises_fisher

from tqdm import tqdm
import numpy as np
import pandas as pd

class Vector:
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

    def to_carthesian(self):
        x = np.sin(self.theta) * np.cos(self.phi)
        y = np.sin(self.theta) * np.sin(self.phi)
        z = np.cos(self.theta)
        return np.array([x, y, z])
    
    def cosine(self, other):
        v1 = self.to_carthesian()
        v2 = other.to_carthesian()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compute_vmf_lut(kappa_values, phi1, theta1, phi2, theta2, n_samples=1000):
    # Match DataFrame order: (kappa, theta1, phi1, theta2, phi2)
    lut = np.zeros(
        (len(kappa_values), len(theta1), len(phi1), len(theta2), len(phi2)),
        dtype=np.float32
    )

    for i, kappa in enumerate(tqdm(kappa_values, desc="Kappa")):
        if kappa == 0:
            vmf_samples = scipy.stats.uniform_direction.rvs(dim=3, size=n_samples)
        else:
            vmf_samples = vonmises_fisher.rvs(mu=[0, 0, 1], kappa=kappa, size=n_samples)

        for k in range(len(theta1)):
            for j in range(len(phi1)):
                w1 = Vector(theta1[k], phi1[j])
                w1c = w1.to_carthesian()

                for m in range(len(theta2)):
                    for l in range(len(phi2)):
                        w2 = Vector(theta2[m], phi2[l])
                        w2c = w2.to_carthesian()

                        cos1 = vmf_samples @ w1c
                        cos2 = vmf_samples @ w2c

                        # If cosine terms should be individually clamped:
                        contributions = np.clip(cos1, 0, np.inf) * np.clip(cos2, 0, np.inf)

                        lut[i, k, j, m, l] = contributions.mean()

    return lut
def spherical_to_cartesian(theta, phi):
    """ Converts spherical (theta, phi) to a (1, 3) Cartesian numpy array.
        Assumes Z-up: theta is angle from Z-axis, phi is azimuth in X-Y plane.
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([[x, y, z]])

def view_example_integrals(df):
    import polyscope as ps
    # 1. Pre-compute unique coordinates and reshape data for fast indexing
    coords = {
        'kappa': sorted(df['kappa'].unique()),
        'th1':   sorted(df['th1'].unique()),
        'phi1':  sorted(df['phi1'].unique()),
        'th2':   sorted(df['th2'].unique()),
        'phi2':  sorted(df['phi2'].unique())
    }
    node_dims = tuple(len(coords[k]) for k in coords)
    
    df_sorted = df.sort_values(['kappa', 'th1', 'phi1', 'th2', 'phi2'])
    full_data = df_sorted["I"].values.astype(np.float32).reshape(node_dims)

    # 2. Setup Polyscope and origin point
    ps.init()
    # Ensure point cloud is registered before the callback runs
    pc = ps.register_point_cloud("Origin", np.array([[0, 0, 0]]), color=(0, 0, 0))

    # 3. Setup state dictionary
    state = {
        "idx_kappa": 0,
        "idx_theta1": 0,
        "idx_phi1": 0,
        "idx_theta2": 0,
        "idx_phi2": 0,
        "needs_update": True
    }

    def callback():
        # Render the UI sliders
        changed_k, state["idx_kappa"] = ps.imgui.SliderInt("Kappa Index", state["idx_kappa"], 0, node_dims[0] - 1)
        changed_t1, state["idx_theta1"] = ps.imgui.SliderInt("Theta1 Index", state["idx_theta1"], 0, node_dims[1] - 1)
        changed_p1, state["idx_phi1"] = ps.imgui.SliderInt("Phi1 Index", state["idx_phi1"], 0, node_dims[2] - 1)
        changed_t2, state["idx_theta2"] = ps.imgui.SliderInt("Theta2 Index", state["idx_theta2"], 0, node_dims[3] - 1)
        changed_p2, state["idx_phi2"] = ps.imgui.SliderInt("Phi2 Index", state["idx_phi2"], 0, node_dims[4] - 1)

        # Update logic
        if any([changed_k, changed_t1, changed_p1, changed_t2, changed_p2, state["needs_update"]]):
            
            # Fetch the actual angles based on slider indices
            th1 = coords['th1'][state["idx_theta1"]]
            phi1 = coords['phi1'][state["idx_phi1"]]
            th2 = coords['th2'][state["idx_theta2"]]
            phi2 = coords['phi2'][state["idx_phi2"]]

            # Convert to Cartesian (1, 3) arrays
            v1_cart = spherical_to_cartesian(th1, phi1)
            v2_cart = spherical_to_cartesian(th2, phi2)

            # Update the vectors dynamically in the viewport
            # Colors are in [0, 1] range!
            pc.add_vector_quantity("w1", v1_cart, color=(1, 0, 0), enabled=True)
            pc.add_vector_quantity("w2", v2_cart, color=(0, 1, 0), enabled=True)

            # Fetch and print intensity from the 5D grid
            val = full_data[state["idx_kappa"], state["idx_theta1"], state["idx_phi1"], state["idx_theta2"], state["idx_phi2"]]
            
            # Display the intensity in the UI for convenience
            ps.imgui.Text(f"Current Intensity: {val:.6f}")

            state["needs_update"] = False

    # 4. Finalize and run
    ps.set_user_callback(callback)
    ps.set_ground_plane_mode("shadow_only")
    ps.show()

def visualize_lut(df):
    import polyscope as ps
    # 1. Extract dimensions and unique values
    # It is crucial that the unique values are sorted to match the grid layout
    coords = {
        'kappa': sorted(df['kappa'].unique()),
        'th1':   sorted(df['th1'].unique()),
        'phi1':  sorted(df['phi1'].unique()),
        'th2':   sorted(df['th2'].unique()),
        'phi2':  sorted(df['phi2'].unique())
    }
    
    node_dims = tuple(len(coords[k]) for k in coords)
    
    # 2. Reshape the intensity data into a 5D hypercube
    # Ensure the dataframe is sorted by the same axes order to match reshape
    df_sorted = df.sort_values(['kappa', 'th1', 'phi1', 'th2', 'phi2'])
    full_data = df_sorted["I"].values.astype(np.float32).reshape(node_dims)

    # 3. Setup Polyscope
    ps.init()
    
    # Define the 3D grid bounds (we will visualize th1, phi1, th2)
    grid_shape = (node_dims[0], node_dims[1], node_dims[2]) # th1, phi1, th2
    low_bound = (coords['kappa'][0], coords['th1'][0], coords['phi1'][0])
    high_bound = (coords['kappa'][-1], coords['th1'][-1], coords['phi1'][-1])
    
    low_bound = (0, 0, 0) 
    high_bound = (2,1,1) # bigger kappa 

    print(f"Grid shape: {grid_shape}")
    ps_grid = ps.register_volume_grid("VMF_Slice_Viewer", grid_shape, low_bound, high_bound)

    # 4. State variables for the sliders
    # Using a dictionary to allow the callback to modify these values
    state = {
        "idx_theta2": 0,
        "idx_phi2": 0,
        "needs_update": True
    }

    def callback():
        # Create sliders in the Polyscope UI
        changed_k, state["idx_theta2"] = ps.imgui.SliderInt("Theta2 Index", state["idx_theta2"], 0, node_dims[3] - 1)
        changed_p, state["idx_phi2"] = ps.imgui.SliderInt("Phi2 Index", state["idx_phi2"], 0, node_dims[4] - 1)

        if changed_k or changed_p or state["needs_update"]:
            # Slice the 5D array: [kappa_idx, :, :, :, phi2_idx]
            # This results in a 3D volume (th1, phi1, th2)
            current_slice = full_data[:, :, :, state["idx_theta2"], state["idx_phi2"]]
            
            # Update the intensity quantity on the existing grid
            ps_grid.add_scalar_quantity("Intensity", current_slice, defined_on='nodes', enabled=True)
            
            # Display current physical values for context
            ps.imgui.Text(f"Current Theta2: {coords['th2'][state['idx_theta2']]:.4f}")
            ps.imgui.Text(f"Current Phi2: {coords['phi2'][state['idx_phi2']]:.4f}")
            
            state["needs_update"] = False

    # Register the callback and run
    ps.set_user_callback(callback)

    ps.set_ground_plane_mode("shadow_only")  # Optional: disable ground plane for better visualization
    ps.show()
    
def plot_2D_slice(df, kappa_idx=0, theta2_idx=0, phi2_idx=0):
    import matplotlib.pyplot as plt
    # Extract unique values for axes
    kappa_values = sorted(df['kappa'].unique())
    theta1_values = sorted(df['th1'].unique())
    phi1_values = sorted(df['phi1'].unique())
    
    # Filter the dataframe for the specified indices
    filtered_df = df[
        (df['kappa'] == kappa_values[kappa_idx]) &
        (df['th2'] == sorted(df['th2'].unique())[theta2_idx]) &
        (df['phi2'] == sorted(df['phi2'].unique())[phi2_idx])
    ]
    
    # Pivot the data to create a 2D grid for theta1 and phi1
    pivot_table = filtered_df.pivot(index='th1', columns='phi1', values='I')
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot_table.values, extent=(min(phi1_values), max(phi1_values), min(theta1_values), max(theta1_values)), origin='lower')
    plt.colorbar(label='Intensity I')
    plt.xlabel('Phi1')
    plt.ylabel('Theta1')
    plt.title(f'Intensity Slice at Kappa={kappa_values[kappa_idx]:.2f}, Theta2={sorted(df["th2"].unique())[theta2_idx]:.4f}, Phi2={sorted(df["phi2"].unique())[phi2_idx]:.4f}')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--visualize", "-viz"]:
        # df = pd.read_csv("vmf_lut.csv")
        df = pd.read_feather("vmf_lut.feather")
        visualize_lut(df)
        exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] in ["--view_integrals", "-view"]:
        df = pd.read_feather("vmf_lut.feather")
        view_example_integrals(df)
        exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] in ["--plot_slice", "-slice"]:
        df = pd.read_feather("vmf_lut.feather")
        plot_2D_slice(df, kappa_idx=0, theta2_idx=0, phi2_idx=0)
        exit(0)

    kappa_values = -1+np.logspace(0, 2, 100) 
    theta1 = np.linspace(-np.pi/2, np.pi/2, 20)     
    phi1 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    theta2 = np.linspace(-np.pi/2, np.pi/2, 20)     
    phi2 = np.linspace(0, 2*np.pi, 10, endpoint=False)
       
    lut = compute_vmf_lut(kappa_values, phi1, theta1, phi2, theta2)
    
    # Save the LUT to a CSV file
    # df = pd.DataFrame(lut.reshape(-1), columns=["I"])
    # Nk, Nt1, Np1, Nt2, Np2 = len(kappa_values), len(theta1), len(phi1), len(theta2), len(phi2)

    # df["kappa"] = np.repeat(kappa_values, Nt1 * Np1 * Nt2 * Np2) # tiles = 1
    # df["th1"]   = np.tile(np.repeat(theta1, Np1 * Nt2 * Np2), Nk)
    # df["phi1"]  = np.tile(np.repeat(phi1, Nt2 * Np2), Nk * Nt1)
    # df["th2"]   = np.tile(np.repeat(theta2, Np2), Nk * Nt1 * Np1)
    # df["phi2"]  = np.tile(phi2, Nk * Nt1 * Np1 * Nt2)            # repeat = 1    
    # df.to_csv("vmf_lut.csv", index=False)

    # can use pandas function instead:
    # 1. Create a Cartesian product of all your axes in the correct order
    idx = pd.MultiIndex.from_product(
        [kappa_values, theta1, phi1, theta2, phi2],
        names=["kappa", "th1", "phi1", "th2", "phi2"]
    )
    df = pd.DataFrame(lut.reshape(-1), index=idx, columns=["I"]).reset_index()
    
    df.to_feather("vmf_lut.feather")

    # visualize_lut(df)
