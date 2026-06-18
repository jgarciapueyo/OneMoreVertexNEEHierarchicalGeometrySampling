"""
Requirements:
    pip install pandas pyarrow polyscope scipy tqdm numpy

Given the integral:
    I(kappa, w1, w2) = int_{Omega} VMF(kappa) * cos(th1) * cos(th2) dOmega

For a VMF centered at +Z, the integral is invariant to joint rotation around Z.
Therefore it depends on:
    I(kappa, theta1, theta2, dphi)

Angle convention:
    theta is the polar angle from +Z
    theta = 0     -> (0, 0, 1)
    theta = pi/2  -> horizon
    theta = pi    -> (0, 0, -1)

where:
    dphi = wrapped absolute azimuth difference in [0, pi]
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


def wrap_delta_phi(phi1, phi2):
    dphi = (phi2 - phi1) % (2 * np.pi)
    return np.minimum(dphi, 2 * np.pi - dphi)


def compute_vmf_lut(kappa_values, theta1, theta2, dphi_values, n_samples=1000):
    # 4D LUT layout: (kappa, theta1, theta2, dphi)
    lut = np.zeros(
        (len(kappa_values), len(theta1), len(theta2), len(dphi_values)),
        dtype=np.float32
    )

    # By rotational symmetry around Z, fix phi1 = 0 and phi2 = dphi
    w1_dirs = np.stack([Vector(th1, 0.0).to_carthesian() for th1 in theta1], axis=0)
    w2_dirs = np.stack(
        [[Vector(th2, dphi).to_carthesian() for dphi in dphi_values] for th2 in theta2],
        axis=0
    )  # shape: (Nt2, Ndphi, 3)

    for i, kappa in enumerate(tqdm(kappa_values, desc="Kappa")):
        if kappa == 0:
            vmf_samples = scipy.stats.uniform_direction.rvs(dim=3, size=n_samples)
        else:
            vmf_samples = vonmises_fisher.rvs(mu=[0, 0, 1], kappa=kappa, size=n_samples)

        cos1 = np.clip(vmf_samples @ w1_dirs.T, 0, np.inf)  # (Ns, Nt1)

        for j in range(len(theta2)):
            cos2 = np.clip(vmf_samples @ w2_dirs[j].T, 0, np.inf)  # (Ns, Ndphi)
            lut[i, :, j, :] = (cos1[:, :, None] * cos2[:, None, :]).mean(axis=0)

    return lut


def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([[x, y, z]])


def lut_to_packed_dataframe(lut, kappa_values, theta_values, dphi_values, packed=True):
    """
    Convert a dense LUT numpy array to a pandas DataFrame.

    If `packed=True` (default) the function only stores the upper-triangular
    (`th2` index >= `th1` index) entries to exploit symmetry, producing
    Nkappa * (Nt*(Nt+1)/2) * Ndphi rows. If `packed=False` the function
    stores all (th1, th2) pairs producing Nkappa * Nt * Nt * Ndphi rows.
    """
    rows = []
    for ik, kappa in enumerate(kappa_values):
        for i, th1 in enumerate(theta_values):
            if packed:
                j_range = range(i, len(theta_values))
            else:
                j_range = range(0, len(theta_values))

            for j in j_range:
                th2 = theta_values[j]
                for idp, dphi in enumerate(dphi_values):
                    rows.append((kappa, th1, th2, dphi, float(lut[ik, i, j, idp])))

    return pd.DataFrame(rows, columns=["kappa", "th1", "th2", "dphi", "I"])


def packed_df_to_dense_lut(df):
    kappa_values = np.array(sorted(df["kappa"].unique()))
    theta_values = np.array(sorted(set(df["th1"].unique()) | set(df["th2"].unique())))
    dphi_values = np.array(sorted(df["dphi"].unique()))

    kappa_index = {v: i for i, v in enumerate(kappa_values)}
    theta_index = {v: i for i, v in enumerate(theta_values)}
    dphi_index = {v: i for i, v in enumerate(dphi_values)}

    full_data = np.zeros(
        (len(kappa_values), len(theta_values), len(theta_values), len(dphi_values)),
        dtype=np.float32
    )

    for row in df.itertuples(index=False):
        ik = kappa_index[row.kappa]
        i = theta_index[row.th1]
        j = theta_index[row.th2]
        idp = dphi_index[row.dphi]

        full_data[ik, i, j, idp] = row.I
        full_data[ik, j, i, idp] = row.I

    coords = {
        "kappa": kappa_values,
        "th1": theta_values,
        "th2": theta_values,
        "dphi": dphi_values,
    }
    return coords, full_data


def view_example_integrals(df):
    import polyscope as ps

    coords, full_data = packed_df_to_dense_lut(df)
    node_dims = full_data.shape

    ps.init()
    pc = ps.register_point_cloud("Origin", np.array([[0, 0, 0]]), color=(0, 0, 0))

    state = {
        "idx_kappa": 0,
        "idx_theta1": 0,
        "idx_theta2": 0,
        "idx_dphi": 0,
        "needs_update": True,
        "current_val": 0.0,
        "current_dphi": 0.0,
    }

    def callback():
        changed_k, state["idx_kappa"] = ps.imgui.SliderInt("Kappa Index", state["idx_kappa"], 0, node_dims[0] - 1)
        changed_t1, state["idx_theta1"] = ps.imgui.SliderInt("Theta1 Index", state["idx_theta1"], 0, node_dims[1] - 1)
        changed_t2, state["idx_theta2"] = ps.imgui.SliderInt("Theta2 Index", state["idx_theta2"], 0, node_dims[2] - 1)
        changed_d, state["idx_dphi"] = ps.imgui.SliderInt("Delta Phi Index", state["idx_dphi"], 0, node_dims[3] - 1)

        if any([changed_k, changed_t1, changed_t2, changed_d, state["needs_update"]]):
            th1 = coords["th1"][state["idx_theta1"]]
            th2 = coords["th2"][state["idx_theta2"]]
            dphi = coords["dphi"][state["idx_dphi"]]

            v1_cart = spherical_to_cartesian(th1, 0.0)
            v2_cart = spherical_to_cartesian(th2, dphi)

            pc.add_vector_quantity("w1", v1_cart, color=(1, 0, 0), enabled=True)
            pc.add_vector_quantity("w2", v2_cart, color=(0, 1, 0), enabled=True)

            state["current_val"] = float(full_data[
                state["idx_kappa"],
                state["idx_theta1"],
                state["idx_theta2"],
                state["idx_dphi"]
            ])
            state["current_dphi"] = float(dphi)
            state["needs_update"] = False

        ps.imgui.Text(f"Current Intensity: {state['current_val']:.6f}")
        ps.imgui.Text(f"dphi: {state['current_dphi']:.6f}")

    ps.set_user_callback(callback)
    ps.set_ground_plane_mode("shadow_only")
    ps.show()


def visualize_lut(df):
    import polyscope as ps

    coords, full_data = packed_df_to_dense_lut(df)
    node_dims = full_data.shape

    ps.init()

    grid_shape = (node_dims[0], node_dims[1], node_dims[2])
    low_bound = (coords["kappa"][0], coords["th1"][0], coords["th2"][0])
    high_bound = (coords["kappa"][-1] * 0.1, coords["th1"][-1], coords["th2"][-1])

    print(f"Grid shape: {grid_shape}")
    ps_grid = ps.register_volume_grid("VMF_Slice_Viewer", grid_shape, low_bound, high_bound)

    state = {
        "idx_dphi": 0,
        "needs_update": True,
        "current_dphi": float(coords["dphi"][0]),
    }

    def callback():
        changed_d, state["idx_dphi"] = ps.imgui.SliderInt("Delta Phi Index", state["idx_dphi"], 0, node_dims[3] - 1)

        if changed_d or state["needs_update"]:
            current_slice = full_data[:, :, :, state["idx_dphi"]]
            ps_grid.add_scalar_quantity("Intensity", current_slice, defined_on="nodes", enabled=True)
            state["current_dphi"] = float(coords["dphi"][state["idx_dphi"]])
            state["needs_update"] = False

        ps.imgui.Text(f"Current dphi: {state['current_dphi']:.6f}")

    ps.set_user_callback(callback)
    ps.set_ground_plane_mode("shadow_only")
    ps.show()


def plot_2D_slice(df, kappa_idx=0, dphi_idx=0):
    import matplotlib.pyplot as plt

    coords, full_data = packed_df_to_dense_lut(df)

    slice_2d = full_data[kappa_idx, :, :, dphi_idx]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        slice_2d,
        extent=(
            coords["th2"].min(), coords["th2"].max(),
            coords["th1"].min(), coords["th1"].max()
        ),
        origin="lower",
        aspect="auto"
    )
    plt.colorbar(label="Intensity I")
    plt.xlabel("Theta2")
    plt.ylabel("Theta1")
    plt.title(f"Intensity Slice at Kappa={coords['kappa'][kappa_idx]:.2f}, dphi={coords['dphi'][dphi_idx]:.4f}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--visualize", "-viz"]:
        df = pd.read_feather("vmf_lut.feather")
        visualize_lut(df)
        exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] in ["--view_integrals", "-view"]:
        df = pd.read_feather("vmf_lut.feather")
        view_example_integrals(df)
        exit(0)

    elif len(sys.argv) > 1 and sys.argv[1] in ["--plot_slice", "-slice"]:
        df = pd.read_feather("vmf_lut.feather")
        plot_2D_slice(df, kappa_idx=0, dphi_idx=0)
        exit(0)

    kappa_values = -1 + np.logspace(0, 2, 100)
    print(f"Using kappa values: {kappa_values}")

    theta1 = np.linspace(0.0, np.pi, 40)
    theta2 = np.linspace(0.0, np.pi, 40)

    if len(theta1) != len(theta2) or not np.allclose(theta1, theta2):
        raise ValueError("Packed symmetry requires theta1 and theta2 to use the same grid.")

    dphi_values = np.linspace(0, np.pi, 10)

    lut = compute_vmf_lut(kappa_values, theta1, theta2, dphi_values, n_samples=100000)
    print("lut size: ", lut.shape)
    df = lut_to_packed_dataframe(lut, kappa_values, theta1, dphi_values, packed=False)
    print(df.describe())
    # df = lut_to_packed_dataframe(lut, kappa_values, theta1, dphi_values, packed=True)
    # print(df.describe())
    df.to_feather("vmf_lut.feather")
