import os
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
# Select the device index (usually 0, 1, 2...)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pandas as pd
import numpy as np
import polyscope as ps
import sys

from scipy.stats import vonmises_fisher
N_samples = 500

def draw_vmf_samples(mu, kappa, n):
    """Draws n samples from a vMF distribution."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    
    # Use scipy's vmf sampler (available in recent versions)
    # or a helper library. 
    model = vonmises_fisher(mu, kappa)
    samples = model.rvs(n).astype(np.float32)
    
    # Ensure Fortran order for Polyscope
    return np.asfortranarray(samples)


def draw_gaussian_ellipsoid(name, mean, cov, sigma=2.0, resolution=30):
    """
    Generates and registers a 3D ellipsoid mesh representing a Gaussian distribution.
    
    sigma: Controls the size. 1.0 covers ~68% of the distribution, 2.0 covers ~95%.
    resolution: How detailed the sphere mesh should be.
    """
    # 1. Generate a generic unit sphere (vertices and triangular faces)
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = np.outer(np.cos(u), np.sin(v)).flatten()
    y = np.outer(np.sin(u), np.sin(v)).flatten()
    z = np.outer(np.ones_like(u), np.cos(v)).flatten()
    
    base_verts = np.stack((x, y, z), axis=1)
    
    # Generate the faces connecting the vertices
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            p1 = i * resolution + j
            p2 = p1 + 1
            p3 = (i + 1) * resolution + j
            p4 = p3 + 1
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])
    faces = np.array(faces, dtype=np.int32)

    # 2. Decompose the covariance matrix to get axes and scales
    # eigenvalues = variance along the axes
    # eigenvectors = rotation of the axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Clip negative values just in case of slight numerical instability
    eigenvalues = np.maximum(eigenvalues, 0)
    
    # 3. Transform the sphere
    # Scale by sigma * standard_deviation (sqrt of variance), then rotate
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues)) * sigma
    
    # Apply transformation and translation
    transformed_verts = base_verts @ transform.T + mean
    
    # 4. Register in Polyscope
    verts_fortran = np.asfortranarray(transformed_verts.astype(np.float32))
    ps_mesh = ps.register_surface_mesh(name, verts_fortran, faces)
    
    # Make it look like a nice translucent wireframe
    ps_mesh.set_color((0.2, 0.6, 0.9)) # Nice blue
    ps_mesh.set_transparency(0.4)              # See-through
    ps_mesh.set_edge_width(1.0)                # Show wireframe skeleton
    
    return ps_mesh

def create_aabb_mesh(df):
    """
    Converts AABB min/max coordinates into a single mesh.
    Each AABB is represented by 8 vertices and 12 triangles (6 faces).
    """
    n_boxes = len(df)
    
    # Extract coordinates
    min_coords = df[['aabb_min_x', 'aabb_min_y', 'aabb_min_z']].values
    max_coords = df[['aabb_max_x', 'aabb_max_y', 'aabb_max_z']].values
    
    # Template for 8 vertices of a unit box [0, 1]^3
    # Order: (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)
    template_v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    
    # Scale and translate template for each box
    # vertices shape: (n_boxes, 8, 3)
    diff = max_coords - min_coords
    vertices = min_coords[:, np.newaxis, :] + template_v[np.newaxis, :, :] * diff[:, np.newaxis, :]
    
    # Flatten vertices to (n_boxes * 8, 3)
    vertices = vertices.reshape(-1, 3)
    
    # Template for 12 triangles (2 per cube face)
    template_f = np.array([
        [0, 1, 2], [0, 2, 3], # Bottom
        [4, 5, 6], [4, 6, 7], # Top
        [0, 1, 5], [0, 5, 4], # Side 1
        [1, 2, 6], [1, 6, 5], # Side 2
        [2, 3, 7], [2, 7, 6], # Side 3
        [3, 0, 4], [3, 4, 7]  # Side 4
    ])
    
    # Offset face indices for each box
    offsets = np.arange(n_boxes) * 8
    faces = template_f[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
    
    # Flatten faces to (n_boxes * 12, 3)
    faces = faces.reshape(-1, 3)
    
    return vertices, faces


def parse_rows(file_path):
    with open(file_path, 'r') as f:
        # Get the last 4 non-empty lines and extract only the data after the ':'
        vals = [line.split(':')[-1].strip() for line in f.readlines() if line.strip()][-4:]

    vmf_mean = np.fromstring(vals[0], sep=',', dtype=np.float32)
    vmf_kappa = float(vals[1])
    gaussian_mean = np.fromstring(vals[2], sep=',', dtype=np.float32)
    
    # Split the matrix by ';' into rows, and let NumPy handle the string conversion
    gaussian_cov = np.array([row.split(',') for row in vals[3].split(';')], dtype=np.float32)

    # Note the return order requested: gaussian first, then vmf
    return gaussian_mean, gaussian_cov, vmf_mean, vmf_kappa

def visualize_normals(csv_path):
    # 1. Load Data. Skip the last 6 rows:
    df = pd.read_csv(csv_path, skipfooter=6, engine='python')
    # last two rows are the gaussian positional fit:
    gaussian_mean, gaussian_cov, vmf_mean, vmf_kappa = parse_rows(csv_path)
    print("Gaussian mean:", gaussian_mean)
    print("Gaussian covariance:\n", gaussian_cov)
    print("VMF mean direction:", vmf_mean, vmf_mean.dtype)
    print("VMF kappa:", vmf_kappa)
    # drop the last two rows
    df = df.iloc[:-2]

    def callback():
        global N_samples
        # 1. Create the Slider
        changed, N_samples = ps.imgui.SliderInt("Number of Samples", N_samples, v_min=0, v_max=1000)
        
        # instead do a single point with many arrows:
        point_repeat = np.array([0.,2.,2.])
        point_repeat = np.asfortranarray(np.tile(point_repeat, (N_samples, 1)))


        # 2. If the user moves the slider, update the points
        if changed:
            new_samples = draw_vmf_samples(vmf_mean, vmf_kappa, N_samples)
            
            # Register/Update the sampled cloud
            ps_samples = ps.register_point_cloud("vMF Samples", point_repeat, radius=0.007)
            ps_samples.add_vector_quantity("Sampled Directions", new_samples, length=0.2, enabled=True)
            ps_samples.set_color((1.0, 0.5, 0.0)) # Make them orange to distinguish


    print(df)
    
    # Initialize Polyscope
    ps.init()
    points_data = np.asfortranarray(df[['px', 'py', 'pz']].values.astype(np.float32))
    normals_data = np.asfortranarray(df[['nx', 'ny', 'nz']].values.astype(np.float32))
    ps_points = ps.register_point_cloud("Sampled points", points_data, radius=0.005)
    ps_points.add_vector_quantity("Normals", normals_data, length=0.05, enabled=True)
    ps_points.add_scalar_quantity("Weight", df['weight'].values.astype(np.float32), enabled=True)
    
    # ps_vmf = ps.register_point_cloud("Fitted VMF mean direction", vmf_mean, length=0.2)

    draw_gaussian_ellipsoid("Fitted Gaussian", gaussian_mean, gaussian_cov, sigma=2.0, resolution=30)
    
    # Enable a helpful view
    ps.set_ground_plane_mode("shadow_only")
    # z up, x front:
    ps.set_up_dir("z_up")
    ps.set_front_dir("x_front")
    ps.set_user_callback(callback)
    ps.set_SSAA_factor(4)  # Enable anti-aliasing for better arrow quality
    ps.show()

if __name__ == "__main__":
    file = "bvhnormals.csv" if len(sys.argv) <= 1 else sys.argv[1]
    visualize_normals(file)

    