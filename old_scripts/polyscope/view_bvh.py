import pandas as pd
import numpy as np

import os
from pathlib import Path

import matplotlib.pyplot as plt

import polyscope as ps
import polyscope.imgui as psim
import sys


VMF_ARROW_LENGTH = 0.03
VMF_DELTA_KAPPA = 80.0
VMF_LOBE_MESH_RESOLUTION = 10
VMF_LOBE_MIN_RADIUS_FRACTION = 0.08
VMF_LOBE_SIZE_DEFAULT = 0.6
VMF_LOBE_SIZE_MIN = 0.3
VMF_LOBE_SIZE_MAX = 1.0

IMPORTANCE_COLORMAP = "viridis"

AABB_IMPORTANCE_OPTIONS = [
    ("I Ours", "imp_ours"),
    ("GT (Vis)", "imp_gt_vis"),
    ("GT (NoVis)", "imp_gt_novis"),
    ("LUT (NoVis)", "imp_lut_novis"),
    ("LUT (Vis)", "imp_lut_vis"),
    ("Unscented MC (NoVis)", "imp_unscented_mc_novis"),
    ("Unscented MC (Vis)", "imp_unscented_mc_vis"),
    ("Unscented LUT (NoVis)", "imp_unscented_lut_novis"),
    ("Unscented LUT (Vis)", "imp_unscented_lut_vis"),
]

AABB_DIFF_TONEMAP_OPTIONS = [
    ("Squared (diff^2)", "square"),
    ("Absolute (abs(diff))", "abs"),
    ("Signed diverging", "diverging"),
]

def load_bvh_dataframe(csv_path):
    df = pd.read_csv(csv_path)

    required_columns = [
        'node_index','level',
        'aabb_min_x', 'aabb_min_y', 'aabb_min_z',
        'aabb_max_x', 'aabb_max_y', 'aabb_max_z',
        'n_subtree_primitives', 'n_leaf_primitives',
        'node_type', 'valid', 'surface_area',
        'center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z',
        'gaussian_mu_x', 'gaussian_mu_y', 'gaussian_mu_z',
        'gaussian_cov_xx', 'gaussian_cov_xy', 'gaussian_cov_xz',
        'gaussian_cov_yy', 'gaussian_cov_yz', 'gaussian_cov_zz',
        'vmf_mean_x', 'vmf_mean_y', 'vmf_mean_z', 'vmf_kappa',
        'imp_gt_vis', 'imp_gt_novis', 'imp_ours',
        'imp_lut_vis', 'imp_lut_novis',
        'imp_unscented_mc_vis', 'imp_unscented_mc_novis',
        'imp_unscented_lut_vis', 'imp_unscented_lut_novis'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required BVH columns: {missing}")

    numeric_check_columns = [
        'gaussian_mu_x', 'gaussian_mu_y', 'gaussian_mu_z',
        'gaussian_cov_xx', 'gaussian_cov_xy', 'gaussian_cov_xz',
        'gaussian_cov_yy', 'gaussian_cov_yz', 'gaussian_cov_zz',
        'vmf_mean_x', 'vmf_mean_y', 'vmf_mean_z', 'vmf_kappa'
    ]
    for col in numeric_check_columns:
        coerced = pd.to_numeric(df[col], errors='coerce')
        invalid_count = int((coerced.isna() & ~df[col].isna()).sum())
        if invalid_count > 0:
            print(f"Warning: {invalid_count} malformed values found in column '{col}'")
        df[col] = coerced

    return df


def _create_unit_sphere_mesh(resolution=18):
    u = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    v = np.linspace(0.0, np.pi, resolution)
    uu, vv = np.meshgrid(u, v, indexing='ij')

    x = np.cos(uu) * np.sin(vv)
    y = np.sin(uu) * np.sin(vv)
    z = np.cos(vv)

    verts = np.stack((x, y, z), axis=-1).reshape(-1, 3).astype(np.float32)

    faces = []
    for i in range(resolution):
        i_next = (i + 1) % resolution
        for j in range(resolution - 1):
            p1 = i * resolution + j
            p2 = i * resolution + (j + 1)
            p3 = i_next * resolution + j
            p4 = i_next * resolution + (j + 1)
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    return verts, np.array(faces, dtype=np.int32)


def create_gaussian_mesh(df_leaves, sigma=2.0, resolution=16):
    if df_leaves.empty:
        print("No leaf nodes found for Gaussian mesh creation.")
        return None, None

    base_verts, base_faces = _create_unit_sphere_mesh(resolution=resolution)
    n_base_verts = base_verts.shape[0]

    all_verts = []
    all_faces = []
    valid_count = 0

    for _, row in df_leaves.iterrows():
        mean = np.array([
            row['gaussian_mu_x'], row['gaussian_mu_y'], row['gaussian_mu_z']
        ], dtype=np.float64)

        cov = np.array([
            [row['gaussian_cov_xx'], row['gaussian_cov_xy'], row['gaussian_cov_xz']],
            [row['gaussian_cov_xy'], row['gaussian_cov_yy'], row['gaussian_cov_yz']],
            [row['gaussian_cov_xz'], row['gaussian_cov_yz'], row['gaussian_cov_zz']]
        ], dtype=np.float64)

        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
            continue

        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)

        transform = (eigvecs @ np.diag(np.sqrt(eigvals))) * sigma
        transformed = base_verts @ transform.T + mean

        offset = valid_count * n_base_verts
        all_verts.append(transformed.astype(np.float32))
        all_faces.append(base_faces + offset)
        valid_count += 1

    print(f"Created Gaussian meshes for {valid_count} nodes")

    if valid_count == 0:
        return None, None

    verts = np.asfortranarray(np.vstack(all_verts))
    faces = np.vstack(all_faces).astype(np.int32)
    return verts, faces


def _normalize_vectors(vectors, fallback=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return vectors

    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        fallback = np.asarray(fallback, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors.copy()
    valid = norms[:, 0] > 1e-8

    if np.any(valid):
        normalized[valid] = normalized[valid] / norms[valid]
    if np.any(~valid):
        normalized[~valid] = fallback

    return normalized.astype(np.float32)


def _compute_area_scales(surface_areas):
    areas = np.asarray(surface_areas, dtype=np.float32)
    if areas.size == 0:
        return areas

    areas = np.maximum(areas, 0.0)
    positive = areas[areas > 0.0]
    if positive.size == 0:
        return np.ones_like(areas, dtype=np.float32)

    reference_area = float(np.median(positive))
    reference_area = max(reference_area, 1e-8)

    scales = np.sqrt(areas / reference_area)
    scales = np.clip(scales, 0.0, 100.0)
    return scales.astype(np.float32)


def _compute_diff_scalar_values(i_values, r_values, tonemap_key):
    i_arr = np.asarray(i_values, dtype=np.float64)
    r_arr = np.asarray(r_values, dtype=np.float64)
    if i_arr.size == 0 or r_arr.size == 0 or i_arr.shape != r_arr.shape:
        return np.empty((0,), dtype=np.float32), 0.0

    diff = i_arr - r_arr
    safe = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    if tonemap_key == "square":
        mapped = np.nan_to_num(safe * safe, nan=0.0, posinf=0.0, neginf=0.0)
    elif tonemap_key == "abs":
        mapped = np.nan_to_num(np.abs(safe), nan=0.0, posinf=0.0, neginf=0.0)
    else:
        mapped = np.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0)

    f32_max = np.finfo(np.float32).max
    mapped = np.clip(mapped, -f32_max, f32_max).astype(np.float32)

    abs_mapped = np.abs(mapped)
    max_abs = float(np.max(abs_mapped)) if abs_mapped.size > 0 else 0.0
    return mapped, max_abs


def create_vmf_level_geometry(df_level, delta_kappa=VMF_DELTA_KAPPA, lobe_resolution=VMF_LOBE_MESH_RESOLUTION):
    if df_level.empty:
        return None

    centers = df_level[['center_of_mass_x', 'center_of_mass_y', 'center_of_mass_z']].to_numpy(dtype=np.float32)
    means = df_level[['vmf_mean_x', 'vmf_mean_y', 'vmf_mean_z']].to_numpy(dtype=np.float32)
    kappas = pd.to_numeric(df_level['vmf_kappa'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    surface_areas = pd.to_numeric(df_level['surface_area'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)

    finite_mask = (
        np.all(np.isfinite(centers), axis=1)
        & np.all(np.isfinite(means), axis=1)
        & np.isfinite(kappas)
        & np.isfinite(surface_areas)
    )
    if not np.any(finite_mask):
        return None

    centers = centers[finite_mask]
    means = _normalize_vectors(means[finite_mask])
    kappas = np.maximum(kappas[finite_mask], 0.0)
    area_scales = _compute_area_scales(surface_areas[finite_mask])

    delta_mask = kappas > float(delta_kappa)
    lobe_mask = ~delta_mask

    result = {
        'lobe': None,
        'delta_points': None,
        'delta_dirs': None,
        'delta_scales': None,
    }

    if np.any(delta_mask):
        result['delta_points'] = np.asfortranarray(centers[delta_mask].astype(np.float32))
        result['delta_dirs'] = np.asfortranarray(means[delta_mask].astype(np.float32))
        result['delta_scales'] = np.asfortranarray(area_scales[delta_mask].astype(np.float32))

    if np.any(lobe_mask):
        lobe_centers = centers[lobe_mask].astype(np.float32)
        lobe_means = means[lobe_mask].astype(np.float32)
        lobe_kappas = kappas[lobe_mask].astype(np.float32)
        lobe_scales = area_scales[lobe_mask].astype(np.float32)

        base_dirs, base_faces = _create_unit_sphere_mesh(resolution=int(max(4, lobe_resolution)))
        n_lobes = lobe_centers.shape[0]
        n_base_verts = base_dirs.shape[0]

        dots = np.clip(lobe_means @ base_dirs.T, -1.0, 1.0)
        exponents = np.clip(lobe_kappas[:, np.newaxis] * (dots - 1.0), -60.0, 0.0)
        weights = np.exp(exponents).astype(np.float32)
        radial = VMF_LOBE_MIN_RADIUS_FRACTION + (1.0 - VMF_LOBE_MIN_RADIUS_FRACTION) * weights

        offsets = (
            radial[:, :, np.newaxis]
            * lobe_scales[:, np.newaxis, np.newaxis]
            * base_dirs[np.newaxis, :, :]
        ).astype(np.float32)
        offsets_vertices = offsets.reshape(-1, 3)
        centers_vertices = np.repeat(lobe_centers, n_base_verts, axis=0).astype(np.float32)

        faces = np.vstack([
            base_faces + lobe_idx * n_base_verts
            for lobe_idx in range(n_lobes)
        ]).astype(np.int32)

        initial_verts = centers_vertices + offsets_vertices * float(VMF_LOBE_SIZE_DEFAULT)

        result['lobe'] = {
            'vertices': np.asfortranarray(initial_verts.astype(np.float32)),
            'faces': faces,
            'centers_vertices': centers_vertices,
            'offsets_vertices': offsets_vertices,
        }

    return result


def update_vmf_lobe_scale(vmf_renderers_by_level, lobe_size):
    size = float(max(VMF_LOBE_SIZE_MIN, min(VMF_LOBE_SIZE_MAX, lobe_size)))
    for level_entry in vmf_renderers_by_level.values():
        lobe_data = level_entry.get('lobe')
        if lobe_data is None: continue

        mesh = lobe_data.get('mesh')
        if mesh is None: continue

        new_verts = lobe_data['centers_vertices'] + lobe_data['offsets_vertices'] * size
        mesh.update_vertex_positions(np.asfortranarray(new_verts.astype(np.float32)))


def create_aabb_mesh(df):
    n_boxes = len(df)
    min_coords = df[['aabb_min_x', 'aabb_min_y', 'aabb_min_z']].values
    max_coords = df[['aabb_max_x', 'aabb_max_y', 'aabb_max_z']].values
    
    template_v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    
    diff = max_coords - min_coords
    vertices = min_coords[:, np.newaxis, :] + template_v[np.newaxis, :, :] * diff[:, np.newaxis, :]
    vertices = vertices.reshape(-1, 3)
    
    template_f = np.array([
        [0, 1, 2], [0, 2, 3], # Bottom
        [4, 5, 6], [4, 6, 7], # Top
        [0, 1, 5], [0, 5, 4], # Side 1
        [1, 2, 6], [1, 6, 5], # Side 2
        [2, 3, 7], [2, 7, 6], # Side 3
        [3, 0, 4], [3, 4, 7]  # Side 4
    ])
    
    offsets = np.arange(n_boxes) * 8
    faces = template_f[np.newaxis, :, :] + offsets[:, np.newaxis, np.newaxis]
    faces = faces.reshape(-1, 3)
    
    return vertices, faces


def _crop_empty_background(image, tolerance=0.02, padding=2):
    arr = np.asarray(image)
    if arr.ndim < 2: return arr

    working = arr[:, :, np.newaxis] if arr.ndim == 2 else arr
    rgb_like = working[:, :, :min(3, working.shape[2])]

    h, w = rgb_like.shape[:2]
    corners = np.stack([rgb_like[0, 0], rgb_like[0, w - 1], rgb_like[h - 1, 0], rgb_like[h - 1, w - 1]], axis=0)
    bg_color = np.median(corners, axis=0)

    diff = np.max(np.abs(rgb_like - bg_color.reshape(1, 1, -1)), axis=2)
    mask = diff > float(tolerance)

    if working.shape[2] >= 4:
        alpha = working[:, :, 3]
        alpha_threshold = 0.02 if np.issubdtype(working.dtype, np.floating) else 5
        mask = np.logical_or(mask, alpha > alpha_threshold)

    coords = np.argwhere(mask)
    if coords.size == 0: return arr

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1

    pad = max(0, int(padding))
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(arr.shape[0], r1 + pad)
    c1 = min(arr.shape[1], c1 + pad)

    return arr[r0:r1, c0:c1] if arr.ndim == 2 else arr[r0:r1, c0:c1, :]


def _to_rgb(image):
    arr = np.asarray(image)
    if arr.ndim == 2: return np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 1: return np.repeat(arr, 3, axis=2)
    if arr.shape[2] >= 3: return arr[:, :, :3]
    return np.stack([arr[:, :, 0], arr[:, :, 0], arr[:, :, 0]], axis=2)


def _stitch_images_horizontally(images, gap_px=2):
    if not images: return None

    rgb_images = [_to_rgb(img) for img in images]
    heights = [img.shape[0] for img in rgb_images]
    widths = [img.shape[1] for img in rgb_images]

    out_h = int(max(heights))
    out_w = int(sum(widths) + max(0, len(rgb_images) - 1) * max(0, int(gap_px)))

    first = rgb_images[0]
    bg = np.median(np.stack([first[0, 0], first[0, -1], first[-1, 0], first[-1, -1]], axis=0), axis=0)

    canvas = np.empty((out_h, out_w, 3), dtype=first.dtype)
    canvas[:] = bg

    cursor_x = 0
    gap = max(0, int(gap_px))
    for img in rgb_images:
        h, w = img.shape[:2]
        y0 = (out_h - h) // 2
        canvas[y0:y0 + h, cursor_x:cursor_x + w, :] = img
        cursor_x += w + gap

    return canvas

def visualize_bvh(csv_path):
    df = load_bvh_dataframe(csv_path)
    df['level'] = pd.to_numeric(df['level'], errors='coerce').fillna(0).astype(int)

    min_level, max_level = int(df['level'].min()), int(df['level'].max())
    available_levels = sorted(int(level) for level in df['level'].dropna().unique())

    screenshots_dir = Path(__file__).resolve().parent / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    importance_screenshots_dir = Path(__file__).resolve().parent / "importance_screenshots"
    importance_screenshots_dir.mkdir(parents=True, exist_ok=True)
    export_prefix = Path(csv_path).stem

    ps.init()

    aabb_meshes_by_level = {}
    
    for (level, node_type), group_df in df.groupby(['level', 'node_type']):
        if group_df.empty: continue
        
        verts, faces = create_aabb_mesh(group_df)
        mesh_name = f"BVH_L{level}_{node_type}"
        ps_mesh = ps.register_surface_mesh(mesh_name, verts, faces)

        mesh_entry = {
            "mesh": ps_mesh,
            "name": mesh_name,
            "importance_arrays": {},       # column_name -> (display_name, np.ndarray)
            "diff_color_quantities": {}    # (i_col, r_col, tonemap_key) -> (diff_label, diff_scalar, cmap, datatype, vminmax)
        }
        aabb_meshes_by_level.setdefault(int(level), []).append(mesh_entry)
    
        # Directly register all base scalar quantities using native API
        for display_name, column_name in AABB_IMPORTANCE_OPTIONS:
            if column_name in group_df.columns:
                values = pd.to_numeric(group_df[column_name], errors='coerce').fillna(0.0).values
                arr_clean = np.ascontiguousarray(np.repeat(values, 12).astype(np.float32))
                
                ps_mesh.add_scalar_quantity(
                    display_name,
                    arr_clean,
                    defined_on='faces',
                    enabled=False,
                    cmap=IMPORTANCE_COLORMAP,
                    onscreen_colorbar_enabled=True
                )
                mesh_entry["importance_arrays"][column_name] = (display_name, arr_clean)

    available_aabb_levels = sorted(aabb_meshes_by_level.keys())

    gaussian_meshes_by_level = {}
    for level, level_df in df.groupby('level'):
        gaussian_verts, gaussian_faces = create_gaussian_mesh(level_df, sigma=2.0, resolution=16)
        if gaussian_verts is not None and gaussian_faces is not None:
            mesh = ps.register_surface_mesh(f"Gaussians L{level}", gaussian_verts, gaussian_faces)
            mesh.set_color((0.2, 0.6, 0.9))
            mesh.set_transparency(0.4)
            mesh.set_edge_width(1.0)
            mesh.set_enabled(False)
            gaussian_meshes_by_level[int(level)] = mesh

    vmf_renderers_by_level = {}
    for level, level_df in df.groupby('level'):
        level_geometry = create_vmf_level_geometry(level_df)
        if level_geometry is None: continue

        level_entry = {}
        if level_geometry.get('lobe'):
            lobe_mesh = ps.register_surface_mesh(f"Node vMF Lobes L{level}", level_geometry['lobe']['vertices'], level_geometry['lobe']['faces'])
            lobe_mesh.set_color((0.95, 0.55, 0.2))
            lobe_mesh.set_transparency(0.45)
            lobe_mesh.set_enabled(False)
            level_geometry['lobe']['mesh'] = lobe_mesh
            level_entry['lobe'] = level_geometry['lobe']

        delta_points = level_geometry.get('delta_points')
        delta_dirs = level_geometry.get('delta_dirs')
        if delta_points is not None and len(delta_points) > 0:
            scales = level_geometry.get('delta_scales')
            scaled_dirs = np.asfortranarray((delta_dirs * scales[:, np.newaxis]).astype(np.float32)) if scales is not None else delta_dirs
            cloud = ps.register_point_cloud(f"Node vMF Delta L{level}", delta_points, radius=0.0025)
            cloud.add_vector_quantity("Mean Direction", scaled_dirs, length=VMF_ARROW_LENGTH, enabled=True)
            cloud.set_enabled(False)
            level_entry['delta_cloud'] = cloud

        if level_entry:
            vmf_renderers_by_level[int(level)] = level_entry

    update_vmf_lobe_scale(vmf_renderers_by_level, VMF_LOBE_SIZE_DEFAULT)

    view_state = {
        'mode': 0, 'level': min_level, 'aabb_color_idx': 0, 'aabb_compare_mode': 0,
        'aabb_diff_i_idx': 0, 'aabb_diff_r_idx': 1 if len(AABB_IMPORTANCE_OPTIONS) > 1 else 0,
        'aabb_diff_tonemap_idx': 0, 'vmf_lobe_size': VMF_LOBE_SIZE_DEFAULT,
    }
    ui_state = {'status_message': ""}

    def _disable_all_aabb_quantities(mesh_entry):
        mesh = mesh_entry["mesh"]
        for display_name, arr in mesh_entry["importance_arrays"].values():
            mesh.add_scalar_quantity(display_name, arr, defined_on='faces', enabled=False, cmap=IMPORTANCE_COLORMAP, onscreen_colorbar_enabled=True)
        for diff_label, diff_scalar, cmap, datatype, vminmax in mesh_entry.get("diff_color_quantities", {}).values():
            mesh.add_scalar_quantity(diff_label, diff_scalar, defined_on='faces', enabled=False, cmap=cmap, datatype=datatype, vminmax=vminmax, onscreen_colorbar_enabled=True)

    def _ensure_diff_color_quantity(mesh_entry, i_column, r_column, tonemap_key):
        cache = mesh_entry.setdefault("diff_color_quantities", {})
        cache_key = (i_column, r_column, tonemap_key)
        if cache_key in cache: return cache[cache_key]

        if i_column not in mesh_entry["importance_arrays"] or r_column not in mesh_entry["importance_arrays"]:
            return None

        _, i_arr = mesh_entry["importance_arrays"][i_column]
        _, r_arr = mesh_entry["importance_arrays"][r_column]

        diff_scalar, diff_max_abs = _compute_diff_scalar_values(i_arr, r_arr, tonemap_key)
        diff_label = f"{i_column}-{r_column} ({tonemap_key})"

        if tonemap_key == "diverging":
            cmap = "coolwarm"
            datatype = "symmetric"
            v_max = max(1e-8, diff_max_abs)
            vminmax = (-v_max, v_max)
        else:
            cmap = "viridis"
            datatype = "standard"
            v_max = max(1e-8, float(np.max(diff_scalar)))
            vminmax = (0.0, v_max)

        mesh_entry["mesh"].add_scalar_quantity(
            diff_label,
            diff_scalar,
            defined_on='faces',
            enabled=False,
            cmap=cmap,
            datatype=datatype,
            vminmax=vminmax,
            onscreen_colorbar_enabled=True
        )

        result = (diff_label, diff_scalar, cmap, datatype, vminmax)
        cache[cache_key] = result
        return result
    
    def _precompute_all_diff_quantities():
        imp_cols = [col for _, col in AABB_IMPORTANCE_OPTIONS]
        tones = [tone for _, tone in AABB_DIFF_TONEMAP_OPTIONS]
        for mesh_entries in aabb_meshes_by_level.values():
            for entry in mesh_entries:
                for i_col in imp_cols:
                    for r_col in imp_cols:
                        for tone in tones:
                            _ensure_diff_color_quantity(entry, i_col, r_col, tone)

    def update_visibility(*args, **kwargs):
        mode = view_state['mode']
        level = view_state['level']
        if mode == 0 and available_aabb_levels and level not in aabb_meshes_by_level:
            level = min(available_aabb_levels, key=lambda x: abs(x - level))

        show_aabb = (mode == 0)
        show_gaussian = (mode == 1)
        show_vmf = (mode == 2)

        sel_col = AABB_IMPORTANCE_OPTIONS[view_state['aabb_color_idx']][1]
        sel_i_col = AABB_IMPORTANCE_OPTIONS[view_state['aabb_diff_i_idx']][1]
        sel_r_col = AABB_IMPORTANCE_OPTIONS[view_state['aabb_diff_r_idx']][1]
        sel_tone = AABB_DIFF_TONEMAP_OPTIONS[view_state['aabb_diff_tonemap_idx']][1]

        for mesh_level, mesh_entries in aabb_meshes_by_level.items():
            is_enabled = show_aabb and (mesh_level == level)
            for entry in mesh_entries:
                entry["mesh"].set_enabled(is_enabled)
                _disable_all_aabb_quantities(entry)

                if is_enabled:
                    if view_state['aabb_compare_mode'] == 0:
                        if sel_col in entry["importance_arrays"]:
                            disp_name, arr = entry["importance_arrays"][sel_col]
                            entry["mesh"].add_scalar_quantity(disp_name, arr, defined_on='faces', enabled=True, cmap=IMPORTANCE_COLORMAP, onscreen_colorbar_enabled=True)
                    else:
                        diff_res = _ensure_diff_color_quantity(entry, sel_i_col, sel_r_col, sel_tone)
                        if diff_res:
                            diff_label, diff_scalar, cmap, datatype, vminmax = diff_res
                            entry["mesh"].add_scalar_quantity(diff_label, diff_scalar, defined_on='faces', enabled=True, cmap=cmap, datatype=datatype, vminmax=vminmax, onscreen_colorbar_enabled=True)

        for mesh_level, gaussian_mesh in gaussian_meshes_by_level.items():
            gaussian_mesh.set_enabled(show_gaussian and mesh_level == level)

        for cloud_level, vmf_entry in vmf_renderers_by_level.items():
            is_enabled = show_vmf and cloud_level == level
            if vmf_entry.get('lobe') and vmf_entry['lobe'].get('mesh'):
                vmf_entry['lobe']['mesh'].set_enabled(is_enabled)
            if vmf_entry.get('delta_cloud'):
                vmf_entry['delta_cloud'].set_enabled(is_enabled)

    _precompute_all_diff_quantities()
    update_visibility()

    def export_levels_figure():
        original_state = view_state.copy()
        try:
            saved_paths = []
            for level in available_levels:
                row_paths = []
                for mode, tag, title in [(0, "aabbs", "AABBs"), (1, "gaussians", "Gaussians"), (2, "vmfs", "vMFs")]:
                    view_state.update({'mode': mode, 'level': level})
                    update_visibility()
                    out_path = screenshots_dir / f"{export_prefix}_level_{level:03d}_{tag}.png"
                    ps.screenshot(str(out_path))
                    row_paths.append((title, out_path))
                saved_paths.append((level, row_paths))

            fig, axes = plt.subplots(len(saved_paths), 3, figsize=(12, 4 * max(1, len(saved_paths))), squeeze=False)
            for r_idx, (level, row_paths) in enumerate(saved_paths):
                for c_idx, (title, img_path) in enumerate(row_paths):
                    axes[r_idx, c_idx].imshow(plt.imread(img_path))
                    axes[r_idx, c_idx].set_title(f"Level {level} - {title}")
                    axes[r_idx, c_idx].axis("off")

            fig.tight_layout()
            out_fig = screenshots_dir / f"{export_prefix}_levels_x_3.png"
            fig.savefig(str(out_fig), dpi=200, bbox_inches="tight")
            plt.close(fig)
            ui_state['status_message'] = f"Saved levels x 3 figure to: {screenshots_dir.resolve()}"
        finally:
            view_state.update(original_state)
            update_visibility()

    def export_importance_rows():
        original_state = view_state.copy()
        export_levels = available_aabb_levels if available_aabb_levels else available_levels
        try:
            saved, saved_img = 0, 0
            for idx, (_, col_name) in enumerate(AABB_IMPORTANCE_OPTIONS):
                if col_name not in df.columns: continue

                row_stem = f"importance_{col_name[4:]}" if col_name.startswith("imp_") else col_name
                level_paths = []

                for level in export_levels:
                    view_state.update({'mode': 0, 'level': level, 'aabb_color_idx': idx, 'aabb_compare_mode': 0})
                    update_visibility()
                    path = importance_screenshots_dir / f"{row_stem}_level_{level:03d}.png"
                    ps.screenshot(str(path))
                    level_paths.append(path)
                    saved_img += 1

                if not level_paths: continue

                cropped = [_crop_empty_background(plt.imread(p)) for p in level_paths]
                row_image = _stitch_images_horizontally(cropped)
                if row_image is not None:
                    plt.imsave(str(importance_screenshots_dir / f"{row_stem}.png"), row_image)
                    saved += 1

            ui_state['status_message'] = f"Saved {saved} rows, {saved_img} PNGs to: {importance_screenshots_dir.resolve()}"
        finally:
            view_state.update(original_state)
            update_visibility()

    def callback():
        changed = False
        
        # Base UI Controls natively via Polyscope ImGui wrapper
        ch_mode, view_state['mode'] = psim.SliderInt("AABB/Gaussian/vMF", view_state['mode'], v_min=0, v_max=2)
        ch_lvl, view_state['level'] = psim.SliderInt("BVH Level", view_state['level'], v_min=min_level, v_max=max_level)
        changed |= ch_mode or ch_lvl

        if view_state['mode'] == 0:
            ch_comp, view_state['aabb_compare_mode'] = psim.Combo("AABB Color Mode", view_state['aabb_compare_mode'], ["Single", "Diff (i-r)"])
            changed |= ch_comp

            if view_state['aabb_compare_mode'] == 0:
                labels = [disp for disp, _ in AABB_IMPORTANCE_OPTIONS]
                ch_col, view_state['aabb_color_idx'] = psim.Combo("Importance", view_state['aabb_color_idx'], labels)
                changed |= ch_col
            else:
                labels = [disp for disp, _ in AABB_IMPORTANCE_OPTIONS]
                tones = [disp for disp, _ in AABB_DIFF_TONEMAP_OPTIONS]
                ch_i, view_state['aabb_diff_i_idx'] = psim.Combo("Diff i", view_state['aabb_diff_i_idx'], labels)
                ch_r, view_state['aabb_diff_r_idx'] = psim.Combo("Diff r", view_state['aabb_diff_r_idx'], labels)
                ch_tone, view_state['aabb_diff_tonemap_idx'] = psim.Combo("Diff Tonemap", view_state['aabb_diff_tonemap_idx'], tones)
                changed |= ch_i or ch_r or ch_tone

        if view_state['mode'] == 2:
            ch_vmf, vmf_lobe = psim.SliderFloat("vMF Lobe Size", view_state['vmf_lobe_size'], v_min=VMF_LOBE_SIZE_MIN, v_max=VMF_LOBE_SIZE_MAX)
            if ch_vmf:
                view_state['vmf_lobe_size'] = float(vmf_lobe)
                update_vmf_lobe_scale(vmf_renderers_by_level, view_state['vmf_lobe_size'])

        if changed:
            update_visibility()

        if psim.Button("Export levels x 3 figure"): export_levels_figure()
        if psim.Button("Export importance rows"): export_importance_rows()

        if ui_state.get('status_message'):
            psim.TextUnformatted(ui_state['status_message'])

    ps.set_ground_plane_mode("shadow_only")
    ps.set_up_dir("z_up")
    ps.set_front_dir("x_front")
    ps.set_user_callback(callback)
    ps.set_SSAA_factor(4)
    ps.show()

if __name__ == "__main__":
    file = "bvh_debug.csv" if len(sys.argv) <= 1 else sys.argv[1]
    visualize_bvh(file)
