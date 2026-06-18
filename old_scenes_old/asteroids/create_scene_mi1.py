"""Generate a Mitsuba 1 scene XML similar to the basic scene, but using a PLY asteroid mesh.

This mirrors create_scene_mi1.py but replaces the ground plane with a provided PLY mesh
(e.g. shapes/Astroid.002-Rock.019-0.ply).
"""

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import numpy as np
import os
from plyfile import PlyData, PlyElement

# Optionally flip camera for debugging
FLIP_CAM = False


def prettify(elem):
    rough = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent='\t', encoding='utf-8')


def vec_to_str(v):
    return f"{v[0]}, {v[1]}, {v[2]}" if hasattr(v, '__len__') else str(v)


def add_integrator(scene, integrator_type='additionalvertex', samples=4):
    integrator = SubElement(scene, 'integrator', type=integrator_type)

    if integrator_type == 'path':
        pass
    elif integrator_type == 'additionalvertex':
        pass
    elif integrator_type == 'guided_path':
        SubElement(integrator, 'string', name="budgetType", value="spp")
        SubElement(integrator, 'float', name="budget", value=str(samples))
        SubElement(integrator, 'string', name="sampleCombination", value="inversevar")
        SubElement(integrator, 'string', name="bsdfSamplingFractionLoss", value="kl")
        SubElement(integrator, 'string', name="spatialFilter", value="stochastic")
        SubElement(integrator, 'string', name="directionalFilter", value="box")
        SubElement(integrator, 'integer', name="sTreeThreshold", value="4000")
        SubElement(integrator, 'integer', name="sppPerPass", value="1")


def add_sensor(scene, origin, target, up, resx=512, fov=45, samples=4):
    sensor = SubElement(scene, 'sensor', type='perspective')
    SubElement(sensor, 'string', name='fovAxis', value='smaller')
    SubElement(sensor, 'float', name='nearClip', value='0.0001')
    SubElement(sensor, 'float', name='farClip', value='1000')

    transform = SubElement(sensor, 'transform', name='toWorld')
    SubElement(transform, 'lookAt', origin=vec_to_str(origin), target=vec_to_str(target), up=vec_to_str(up))

    SubElement(sensor, 'float', name='fov', value=str(fov))

    sampler = SubElement(sensor, 'sampler', type='independent')
    SubElement(sampler, 'integer', name='sampleCount', value=str(samples))

    film = SubElement(sensor, 'film', type='hdrfilm')
    SubElement(film, 'integer', name='width', value=str(resx))
    SubElement(film, 'integer', name='height', value=str(resx))


def add_light(scene, mode=0):
    if mode == 0:
        emitter = SubElement(scene, 'emitter', type='spot')
        transform = SubElement(emitter, 'transform', name='toWorld')
        SubElement(transform, 'lookAt', origin='-5, 0, 2', target='2, 0, 5', up='0, 0, 1')
        SubElement(emitter, 'rgb', name='intensity', value='25000')
    elif mode == 1:
        emitter = SubElement(scene, 'emitter', type='spot')
        transform = SubElement(emitter, 'transform', name='toWorld')
        SubElement(transform, 'lookAt', origin='-5, 0, 2', target='2, 0, 5', up='0, 0, 1')
        SubElement(emitter, 'rgb', name='intensity', value='250')
        SubElement(emitter, 'float', name='cutoffAngle', value='40')
    elif mode == 2:
        emitter = SubElement(scene, 'emitter', type='constant')
        SubElement(emitter, 'rgb', name='irradiance', value='1')
    elif mode == 3:
        # Area light above the scene with a blocker rectangle below it
        light_shape = SubElement(scene, 'shape', type='rectangle', name='area_light')
        t_ls = SubElement(light_shape, 'transform', name='toWorld')
        SubElement(t_ls, 'rotate', x='0', y='1', z='0', angle='90')
        SubElement(t_ls, 'translate', x='-5', y='0', z='1')
        emitter = SubElement(light_shape, 'emitter', type='area')
        SubElement(emitter, 'rgb', name='radiance', value='10000')

        blocker_shape = SubElement(scene, 'shape', type='rectangle', name='area_blocker')
        t_bs = SubElement(blocker_shape, 'transform', name='toWorld')
        SubElement(t_bs, 'scale', x='1', y='10', z='1')
        SubElement(t_bs, 'rotate', x='0', y='1', z='0', angle='90')
        SubElement(t_bs, 'translate', x='-4', y='0', z='0.9')
        bsdf = SubElement(blocker_shape, 'bsdf', type='diffuse')
        SubElement(bsdf, 'rgb', name='reflectance', value='0.8')


def add_plane(scene, checkerboard=True):
    plane = SubElement(scene, 'shape', type='rectangle')
    transform = SubElement(plane, 'transform', name='toWorld')
    SubElement(transform, 'scale', x='10', y='10', z='1')
    SubElement(transform, 'translate', x='0', y='0', z='0')

    bsdf = SubElement(plane, 'bsdf', type='diffuse')
    if checkerboard:
        tex = SubElement(bsdf, 'texture', type='checkerboard', name='reflectance')
        SubElement(tex, 'float', name='uscale', value='50')
        SubElement(tex, 'float', name='vscale', value='50')
    else:
        SubElement(bsdf, 'rgb', name='reflectance', value='0.8')


def create_floor_ply(path='floor.ply', size=10.0, checkerboard=True):
    """
    Generate a PLY file for a rectangular floor mesh.
    - size: Half-extent of the square (e.g., 10.0 for 10x10).
    - checkerboard: If True, add UV coordinates for texture mapping.
    """
    # Define vertices for a quad in XY plane (Z=0), centered at origin
    half_size = size / 2.0
    verts = np.array([
        (-half_size, -half_size, 0.0),  # Bottom-left
        ( half_size, -half_size, 0.0),  # Bottom-right
        ( half_size,  half_size, 0.0),  # Top-right
        (-half_size,  half_size, 0.0)   # Top-left
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    # Add UV coordinates if checkerboard is enabled
    if checkerboard:
        uvs = np.array([
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)
        ], dtype=[('u', 'f4'), ('v', 'f4')])
        # Merge position and uv fields into a single structured array
        verts = verts.copy()
        verts = verts.reshape(4)
        verts = verts.astype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('u', 'f4'), ('v', 'f4')])
        verts['x'] = np.array([v[0] for v in [(-half_size, -half_size, 0.0), ( half_size, -half_size, 0.0), ( half_size,  half_size, 0.0), (-half_size,  half_size, 0.0)]], dtype='f4')
        verts['y'] = np.array([v[1] for v in [(-half_size, -half_size, 0.0), ( half_size, -half_size, 0.0), ( half_size,  half_size, 0.0), (-half_size,  half_size, 0.0)]], dtype='f4')
        verts['z'] = np.array([v[2] for v in [(-half_size, -half_size, 0.0), ( half_size, -half_size, 0.0), ( half_size,  half_size, 0.0), (-half_size,  half_size, 0.0)]], dtype='f4')
        verts['u'] = uvs['u']
        verts['v'] = uvs['v']

    # Define faces (two triangles for the quad)
    faces = np.array([([0, 1, 2],), ([0, 2, 3],)], dtype=[('vertex_indices', 'i4', (3,))])

    # Create PlyElements
    vertex_element = PlyElement.describe(verts, 'vertex')
    face_element = PlyElement.describe(faces, 'face')

    # Write binary PLY for efficiency
    PlyData([vertex_element, face_element], text=False).write(path)
    print(f"Floor PLY generated at: {path}")


def add_ply_plane(scene, checkerboard=True):
    """
    Add a PLY-based floor plane to the scene by generating the PLY on the fly.
    """
    ply_path = 'floor.ply'  # Relative path; generated in script dir
    create_floor_ply(path=ply_path, size=10.0, checkerboard=checkerboard)
    add_ply(scene, ply_path, name='floor', scale=1.0, translate=(0, 0, 0), bsdf_checker=checkerboard, random_rotate=False)


def add_ply(scene, ply_path, name='asteroid', scale=1.0, translate=(0, 0, 0), bsdf_checker=False, random_rotate=True):
    """Add a PLY mesh to the scene. ply_path is the filesystem path to the .ply file."""
    shape = SubElement(scene, 'shape', type='ply', name=name)
    # filename string tag expected by Mitsuba 1
    SubElement(shape, 'string', name='filename', value=ply_path)
    t = SubElement(shape, 'transform', name='toWorld')
    if random_rotate:
        SubElement(t, 'rotate', x='1', y='0', z='0', angle=str(np.random.uniform(0, 360)))  # flip upside down
        SubElement(t, 'rotate', x='0', y='1', z='0', angle=str(np.random.uniform(0, 360)))  # random rotation around y
        SubElement(t, 'rotate', x='0', y='0', z='1', angle=str(np.random.uniform(0, 360)))  # random rotation around z
    SubElement(t, 'scale', x=str(scale), y=str(scale), z=str(scale))
    SubElement(t, 'translate', x=str(translate[0]), y=str(translate[1]), z=str(translate[2]))
    
    bsdf = SubElement(shape, 'bsdf', type='diffuse')
    if bsdf_checker:
        tex = SubElement(bsdf, 'texture', type='checkerboard', name='reflectance')
        t2 = SubElement(tex, 'transform', name='toUV')
        SubElement(t2, 'scale', x='10', y='10', z='1')
    else:
        SubElement(bsdf, 'rgb', name='reflectance', value='0.8')


def get_asteroid_soup_no_group(scene, ply_path, bbox_min=[-1, -1, -1], bbox_max=[1, 1, 1], n=10, scale=0.1, bsdf_checker=False):
    rng = np.random.default_rng(0)
    pos = rng.uniform(low=bbox_min, high=bbox_max, size=(n, 3))
    u = rng.random((n,))
    v = rng.random((n,))
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    dirs = np.vstack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))).T

    for i in range(n):
        origin = pos[i].tolist()
        target = (pos[i] + dirs[i]).tolist()
        up = [0, 0, 1]
        add_ply(scene, ply_path, name=f'asteroid_{i:05d}', scale=scale, translate=origin, bsdf_checker=bsdf_checker)
        # add_quad(scene, f'quad_{i:05d}', origin, target, up, scale=scale, bsdf_is_checker=bsdf_checker)
    return scene


def add_quad(scene, name, origin, target, up, scale=0.1, bsdf_is_checker=False):
    shape = SubElement(scene, 'shape', type='rectangle', name=name)
    t = SubElement(shape, 'transform', name='toWorld')
    SubElement(t, 'scale', x=str(scale), y=str(scale), z=str(scale))
    SubElement(t, 'lookAt', origin=vec_to_str(origin), target=vec_to_str(target), up=vec_to_str(up))

    bsdf = SubElement(shape, 'bsdf', type='diffuse')
    if bsdf_is_checker:
        tex = SubElement(bsdf, 'texture', type='checkerboard', name='reflectance')
        t2 = SubElement(tex, 'transform', name='toUV')
        SubElement(t2, 'scale', x='10', y='10', z='1')
    else:
        SubElement(bsdf, 'rgb', name='reflectance', value='0.8')


def get_quad_soup_no_group(scene, bbox_min=[-1, -1, -1], bbox_max=[1, 1, 1], n=10, scale=0.1, bsdf_checker=False):
    rng = np.random.default_rng(0)
    pos = rng.uniform(low=bbox_min, high=bbox_max, size=(n, 3))
    u = rng.random((n,))
    v = rng.random((n,))
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    dirs = np.vstack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))).T

    for i in range(n):
        origin = pos[i].tolist()
        target = (pos[i] + dirs[i]).tolist()
        up = [0, 0, 1]
        add_quad(scene, f'quad_{i:05d}', origin, target, up, scale=scale, bsdf_is_checker=bsdf_checker)
    return scene


def set_up_scene(integrator='additionalvertex', simple_geo_level=1, simple_cam=True,
                 resx=720, fov=45, samples=4, checkerboard=False, radiancemeter=False,
                 n_quads=None, light_mode=0, ply_path=None, ply_scale=1.0, return_xml_tree=False):
    scene = Element('scene', version='0.6.0')
    add_integrator(scene, integrator, samples)

    origin = [-5, 0, 3] if FLIP_CAM else [0, 5, 3]
    target = [0, 0, 1]
    up = [0, 0, 1]
    add_sensor(scene, origin, target, up, resx=resx, fov=fov, samples=samples)

    add_light(scene, mode=light_mode)

    add_ply_plane(scene, checkerboard=checkerboard)

    # Default ply path if not provided. Relative to this script: ../shapes/...
    if ply_path is None:
        default_p = os.path.join(os.path.dirname(__file__), '..', 'shapes', 'Astroid.002-Rock.019-0.ply')
        ply_path = os.path.normpath(default_p)

    # Add asteroid mesh instead of a plane. Position/scale vary by simple_geo_level.
    bsdf_checker = False
    if n_quads is not None and n_quads > 0:
        scene = get_asteroid_soup_no_group(scene, ply_path, n=n_quads, scale=0.04, bbox_min=[-2, -2, 0], bbox_max=[2, 2, 5], bsdf_checker=bsdf_checker)
    else:
        if simple_geo_level == 1:
            scene = get_asteroid_soup_no_group(scene, ply_path, n=1000, scale=0.03, bbox_min=[-1, -1, 0], bbox_max=[1, 1, 3], bsdf_checker=bsdf_checker)
        elif simple_geo_level == 2:
            scene = get_asteroid_soup_no_group(scene, ply_path, n=20, scale=0.1, bbox_min=[-1, -1, 2], bbox_max=[1, 1, 3], bsdf_checker=bsdf_checker)
        elif simple_geo_level == 3:
            # single large rectangle acting as geometry
            shape = SubElement(scene, 'shape', type='rectangle', name='scene_geometry')
            t = SubElement(shape, 'transform', name='toWorld')
            SubElement(t, 'lookAt', origin='0, 0, 1.5', target='-5, 0, 1.5', up='0, 1, 0')
            SubElement(t, 'scale', x='10', y='10', z='1')
            bsdf = SubElement(shape, 'bsdf', type='diffuse')
            SubElement(bsdf, 'rgb', name='reflectance', value='0.8')
        else:
            raise ValueError('simple_geo_level must be 1, 2, or 3')

    if radiancemeter:
        origin_rm = [5, 0, 3] if not FLIP_CAM else [-5, 0, 3]
        target_rm = [0, 0, 1] if not FLIP_CAM else [0, 0, -1]
        sensor = SubElement(scene, 'sensor', type='radiancemeter')
        transform = SubElement(sensor, 'transform', name='toWorld')
        SubElement(transform, 'lookAt', origin=vec_to_str(origin_rm), target=vec_to_str(target_rm), up='0, 0, 1')
        film = SubElement(sensor, 'film', type='hdrfilm')
        SubElement(film, 'integer', name='width', value='1')
        SubElement(film, 'integer', name='height', value='1')
        SubElement(film, 'filter', type='box')
    elif simple_cam:
        pass

    if return_xml_tree:
        return scene

    return prettify(scene)


def write_scene(path='scene.xml', **kwargs):
    xml_bytes = set_up_scene(**kwargs)
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode('utf-8')

    integrator = str(kwargs.get('integrator', 'unknown'))
    path_complete = integrator + '_' + path
    with open(path_complete, 'wb') as f:
        f.write(xml_bytes)


if __name__ == '__main__':
    # Example usage: writes guided_path_asteroid_scene.xml with the asteroid PLY (adjust ply_path if needed)
    default_ply = os.path.join(os.path.dirname(__file__), 'shapes', 'Astroid.002-Rock.019-0.ply')
    write_scene(path='asteroid_scene.xml', integrator='path', simple_geo_level=1,
                samples=4096, resx=512, fov=45, checkerboard=False, n_quads=300,
                light_mode=3, ply_path=os.path.normpath(default_ply), ply_scale=1.0)