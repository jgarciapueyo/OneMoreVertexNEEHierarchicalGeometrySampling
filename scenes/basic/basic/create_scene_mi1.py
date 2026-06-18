"""Generate a Mitsuba 1 scene XML similar to the Mitsuba3 Python scene builder.

This script mirrors the structure of `create_scene_mi3.py` but writes an XML
file compatible with Mitsuba 1 using `xml.etree.ElementTree`.
"""
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import numpy as np

# Look at the original script: optionally flip camera for debugging
FLIP_CAM = True


def prettify(elem):
    rough = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent='\t', encoding='utf-8')


def vec_to_str(v):
    return f"{v[0]}, {v[1]}, {v[2]}" if hasattr(v, '__len__') else str(v)


def add_integrator(scene, integrator_type='additionalvertex', samples=4):
    integrator = SubElement(scene, 'integrator', type=integrator_type)

    if integrator_type == 'path':
        integrator_type = 'path'
    elif integrator_type == 'additionalvertex':
        integrator_type = 'additionalvertex'
    elif integrator_type == 'guided_path':
        SubElement(integrator, 'string', name="budgetType", value="spp")
        SubElement(integrator, 'float', name="budget", value=str(samples))

        if True:
            SubElement(integrator, 'string', name="sampleCombination", value="inversevar")
            SubElement(integrator, 'string', name="bsdfSamplingFractionLoss", value="kl")
            SubElement(integrator, 'string', name="spatialFilter", value="stochastic")
            SubElement(integrator, 'string', name="directionalFilter", value="box")
            SubElement(integrator, 'integer', name="sTreeThreshold", value="4000")
            SubElement(integrator, 'integer', name="sppPerPass", value="1")
    elif integrator_type == 'doublestep':
        SubElement(integrator, 'integer', name="maxDepth", value="3")
        SubElement(integrator, 'integer', name="rrDepth", value="5")
        SubElement(integrator, 'boolean', name="doAdditionalVertex", value="true")
        SubElement(integrator, 'boolean', name="useBVH", value="true")
        SubElement(integrator, 'boolean', name="disableNee", value="false")

def add_geometry_bvh(scene):
    geometrybvh = SubElement(scene, 'geometrybvh', type='geometrybvh')
    SubElement(geometrybvh, 'integer', name='maxLeafSize', value='4')
    SubElement(geometrybvh, 'string', name='samplingMode', value='Primitive')
    SubElement(geometrybvh, 'float', name='sggxCrossSectionScale', value='1.0')
    SubElement(geometrybvh, 'float', name='solidAngleScale', value='0.0')
    SubElement(geometrybvh, 'float', name='variancePenaltyScale', value='0.0')
    SubElement(geometrybvh, 'float', name='cosinePenaltyScale', value='1.0')

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
        # Area light positioned above a blocking plane so direct connections are impeded.
        # Create an area-emitting rectangle (faces downward).
        light_shape = SubElement(scene, 'shape', type='rectangle', name='area_light')
        t_ls = SubElement(light_shape, 'transform', name='toWorld')
        # position the area light at z=5 and point it downwards
        SubElement(t_ls, 'rotate', x='0', y='1', z='0', angle='90')
        SubElement(t_ls, 'translate', x='-5', y='0', z='1')
        # SubElement(t_ls, 'lookAt', origin='0, 0, 5', target='0, 0, 4', up='0, 1, 0')
        # SubElement(t_ls, 'scale', x='0.5', y='0.5', z='1')
        emitter = SubElement(light_shape, 'emitter', type='area')
        SubElement(emitter, 'rgb', name='radiance', value='10000')

        # Add a diffuse blocker plane directly beneath the area light so the scene
        # cannot see the emitter directly; light first hits this plane and then
        # illuminates the scene indirectly.
        blocker_shape = SubElement(scene, 'shape', type='rectangle', name='area_light')
        t_bs = SubElement(blocker_shape, 'transform', name='toWorld')
        # position the area light at z=5 and point it downwards
        SubElement(t_bs, 'scale', x='1', y='10', z='1')
        SubElement(t_bs, 'rotate', x='0', y='1', z='0', angle='90')
        SubElement(t_bs, 'translate', x='-4', y='0', z='0.9')


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


def add_quad_as_cube(scene, name, origin, target, up, scale=0.1, bsdf_is_checker=False):
    """Add a cube-shaped object in place of the rectangle quad.

    Parameters match `add_quad` so it can be used interchangeably:
      - scene, name, origin, target, up, scale, bsdf_is_checker

    The cube uses a `shape` of type `cube` and the same BSDF setup.
    """
    shape = SubElement(scene, 'shape', type='cube', name=name)
    t = SubElement(shape, 'transform', name='toWorld')
    SubElement(t, 'scale', x=str(scale), y=str(scale), z=str(0.0005))
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
    # sample random directions on sphere
    u = rng.random((n,))
    v = rng.random((n,))
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    dirs = np.vstack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))).T

    for i in range(n):
        origin = pos[i].tolist()
        target = (pos[i] + dirs[i]).tolist()
        up = [0, 0, 1]
        add_quad_as_cube(scene, f'quad_{i:05d}', origin, target, up, scale=scale, bsdf_is_checker=bsdf_checker)
    return scene


def set_up_scene(integrator='additionalvertex', simple_geo_level=1, simple_cam=True,
                 resx=720, fov=45, samples=4, checkerboard=True, radiancemeter=False, n_quads=None,
                 light_mode=0, return_xml_tree=False):
    if integrator == 'guided_path':
        scene = Element('scene', version='0.5.0')
    else:
        scene = Element('scene', version='0.6.0')

    add_integrator(scene, integrator, samples)
    add_geometry_bvh(scene)

    # Camera setup
    origin = [5, 0, 3] if not FLIP_CAM else [-5, 0, 3]
    target = [0, 0, 1]
    up = [0, 0, 1]
    add_sensor(scene, origin, target, up, resx=resx, fov=fov, samples=samples)

    add_light(scene, mode=light_mode)

    add_plane(scene, checkerboard=checkerboard)

    bsdf_checker = False
    if n_quads is not None and n_quads > 0:
        scene = get_quad_soup_no_group(scene, n=n_quads, scale=0.05, bbox_min=[-1, -1, 0], bbox_max=[1, 1, 3], bsdf_checker=bsdf_checker)
    else:
        if simple_geo_level == 1:
            scene = get_quad_soup_no_group(scene, n=1000, scale=0.03, bbox_min=[-1, -1, 0], bbox_max=[1, 1, 3], bsdf_checker=bsdf_checker)
        elif simple_geo_level == 2:
            scene = get_quad_soup_no_group(scene, n=20, scale=0.1, bbox_min=[-1, -1, 2], bbox_max=[1, 1, 3], bsdf_checker=bsdf_checker)
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
        # adjust the perspective camera film and fov already added above
        pass

    if return_xml_tree:
        return scene

    return prettify(scene)


def write_scene(path='scene.xml', **kwargs):
    xml_bytes = set_up_scene(**kwargs)
    # set_up_scene returns bytes when not returning tree
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode('utf-8')
    
    path_complete = path
    with open(path_complete, 'wb') as f:
        f.write(xml_bytes)


if __name__ == '__main__':
    # example: write a reasonably small scene
    # xml = set_up_scene(integrator='path', simple_geo_level=2, samples=4096, resx=256, fov=45, checkerboard=True, n_quads=100, light_mode=3)
    for n_quads in [1, 50, 100, 300, 500, 1000]:
        for integrator in ['path', 'bdpt', 'additional_vertex', 'doublestep', 'guided_path']:
            write_scene(path=f'scene_quads{n_quads}_integrator{integrator}.xml',
                        integrator=integrator,
                        simple_geo_level=2,
                        samples=256,
                        resx=256,
                        fov=45,
                        checkerboard=True,
                        n_quads=n_quads,
                        light_mode=3
                    )