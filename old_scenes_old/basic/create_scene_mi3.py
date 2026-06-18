import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import numpy as np


# Set to true to look at the scene from the side of the light (for debugging):
FLIP_CAM = True 


def get_quad_soup_no_group(scene_dict, bbox_min=[-1,-1,-1], bbox_max=[1,1,1], n=10, scale=.1, bsdf={"type":"diffuse","reflectance":{"type":"rgb","value":0.8}}):
    """ Add n quads randomly in the scene within the given bounding box """
    rng = np.random.default_rng(0)
    pos = rng.uniform(bbox_min, bbox_max, size=(n,3))
    direction = mi.warp.square_to_uniform_sphere(mi.Point2f(rng.random((2,n)))).numpy().reshape(n,3)

    quads = {
        f"quad_{i:05d}" : {
            "type" : "rectangle",
            "to_world" : mi.ScalarTransform4f().look_at(
                origin=pos[i],
                target=pos[i] + direction[i],
                up=[0,0,1]
            ).scale(scale),
            "bsdf" : bsdf # important: same bsdf. Otherwise, they do not get merged 
        }
        for i in range(n)
    }

    scene_dict.update(quads)
    return scene_dict


def get_quad_soup_merged(scene_dict, bbox_min=[-1,-1,-1], bbox_max=[1,1,1], n=10, scale=.1, bsdf={"type":"diffuse","reflectance":{"type":"rgb","value":0.8}}):
    """ 
    Add n quads randomly in the scene within the given bounding box 
    Merges them into a single mesh - much faster. However, it seems like mitsuba does this automaticaly, so this and get_quad_soup_no_group are exactly equivalent in practice 
    """
    rng = np.random.default_rng(0)
    pos = rng.uniform(bbox_min, bbox_max, size=(n,3))
    direction = mi.warp.square_to_uniform_sphere(mi.Point2f(rng.random((2,n)))).numpy().reshape(n,3)

    # use the secret type of shape: merge
    quads = {"type" : "merge"}
    for i in range(n):
        quads[f"quad_{i:05d}"] = {
            "type" : "rectangle",
            "to_world" : mi.ScalarTransform4f().look_at(
                origin=pos[i],
                target=pos[i] + direction[i],
                up=[0,0,1]
            ).scale(scale),
            "bsdf" : bsdf # important: same bsdf. Otherwise, they do not get merged 
        }
    scene_dict["quads_merged"] = quads
    
    return scene_dict


def light_0():
    return {
        'type' : 'spot',
        'to_world' : mi.ScalarTransform4f.look_at(
            origin=[-5, 0, 2],
            target=[2, 0, 5],
            up=[0, 0, 1]
        ),
        'intensity' : {
            'type' : 'rgb',
            'value' : 25000
        },
    }


def light_1():
    return {
        'type' : 'spot',
        'to_world' : mi.ScalarTransform4f.look_at(
            origin=[-5, 0, 2],
            target=[2, 0, 5],
            up=[0, 0, 1]
        ),
        'intensity' : {
            'type' : 'rgb',
            'value' : 250
        },
        'cutoff_angle': 40 # default is 20
    }


def set_up_scene(integrator: str, simple_geo_level=1, simple_cam=True, resx=720, fov=45, checkerboard=True, radiancemeter=False, n_quads=None,
                 optimize_geometry=False, light_mode=0, return_dict=False):
    """ 
    simple_geo_level: can go from 1 (hard) to 3 (easy) 
    simple_cam: determines whether to use a simple camera setup: few pixels looking at the floor, very narrow FOV
    resx: resolution of the output image (resx x resx)
    fov: field of view of the perspective camera
    checkerboard: whether to use a checkerboard texture on the floor (harder)
    radiancemeter: whether to use a radiancemeter instead of a perspective camera
    n_quads: if not None, number of quads to add in the scene (overrides simple_geo_level)
    optimize_geometry: whether to enable mitsuba's automatic geometry optimization (merging, etc)
    return_dict: if True, return the scene dictionary instead of the mitsuba Scene object
    light_mode: 0 or 1, selects between two different light intensities
    """
    floor_reflectance = ({'type': 'rgb', 'value': 0.8} if not checkerboard else 
                         {'type': 'checkerboard', 'to_uv': mi.ScalarAffineTransform4f().scale([50, 50, 1])})
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': integrator,
        },
        'camera': {
            'type': 'perspective',
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[5, 0, 3] if not FLIP_CAM else [-5, 0, 3],
                target=[0, 0, 1],
                up=[0, 0, 1]
            ),
            'fov': fov,
            'sampler': {
                'type': 'independent',
                'sample_count': 4
            },
            'film': {
                'type': 'hdrfilm',
                'width': resx,
                'height': resx,
                'filter': { 'type': 'gaussian' }    
            }
        },
        'light' : light_0() if light_mode==0 else light_1(),
        'plane': {
            'type': 'rectangle',
            'to_world': mi.ScalarTransform4f().translate([0, 0, 0]).scale([10, 10, 1]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': floor_reflectance
            }
        }
    }
    
    bsdf = {
        'type': 'diffuse',
        'reflectance': {
            'type': 'rgb',
            'value': 0.8
        }
    }
    # 'type' : 'twosided',
    # 'mat' : {
    #     'type': 'diffuse',
    #     'reflectance': {
    #         'type': 'rgb',
    #         'value': 0.8
    #     }
    # } 
    if n_quads is not None and n_quads > 0:
        print(f"Adding {n_quads} quads to the scene")
        scene_dict = get_quad_soup_no_group(scene_dict, n=n_quads, scale=.05, bbox_min=[-1,-1,0], bbox_max=[1,1,3], bsdf=bsdf)
    elif n_quads is None:
        # scene_dict = get_quad_soup_no_group(scene_dict, n=5000, scale=.01, bbox_min=[-1,-1,0], bbox_max=[1,1,3])
        if simple_geo_level == 1:
            scene_dict = get_quad_soup_no_group(scene_dict, n=1000, scale=.03, bbox_min=[-1,-1,0], bbox_max=[1,1,3], bsdf=bsdf)
        elif simple_geo_level == 2:
            scene_dict = get_quad_soup_no_group(scene_dict, n=20, scale=.1, bbox_min=[-1,-1,2], bbox_max=[1,1,3], bsdf=bsdf)
        elif simple_geo_level == 3:
            scene_dict['scene_geometry'] = {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f().look_at(
                    origin=[0, 0, 1.5],
                    target=[-5, 0, 1.5],
                    up=[0, 1, 0]
                ).scale(10), 
                'bsdf': bsdf
            }
        else:
            raise ValueError("simple_geo_level must be 1, 2, or 3")
    
    if radiancemeter:
        target = [0, 0, 1] if not FLIP_CAM else [0, 0, -1]
        origin = [5, 0, 3] if not FLIP_CAM else [-5, 0, 3]
        scene_dict['camera'] = {
            'type': 'radiancemeter',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=origin,
                target=target,
                up=[0, 0, 1]
            ),
            'film': {
                'type': 'hdrfilm', 'width': 1, 'height': 1, 'filter': { 'type': 'box' }
            }
        }
    elif simple_cam:
        scene_dict['camera']['film']['width'] = resx
        scene_dict['camera']['film']['height'] = resx
        scene_dict['camera']['fov'] = 1
        if FLIP_CAM:
            scene_dict['camera']['to_world'] = mi.ScalarTransform4f.look_at(
                origin=[-3, 0, 1.5],
                target=[0, 0, -1], # a bit closer to the cam
                up=[0, 0, 1]
            )
        else:
            scene_dict['camera']['to_world'] = mi.ScalarTransform4f.look_at(
                origin=[3, 0, 1.5],
                target=[0, 0, 1], # a bit closer to the cam
                up=[0, 0, 1]
            )
    

    return mi.load_dict(scene_dict, optimize=optimize_geometry) if not return_dict else scene_dict
