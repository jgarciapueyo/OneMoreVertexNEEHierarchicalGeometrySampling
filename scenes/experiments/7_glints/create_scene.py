#!/usr/bin/env python3
"""Generate a glints test scene for Mitsuba 0.6.

Layout (y-up):
  - Floor: large diffuse plane at y=0
  - Spotlight: upper-right, pointing down toward the scene centre
  - Camera: front-facing, looking in the +z direction
  - Snowflakes: N small specular quads (roughconductor, small alpha)
    scattered in a volume in front of the camera, randomly oriented

The scene is designed to test the glinttracer: each snowflake quad has a
highly specular material, so only the handful of quads whose normal
happens to align with the half-vector between the spot and the camera
will produce a visible glint.  The glinttracer should find those
snowflakes efficiently via BVH sampling from the camera origin.

Coordinate convention: y-up.
  Camera at (10, 0, 5) looks toward (0, 0, 5) with up (0, 0, 1).
  Spotlight at (0, 5, 7) points toward (0, 0, 7).
  Floor at y=0.
  Snowflakes in volume x∈[-5,5], y∈[0,10], z∈[-5,-5].

Usage examples
--------------
# Glinttracer scene (default):
python scenes/experiments/7_glints/create_scene.py

# Path-tracer reference (same scene, different integrator):
python scenes/experiments/7_glints/create_scene.py --integrator path --samples 4096

# Dense snowstorm, lower roughness:
python scenes/experiments/7_glints/create_scene.py --n_snowflakes 2000 --alpha 0.02

# Ptracer reference (time-reverse of glinttracer, should match in expectation):
python scenes/experiments/7_glints/create_scene.py --integrator ptracer --samples 4096
"""

import argparse
import math
import os
import random
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Vec3 = tuple  # (float, float, float)

QUAD_OBJ = "scenes/experiments/scene_geometries/quad_2tri.obj"
SNOWFLAKE_OBJ = "scenes/experiments/7_glints/snowflake.obj"

def prettify(elem):
    raw = tostring(elem, "utf-8")
    return minidom.parseString(raw).toprettyxml(indent="\t", encoding="utf-8")


def _f(v):
    return f"{v:.6g}"


def normalize(v: Vec3) -> Vec3:
    n = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if n < 1e-12:
        return (0.0, 1.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def stable_up(normal: Vec3) -> Vec3:
    """Return a unit vector that is not parallel to *normal*, suitable as
    lookAt up vector."""
    candidates = [(0.0, 0.0, 1.0)]
    for c in candidates:
        if abs(dot(normal, c)) < 0.9:
            return c
    return candidates[0]


def sample_uniform_sphere(rng: random.Random) -> Vec3:
    z = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * math.pi)
    r = math.sqrt(max(0.0, 1.0 - z*z))
    return (r * math.cos(phi), r * math.sin(phi), z)


def sample_vmf(mean_dir: Vec3, kappa: float, rng: random.Random) -> Vec3:
    """Von Mises-Fisher sample around mean_dir with concentration kappa.
    kappa=0 → uniform sphere."""
    if kappa < 1e-6:
        return sample_uniform_sphere(rng)
    u = rng.random()
    w = 1.0 + math.log(u + (1.0 - u) * math.exp(-2.0 * kappa)) / kappa
    w = max(-1.0, min(1.0, w))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    sin_t = math.sqrt(max(0.0, 1.0 - w*w))
    # build orthonormal frame around mean_dir
    helper = (0.0, 0.0, 1.0) if abs(mean_dir[2]) < 0.999 else (1.0, 0.0, 0.0)
    t = normalize(cross(helper, mean_dir))
    b = cross(mean_dir, t)
    x = sin_t * math.cos(phi)
    y = sin_t * math.sin(phi)
    return normalize((x*t[0] + y*b[0] + w*mean_dir[0],
                      x*t[1] + y*b[1] + w*mean_dir[1],
                      x*t[2] + y*b[2] + w*mean_dir[2]))


# ---------------------------------------------------------------------------
# Scene-element builders
# ---------------------------------------------------------------------------

def add_integrator(scene, integrator_type: str):
    """Add the chosen integrator block (no sampleCount here — lives in sampler)."""
    if integrator_type == "glinttracer":
        integ = SubElement(scene, "integrator", type="glinttracer")
        SubElement(integ, "integer", name="granularity", value="200000")

    elif integrator_type == "path":
        integ = SubElement(scene, "integrator", type="path")
        # maxDepth=2: camera→surface→light (direct-only specular), same path class
        # as glinttracer.  Use maxDepth=-1 for full indirect.
        SubElement(integ, "integer", name="maxDepth", value="2")

    elif integrator_type == "ptracer":
        # Natural reference: time-reverse of glinttracer.
        # Should match glinttracer in expectation (by reciprocity).
        integ = SubElement(scene, "integrator", type="ptracer")
        SubElement(integ, "integer", name="maxDepth", value="2")

    elif integrator_type == "doublestep":
        integ = SubElement(scene, "integrator", type="doublestep")
        SubElement(integ, "integer", name="maxDepth", value="3")
        SubElement(integ, "integer", name="rrDepth", value="5")
        SubElement(integ, "boolean", name="doAdditionalVertex", value="true")
        SubElement(integ, "boolean", name="useBVH", value="true")
        SubElement(integ, "boolean", name="disableNEE", value="false")

    else:
        raise ValueError(f"Unknown integrator: {integrator_type}")


def add_geometry_bvh(scene, max_leaf_size: int = 4,
                     sampling_mode: str = "Primitive"):
    """Add the BVH plugin (required by glinttracer and doublestep)."""
    gbvh = SubElement(scene, "geometrybvh", type="geometrybvh")
    SubElement(gbvh, "integer", name="maxLeafSize",   value=str(max_leaf_size))
    SubElement(gbvh, "string",  name="samplingMode",  value=sampling_mode)
    SubElement(gbvh, "float",   name="defensivePDF",  value="0.000001")
    SubElement(gbvh, "float",   name="debug1",        value="1")   # full importance
    SubElement(gbvh, "float",   name="debug2",        value="0")   # diffuse importance
    SubElement(gbvh, "float",   name="debug3",        value="0")   # standard LUT


def add_sensor(scene, resx: int, resy: int, samples: int, fov: float = 45.0):
    sensor = SubElement(scene, "sensor", type="perspective")
    SubElement(sensor, "string", name="fovAxis", value="smaller")
    SubElement(sensor, "float", name="nearClip", value="0.0001")
    SubElement(sensor, "float", name="farClip", value="100")
    t = SubElement(sensor, "transform", name="toWorld")
    SubElement(t, "lookAt", origin="10, 0, 5", target="0, 0, 5", up="0, 0, 1")
    SubElement(sensor, "float", name="fov", value=str(fov))
    sampler = SubElement(sensor, "sampler", type="independent")
    SubElement(sampler, "integer", name="sampleCount", value=str(samples))
    film = SubElement(sensor, "film", type="hdrfilm")
    SubElement(film, "integer", name="width", value=str(resx))
    SubElement(film, "integer", name="height", value=str(resy))


def add_spotlight(scene,
                  origin: Vec3 = (0.0, 5.0, 7.0),
                  target: Vec3 = (0.0, 0.0, 7.0),
                  intensity: float = 50.0,
                  cutoff_angle: float = 35.0,
                  beam_width: float = 25.0):
    """Spot emitter.  Mitsuba's spot emits along its local -z axis, and
    lookAt maps local -z to the direction (target - origin), so this
    positions the light at *origin* pointing toward *target*."""
    emitter = SubElement(scene, "emitter", type="spot", name="spotlight")
    t = SubElement(emitter, "transform", name="toWorld")
    SubElement(t, "lookAt",
               origin=f"{_f(origin[0])}, {_f(origin[1])}, {_f(origin[2])}",
               target=f"{_f(target[0])}, {_f(target[1])}, {_f(target[2])}",
               up="0, 0, 1")
    r_v = _f(intensity)
    r_g = _f(intensity * 0.9)
    r_b = _f(intensity * 0.78)
    SubElement(emitter, "rgb",   name="intensity",    value=f"{r_v}, {r_g}, {r_b}")
    SubElement(emitter, "float", name="cutoffAngle",  value=_f(cutoff_angle))
    SubElement(emitter, "float", name="beamWidth",    value=_f(beam_width))


def add_floor(scene, size: float = 10.0, y: float = 0.0,
              color: tuple = (0.5, 0.5, 0.5),
              obj_path: str = QUAD_OBJ):
    """Large diffuse plane at y=*y*, centred at origin, spanning *size* × *size*.

    The quad_2tri.obj is a unit square in the XY plane with normal +Z.
    We scale it, rotate -90° around X (XY → XZ, normal +Z → +Y), then
    translate to the target y height.
    """
    shape = SubElement(scene, "shape", type="obj", name="floor")
    SubElement(shape, "string", name="filename", value=obj_path)
    t = SubElement(shape, "transform", name="toWorld")
    SubElement(t, "scale", x=_f(size), y=_f(size), z="1")
    SubElement(t, "rotate", x="1", y="0", z="0", angle="0")
    SubElement(t, "translate", x="0", y="0", z="0")
    bsdf = SubElement(shape, "bsdf", type="diffuse")
    SubElement(bsdf, "rgb", name="reflectance",
               value=f"{color[0]}, {color[1]}, {color[2]}")


def add_snowflakes(scene,
                   n: int,
                   size: float,
                   vol_min: Vec3,
                   vol_max: Vec3,
                   normal_kappa: float,
                   normal_mean_dir: Vec3,
                   obj_path: str,
                   seed: int,
                   base_color: tuple = (0.9, 0.9, 0.9),
                   roughness: float  = 0.05,
                   specular: float   = 1.0,
                   metallic: float   = 1.0):
    """Scatter *n* small Disney-BSDF quads in the given axis-aligned volume.

    Each snowflake is an OBJ quad scaled to *size* × *size* and given a
    random orientation sampled from a VMF with concentration *normal_kappa*
    around *normal_mean_dir* (kappa=0 → fully random / uniform sphere).

    BSDF: basicprincipled (Disney) with four parameters:
        base_color  – diffuse/metallic base colour (rgb)
        roughness   – GGX roughness (0 = mirror, 1 = fully diffuse)
        specular    – specular weight  (0 = no specular, 1 = full)
        metallic    – metallic weight  (0 = dielectric, 1 = conductor-like)

    For glinty snowflakes use low roughness (0.02–0.1), high specular and
    high metallic.
    """
    rng = random.Random(seed)

    for i in range(n):
        pos = (
            rng.uniform(vol_min[0], vol_max[0]),
            rng.uniform(vol_min[1], vol_max[1]),
            rng.uniform(vol_min[2], vol_max[2]),
        )
        normal = sample_vmf(normal_mean_dir, normal_kappa, rng)
        up = stable_up(normal)

        target = (pos[0] + normal[0],
                  pos[1] + normal[1],
                  pos[2] + normal[2])

        shape = SubElement(scene, "shape", type="obj",
                           name=f"snowflake_{i:05d}")
        SubElement(shape, "string", name="filename", value=obj_path)
        t = SubElement(shape, "transform", name="toWorld")
        SubElement(t, "scale", x=_f(size), y=_f(size), z="1")
        SubElement(t, "lookAt",
                   origin=f"{_f(pos[0])}, {_f(pos[1])}, {_f(pos[2])}",
                   target=f"{_f(target[0])}, {_f(target[1])}, {_f(target[2])}",
                   up=f"{_f(up[0])}, {_f(up[1])}, {_f(up[2])}")

        """
        # doublesided = SubElement(shape, "bsdf", type="twosided")
        bsdf = SubElement(shape, "bsdf", type="roughconductor")
        SubElement(bsdf, "float", name="alpha", value=_f(roughness))
        """
        bsdf = SubElement(shape, "bsdf", type="basicprincipled")
        SubElement(bsdf, "rgb",   name="reflectance",
                   value=f"{_f(base_color[0])}, {_f(base_color[1])}, {_f(base_color[2])}")
        SubElement(bsdf, "float", name="roughness", value=_f(roughness))
        SubElement(bsdf, "float", name="specular",  value=_f(specular))
        SubElement(bsdf, "float", name="metallic",  value=_f(metallic))


# ---------------------------------------------------------------------------
# Top-level scene assembly
# ---------------------------------------------------------------------------

def create_scene(
    integrator:       str   = "glinttracer",
    n_snowflakes:     int   = 500,
    snowflake_size:   float = 0.06,
    # Disney BSDF parameters for the snowflakes
    base_color:       tuple = (0.9, 0.9, 0.9),
    roughness:        float = 0.05,
    specular:         float = 1.0,
    metallic:         float = 1.0,
    normal_kappa:     float = 0.0,     # 0 = fully random orientations
    normal_mean_dir:  Vec3  = (0.0, 1.0, 0.0),
    vol_min:          Vec3  = (-5.0, -5.0, 0.0),
    vol_max:          Vec3  = (5.0, 5.0, 10.0),
    spot_origin:      Vec3  = (0.0,  5.0,  7.0),
    spot_target:      Vec3  = (0.0,  0.0,  7.0),
    spot_intensity:   float = 50.0,
    spot_cutoff:      float = 35.0,
    spot_beam_width:  float = 25.0,
    floor_size:       float = 10.0,
    resx:             int   = 512,
    resy:             int   = 512,
    samples:          int   = 256,
    fov:              float = 45.0,
    max_leaf_size:    int   = 4,
    sampling_mode:    str   = "Primitive",
    obj_path:         str   = QUAD_OBJ,
    seed:             int   = 42,
):
    scene = Element("scene", version="0.6.0")

    add_integrator(scene, integrator)

    # BVH is required by glinttracer and doublestep; harmless for others
    add_geometry_bvh(scene, max_leaf_size=max_leaf_size,
                     sampling_mode=sampling_mode)

    add_sensor(scene, resx=resx, resy=resy, samples=samples, fov=fov)

    add_spotlight(scene,
                  origin=(3.0,  4.0,  7.0),
                  target=(0.0,  0.0,  7.0),
                  intensity=spot_intensity,
                  cutoff_angle=spot_cutoff,
                  beam_width=spot_beam_width)
    
    """
    add_spotlight(scene,
                  origin=(2.0,  5.0,  7.0),
                  target=(5.0,  0.0,  7.0),
                  intensity=spot_intensity,
                  cutoff_angle=spot_cutoff,
                  beam_width=spot_beam_width)
    """

    add_floor(scene, size=100, y=3,
               color=(0.45, 0.45, 0.45), obj_path=QUAD_OBJ)

    add_snowflakes(scene,
                   n=n_snowflakes,
                   size=snowflake_size,
                   vol_min=vol_min,
                   vol_max=vol_max,
                   normal_kappa=normal_kappa,
                   normal_mean_dir=normal_mean_dir,
                   obj_path=obj_path,
                   seed=seed,
                   base_color=base_color,
                   roughness=roughness,
                   specular=specular,
                   metallic=metallic)

    # Add constant emitter
    # emitter = SubElement(scene, "emitter", type="constant", name="envmap")
    # SubElement(emitter, "rgb", name="radiance", value="0.003, 0.006, 0.01")
    return scene


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a glints test scene for Mitsuba 0.6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--integrator", default="path",
                        choices=["glinttracer", "path", "ptracer", "doublestep"],
                        help="Integrator to embed in the scene XML.")
    parser.add_argument("--n_snowflakes", type=int, default=10000,
                        help="Number of snowflake quads.")
    parser.add_argument("--snowflake_size", type=float, default=0.0001,
                        help="Half-size of each snowflake quad (world units).")
    parser.add_argument("--base_color", nargs=3, type=float,
                        default=[0.99, 0.99, 0.99], metavar=("R", "G", "B"),
                        help="Disney BSDF base colour (rgb, 0–1).")
    parser.add_argument("--roughness", type=float, default=0.01,
                        help="Disney roughness (0=mirror, 1=diffuse). "
                             "Lower → sharper glints (try 0.01–0.1).")
    parser.add_argument("--specular", type=float, default=0.0,
                        help="Disney specular weight (0–1).")
    parser.add_argument("--metallic", type=float, default=0.0,
                        help="Disney metallic weight (0=dielectric, 1=conductor-like).")
    parser.add_argument("--normal_kappa", type=float, default=4.0,
                        help="VMF concentration for snowflake normals. "
                             "0 = fully random (uniform sphere). "
                             "High kappa (e.g. 50) biases normals toward "
                             "--normal_mean_dir.")
    parser.add_argument("--normal_mean_dir", nargs=3, type=float,
                        default=[1.0, 1.0, 0.0],
                        metavar=("NX", "NY", "NZ"),
                        help="Mean normal direction for VMF (only used when "
                             "normal_kappa > 0). Default: pointing up.")
    parser.add_argument("--spot_intensity", type=float, default=10000.0,
                        help="Spot light intensity (W/sr).")
    parser.add_argument("--spot_cutoff", type=float, default=15.0,
                        help="Spot cutoff angle (degrees).")
    parser.add_argument("--samples", type=int, default=256,
                        help="Samples per pixel (spp).")
    parser.add_argument("--resx", type=int, default=512)
    parser.add_argument("--resy", type=int, default=512)
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--max_leaf_size", type=int, default=4)
    parser.add_argument("--sampling_mode", default="Primitive",
                        choices=["Primitive", "SphericalAABB"])
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for snowflake placement.")
    parser.add_argument("--obj_path", default=SNOWFLAKE_OBJ,
                        help="Path to snowflake OBJ (relative to Mitsuba working dir).")
    parser.add_argument("--output", default=None,
                        help="Output XML path.  Default: "
                             "7_glints/glints_<integrator>_N<n>_a<alpha>.xml")

    args = parser.parse_args()

    normal_mean = tuple(args.normal_mean_dir)
    # normalize in case the user passed a non-unit vector
    n = math.sqrt(sum(x*x for x in normal_mean))
    if n > 1e-9:
        normal_mean = tuple(x/n for x in normal_mean)

    scene = create_scene(
        integrator=args.integrator,
        n_snowflakes=args.n_snowflakes,
        snowflake_size=args.snowflake_size,
        base_color=tuple(args.base_color),
        roughness=args.roughness,
        specular=args.specular,
        metallic=args.metallic,
        normal_kappa=args.normal_kappa,
        normal_mean_dir=normal_mean,
        spot_intensity=args.spot_intensity,
        spot_cutoff=args.spot_cutoff,
        spot_beam_width=args.spot_cutoff * 0.7,
        samples=args.samples,
        resx=args.resx,
        resy=args.resy,
        fov=args.fov,
        max_leaf_size=args.max_leaf_size,
        sampling_mode=args.sampling_mode,
        obj_path=args.obj_path,
        seed=args.seed,
    )

    xml_bytes = prettify(scene)
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8")

    size_tag = f"{args.snowflake_size:.6f}".replace(".", "")
    metallic_tag = f"m{args.metallic:.1f}".replace(".", "")
    specular_tag = f"s{args.specular:.1f}".replace(".", "")
    rough_tag = str(args.roughness).replace(".", "")
    default_name = (f"glints_{args.integrator}"
                    f"_N{args.n_snowflakes}"
                    f"_size{size_tag}"
                    f"_m{metallic_tag}"
                    f"_s{specular_tag}"
                    f"_r{rough_tag}"
                    f"_spp{args.samples}.xml")
    out_path = args.output or os.path.join(os.path.dirname(__file__), default_name)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(xml_bytes)

    print(f"Wrote {out_path}")
    print(f"  integrator  : {args.integrator}")
    print(f"  snowflakes  : {args.n_snowflakes} quads  (size={args.snowflake_size})")
    print(f"  material    : basicprincipled  "
          f"roughness={args.roughness}  specular={args.specular}  "
          f"metallic={args.metallic}  base_color={args.base_color}")
    print(f"  normal kappa: {args.normal_kappa}  "
          f"mean={[round(x,3) for x in normal_mean]}")
    print(f"  spotlight   : intensity={args.spot_intensity}  "
          f"cutoff={args.spot_cutoff}°")
    print(f"  render      : {args.resx}×{args.resy}  spp={args.samples}")


if __name__ == "__main__":
    main()
