"""Generate a Cornell Box-like scene for Mitsuba 0.6.

Each of the 5 walls (back, floor, ceiling, left, right) is composed of N×N
mini-quads instead of a single large quad, so our technique can operate on
scenes with small primitives.

Coordinate system: y-up, room spans [-1,1] in x/y/z.
Camera at (0, 0, -3) looking toward origin.

Transform recipe: Mitsuba applies listed transforms in order (first = applied
first to object points).  For each tile:
  scale(hs, hs, 1)           -- shrink unit rect to tile half-size
  translate(c1, c2, 0)       -- move to 2D tile center in wall-local plane
  rotate_axis(angle)         -- orient the wall plane
  translate(wall_position)   -- shift to wall's world position
"""

import argparse
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import numpy as np


def prettify(elem):
    rough = tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="\t", encoding="utf-8")


def _f(v):
    """Format a float compactly."""
    return f"{v:.6g}"


def add_integrator(scene, integrator_type="additionalvertex"):
    integrator = SubElement(scene, "integrator", type=integrator_type)
    if integrator_type in ("additionalvertex", "path", "bdpt"):
        SubElement(integrator, "integer", name="maxDepth", value="5")
    elif integrator_type == "doublestep":
        SubElement(integrator, "integer", name="maxDepth", value="5")
        SubElement(integrator, "integer", name="rrDepth", value="5")
        SubElement(integrator, "boolean", name="doAdditionalVertex", value="true")
        SubElement(integrator, "boolean", name="useBVH", value="true")
        SubElement(integrator, "boolean", name="disableNee", value="false")


def add_geometry_bvh(scene):
    gbvh = SubElement(scene, "geometrybvh", type="geometrybvh")
    SubElement(gbvh, "integer", name="maxLeafSize", value="4")
    SubElement(gbvh, "string", name="samplingMode", value="Primitive")
    SubElement(gbvh, "float", name="sggxCrossSectionScale", value="1.0")
    SubElement(gbvh, "float", name="solidAngleScale", value="0.0")
    SubElement(gbvh, "float", name="variancePenaltyScale", value="0.0")
    SubElement(gbvh, "float", name="cosinePenaltyScale", value="1.0")


def add_sensor(scene, resx=512, resy=512, samples=64, fov=45):
    sensor = SubElement(scene, "sensor", type="perspective")
    SubElement(sensor, "string", name="fovAxis", value="smaller")
    SubElement(sensor, "float", name="nearClip", value="0.0001")
    SubElement(sensor, "float", name="farClip", value="100")
    t = SubElement(sensor, "transform", name="toWorld")
    SubElement(t, "lookAt", origin="0, 0, -3", target="0, 0, 0", up="0, 1, 0")
    SubElement(sensor, "float", name="fov", value=str(fov))
    sampler = SubElement(sensor, "sampler", type="independent")
    SubElement(sampler, "integer", name="sampleCount", value=str(samples))
    film = SubElement(sensor, "film", type="hdrfilm")
    SubElement(film, "integer", name="width", value=str(resx))
    SubElement(film, "integer", name="height", value=str(resy))


def add_area_light(scene, obj_path, half_size=0.25, radiance=20.0):
    """Single rectangle emitter near the ceiling center, facing downward."""
    shape = SubElement(scene, "shape", type="obj", name=f"area_light")
    SubElement(shape, "string", name="filename", value=obj_path)
    t = SubElement(shape, "transform", name="toWorld")
    # scale to desired size, then orient downward (rotate_x -90° → normal faces -y),
    # then lift to y=0.98
    SubElement(t, "scale", x=_f(half_size), y=_f(half_size), z="1")
    SubElement(t, "rotate", x="1", y="0", z="0", angle="90")
    SubElement(t, "translate", x="0.0", y="0.98", z="0")
    emitter = SubElement(shape, "emitter", type="area")
    SubElement(emitter, "rgb", name="radiance", value=f"{radiance}, {radiance}, {radiance}")


OBJ_PATH = "scenes/experiments/scene_geometries/quad_2tri.obj"


def add_wall(scene, wall, n_tiles, tile_scale, color, obj_path=OBJ_PATH):
    """Add n_tiles×n_tiles mini-quads for one wall.

    wall: one of 'back', 'floor', 'ceiling', 'left', 'right'
    tile_scale: scale factor for tile size
    color: (r, g, b) tuple
    obj_path: path to quad OBJ file (relative to Mitsuba working directory)
    """
    hs = 1.0 / n_tiles  # half tile size (room half-size = 1.0)
    centers = np.linspace(-(1.0 - hs), 1.0 - hs, n_tiles)
    color_str = f"{color[0]}, {color[1]}, {color[2]}"

    for i, c1 in enumerate(centers):
        for j, c2 in enumerate(centers):
            shape = SubElement(scene, "shape", type="obj",
                               name=f"wall_{wall}_{i:03d}_{j:03d}")
            SubElement(shape, "string", name="filename", value=obj_path)
            t = SubElement(shape, "transform", name="toWorld")

            # Step 1: scale unit rect to tile half-size
            SubElement(t, "scale", x=_f(hs * tile_scale), y=_f(hs * tile_scale), z="1")

            # Steps 2-4 depend on which wall we're building
            if wall == "back":
                # XY plane at z=1; rotate_y 180° flips normal to face -z (toward camera)
                SubElement(t, "translate", x=_f(c1), y=_f(c2), z="0")
                SubElement(t, "rotate", x="0", y="1", z="0", angle="180")
                SubElement(t, "translate", x="0", y="0", z="1")

            elif wall == "floor":
                # rotate_x 90°: XY → XZ plane, normal +z → +y; then shift to y=-1
                SubElement(t, "translate", x=_f(c1), y=_f(c2), z="0")
                SubElement(t, "rotate", x="1", y="0", z="0", angle="-90")
                SubElement(t, "translate", x="0", y="-1", z="0")

            elif wall == "ceiling":
                # rotate_x -90°: XY → XZ plane, normal +z → -y; then shift to y=1
                SubElement(t, "translate", x=_f(c1), y=_f(c2), z="0")
                SubElement(t, "rotate", x="1", y="0", z="0", angle="90")
                SubElement(t, "translate", x="0", y="1", z="0")

            elif wall == "left":
                # rotate_y 90°: normal +z → +x (faces inward); then shift to x=-1
                SubElement(t, "translate", x=_f(c1), y=_f(c2), z="0")
                SubElement(t, "rotate", x="0", y="1", z="0", angle="90")
                SubElement(t, "translate", x="-1", y="0", z="0")

            elif wall == "right":
                # rotate_y -90°: normal +z → -x (faces inward); then shift to x=1
                SubElement(t, "translate", x=_f(c1), y=_f(c2), z="0")
                SubElement(t, "rotate", x="0", y="1", z="0", angle="-90")
                SubElement(t, "translate", x="1", y="0", z="0")

            bsdf = SubElement(shape, "bsdf", type="diffuse")
            SubElement(bsdf, "rgb", name="reflectance", value=color_str)


def add_cube(scene, x=0.0, y=0.0, z=0.0, scale_y=1.0, rot_y=0.0,
             color=(0.725, 0.725, 0.725), obj_path=OBJ_PATH,
             name="cube"):
    """Add a box with one quad per face.

    Local-space dimensions: x∈[-1,1], z∈[-1,1], y∈[0, 2·scale_y].
    The base center (0, 0, 0) is placed at world position (x, y, z) after
    rotating rot_y degrees around the y-axis.

    The unit quad (quad_2tri.obj) spans [-1,1]×[-1,1] in XY with normal +Z.
    Transforms below map it onto each face (Mitsuba applies in listed order):

      bottom  scale(1, 1, 1); rotate_x(-90°) → xz-plane, normal +y
      top     scale(1, 1, 1); rotate_x(+90°) → xz-plane, normal -y;
              translate(0, 2·sy, 0)
      front   scale(1, sy, 1); rotate_y(180°) → normal -z;
              translate(0, sy, -1)
      back    scale(1, sy, 1); translate(0, sy, +1)
      left    scale(1, sy, 1); rotate_y(+90°) → yz-plane, normal +x;
              translate(-1, sy, 0)
      right   scale(1, sy, 1); rotate_y(-90°) → yz-plane, normal -x;
              translate(+1, sy, 0)
    """
    color_str = f"{color[0]}, {color[1]}, {color[2]}"

    faces = [
        ("top", [
            ("scale",     {"x": "0.25",    "y": "0.25",    "z": "1"}),
            ("rotate",    {"x": "1", "y": "0", "z": "0", "angle": "-90"}),
            ("translate", {"x": "0", "y": str(0.5*scale_y), "z": "0"}),
        ]),
        ("bottom", [
            ("scale",     {"x": "0.25",    "y": "0.25",    "z": "1"}),
            ("rotate",    {"x": "1", "y": "0", "z": "0", "angle": "90"}),
            ("translate", {"x": "0", "y": "0.0", "z": "0"}),
        ]),
        ("front", [
            ("scale",     {"x": "0.25",    "y": str(0.25*scale_y),    "z": "1"}),
            ("rotate",    {"x": "1", "y": "0", "z": "0", "angle": "180"}),
            ("translate", {"x": "0", "y": str(0.25*scale_y), "z": "-0.25"}),
        ]),
        ("back", [
            ("scale",     {"x": "0.25",    "y": str(0.25*scale_y),    "z": "1"}),
            ("translate", {"x": "0", "y": str(0.25*scale_y), "z": "0.25"}),
        ]),
        ("right", [
            ("scale",     {"x": "0.25",    "y": str(0.25*scale_y),    "z": "1"}),
            ("rotate",    {"x": "0", "y": "1", "z": "0", "angle": "90"}),
            ("translate", {"x": "0.25", "y": str(0.25*scale_y), "z": "0"}),
        ]),
        ("left", [
            ("scale",     {"x": "0.25",    "y": str(0.25*scale_y),    "z": "1"}),
            ("rotate",    {"x": "0", "y": "1", "z": "0", "angle": "-90"}),
            ("translate", {"x": "-0.25", "y": str(0.25*scale_y), "z": "0"}),
        ])
    ]

    for face_suffix, transforms in faces:
        shape = SubElement(scene, "shape", type="obj",
                           name=f"{name}_{face_suffix}")
        SubElement(shape, "string", name="filename", value=obj_path)
        t = SubElement(shape, "transform", name="toWorld")
        # SubElement(t, "scale", x="1", y=_f(scale_y), z="1")
        for tag, attrs in transforms:
            SubElement(t, tag, **attrs)
        if rot_y != 0.0:
            SubElement(t, "rotate", x="0", y="1", z="0", angle=_f(rot_y))
        SubElement(t, "translate", x=_f(x), y=_f(y), z=_f(z))
        bsdf = SubElement(shape, "bsdf", type="diffuse")
        SubElement(bsdf, "rgb", name="reflectance", value=color_str)


def create_scene(n_tiles=10, tile_scale=0.2, integrator="additionalvertex",
                 resx=512, resy=512, samples=64, obj_path=OBJ_PATH
                 , complete=False):
    scene = Element("scene", version="0.6.0")

    if complete:
        add_integrator(scene, integrator)
        add_geometry_bvh(scene)
        add_sensor(scene, resx=resx, resy=resy, samples=samples)
    add_area_light(scene, obj_path, half_size=0.1, radiance=100)

    grey = (0.725, 0.725, 0.725)
    red = (0.630, 0.065, 0.050)
    green = (0.140, 0.450, 0.091)

    add_wall(scene, "back",    n_tiles, tile_scale, grey,  obj_path)
    add_wall(scene, "floor",   n_tiles, tile_scale, grey,  obj_path)
    add_wall(scene, "ceiling", n_tiles, tile_scale, grey,  obj_path)
    add_wall(scene, "left",    n_tiles, tile_scale, red,   obj_path)
    add_wall(scene, "right",   n_tiles, tile_scale, green, obj_path)
    add_cube(scene, x=0.4, y=-1, z=0.3, scale_y=2, rot_y=50.0,
             color=(0.725, 0.725, 0.725), obj_path=obj_path)
    add_cube(scene, x=-0.35, y=-1, z=-0.1, scale_y=1.5, rot_y=-35.0,
             color=(0.725, 0.725, 0.725), obj_path=obj_path)
    return scene


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Cornell Box Mitsuba 0.6 scene with mini-quad walls.")
    parser.add_argument("--n_tiles", type=int, default=10,
                        help="Number of tiles per dimension per wall (default: 10)")
    parser.add_argument("--tile_scale", type=float, default=0.2,
                        help="Scale factor for tile size (default: 0.2)")
    parser.add_argument("--integrator", default="path",
                        choices=["additionalvertex", "path", "bdpt", "doublestep"],
                        help="Integrator type (default: additionalvertex)")
    parser.add_argument("--resx", type=int, default=512)
    parser.add_argument("--resy", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output", default=None,
                        help="Output XML path (default: cbox_geometry.xml next to this script)")
    parser.add_argument("--obj_path", default=OBJ_PATH,
                        help="Path to quad OBJ (relative to Mitsuba working dir)")
    parser.add_argument("--complete", action="store_true",
                        help="Generate a complete scene with camera and light (default: false)")
    args = parser.parse_args()

    scene = create_scene(
        n_tiles=args.n_tiles,
        tile_scale=args.tile_scale,
        integrator=args.integrator,
        resx=args.resx,
        resy=args.resy,
        samples=args.samples,
        obj_path=args.obj_path,
        complete=args.complete
    )

    xml_bytes = prettify(scene)
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8")

    if args.complete:
        out_path = args.output or os.path.join(os.path.dirname(__file__), "cbox_geometry.xml")
    else:
        out_path = args.output or os.path.join(os.path.dirname(__file__), "geometries/", f"cbox_geometries_{args.n_tiles}_{args.tile_scale}.xml")

    with open(out_path, "wb") as f:
        f.write(xml_bytes)

    n_quads = 5 * args.n_tiles * args.n_tiles
    print(f"Wrote {out_path}")
    print(f"  {args.n_tiles}×{args.n_tiles} tiles per wall × 5 walls = {n_quads} mini-quads")


if __name__ == "__main__":
    main()
