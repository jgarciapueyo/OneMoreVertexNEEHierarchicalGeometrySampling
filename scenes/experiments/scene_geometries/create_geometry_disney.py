#!/usr/bin/env python3

"""
Like create_geometry.py but with the option to use a Disney (basicprincipled) BSDF
instead of a plain diffuse BSDF.

Three placement/material modes are available via --mode:

  homogeneous          (default) All quads share the same BSDF, controlled by
                       --bsdf, -r/--reflectance, --roughness, --specular, --metallic.
  grouped              Quads are placed in N spatial clusters (--num-clusters).
                       Each cluster is assigned a distinct material from MULTI_MATERIAL_SET;
                       varying reflectances make some clusters contribute more than others,
                       so the BVH structure aligns with material importance.
  random_heterogeneous Quads are placed randomly (same as homogeneous) but each quad is
                       independently assigned a random material from MULTI_MATERIAL_SET,
                       breaking any BVH-material spatial correlation.

Examples:

  # Diffuse quads (same as create_geometry.py):
  python scenes/experiments/scene_geometries/create_geometry_disney.py \\
    -n 3000 -f scenes/experiments/scene_geometries/quad_2tri.obj \\
    -d -1 0 0 -k 40 -a 0 -10 1 -b 10 10 3 \\
    --bsdf diffuse -r 0.8 \\
    -o scenes/experiments/scene_geometries/quads-disney-diffuse-k40.xml -s 42

  # Semi-specular principled quads:
  python scenes/experiments/scene_geometries/create_geometry_disney.py \\
    -n 3000 -f scenes/experiments/scene_geometries/quad_2tri.obj \\
    -d -1 0 0 -k 40 -a 0 -10 1 -b 10 10 3 \\
    --bsdf principled -r 0.8 --roughness 0.3 --specular 0.5 \\
    -o scenes/experiments/scene_geometries/quads-disney-rough0.3-k40.xml -s 42

  # Metallic principled quads:
  python scenes/experiments/scene_geometries/create_geometry_disney.py \\
    -n 3000 -f scenes/experiments/scene_geometries/quad_2tri.obj \\
    -d -1 0 0 -k 40 -a 0 -10 1 -b 10 10 3 \\
    --bsdf principled -r 0.8 --roughness 0.2 --specular 0.0 --metallic 1.0 \\
    -o scenes/experiments/scene_geometries/quads-disney-metal1.0-rough0.2-k40.xml -s 42

  # Spatially grouped quads with 6 clusters (each cluster a distinct material):
  python scenes/experiments/scene_geometries/create_geometry_disney.py \\
    -n 3000 -f scenes/experiments/scene_geometries/quad_2tri.obj \\
    -d -1 0 0 -k 40 -a 0 -10 1 -b 10 10 3 \\
    --mode grouped --num-clusters 6 --cluster-kappa 100 \\
    -o scenes/experiments/scene_geometries/quads-disney-grouped-6clusters-k40.xml -s 42

  # Randomly placed quads with per-quad heterogeneous materials:
  python scenes/experiments/scene_geometries/create_geometry_disney.py \\
    -n 3000 -f scenes/experiments/scene_geometries/quad_2tri.obj \\
    -d -1 0 0 -k 40 -a 0 -10 1 -b 10 10 3 \\
    --mode random_heterogeneous \\
    -o scenes/experiments/scene_geometries/quads-disney-heterogeneous-k40.xml -s 42
"""


import argparse
import math
import random
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]
# (reflectance, roughness, specular, metallic)
MaterialSpec = Tuple[float, float, float, float]
# (position, lookAt direction, material)
ObjectSpec = Tuple[Vec3, Vec3, MaterialSpec]

# Predefined material set used in grouped and random_heterogeneous modes.
# Spans bright/dark diffuse, rough/smooth specular, and rough/smooth metallic.
# Index in this list determines cluster assignment in grouped mode.
MULTI_MATERIAL_SET: List[MaterialSpec] = [
    (0.9, 0.2,  0.0, 0.0),   # bright diffuse
    (0.1, 0.2,  0.0, 0.0),   # dark diffuse
    (0.8, 0.2,  1.0, 0.0),   # bright specular rough
    (0.8, 0.05, 1.0, 0.0),   # bright specular smooth
    (0.8, 0.2,  0.0, 1.0),   # metallic rough
    (0.8, 0.05, 0.0, 1.0),   # metallic smooth
]


SHORT_FLAG_MAP = {
    "--n-objects": "-n",
    "--obj-filename": "-f",
    "--mean-dir": "-d",
    "--kappa": "-k",
    "--min-pos": "-a",
    "--max-pos": "-b",
    "--output": "-o",
    "--seed": "-s",
    "--name-prefix": "-p",
    "--scale": "-c",
    "--up": "-u",
    "--reflectance": "-r",
    "--instances": "-i",
}


def normalize(v: Vec3) -> Vec3:
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm == 0.0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def orthonormal_basis(axis: Vec3) -> Tuple[Vec3, Vec3]:
    helper = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.999 else (1.0, 0.0, 0.0)
    tangent = normalize(cross(helper, axis))
    bitangent = cross(axis, tangent)
    return tangent, bitangent


def sample_uniform_sphere(rng: random.Random) -> Vec3:
    z = 2.0 * rng.random() - 1.0
    phi = 2.0 * math.pi * rng.random()
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(phi), r * math.sin(phi), z)


def sample_vmf(mean_dir: Vec3, kappa: float, rng: random.Random) -> Vec3:
    if kappa <= 1e-6:
        return sample_uniform_sphere(rng)

    u = rng.random()
    exp_term = math.exp(-2.0 * kappa)
    w = 1.0 + math.log(u + (1.0 - u) * exp_term) / kappa
    w = max(-1.0, min(1.0, w))

    phi = 2.0 * math.pi * rng.random()
    sin_theta = math.sqrt(max(0.0, 1.0 - w * w))
    tangent, bitangent = orthonormal_basis(mean_dir)

    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)

    direction = (
        x * tangent[0] + y * bitangent[0] + w * mean_dir[0],
        x * tangent[1] + y * bitangent[1] + w * mean_dir[1],
        x * tangent[2] + y * bitangent[2] + w * mean_dir[2],
    )
    return normalize(direction)


def format_float(value: float) -> str:
    return f"{value:.16g}"


def format_vec3(value: Vec3) -> str:
    return ", ".join(format_float(component) for component in value)


def sample_position(min_pos: Vec3, max_pos: Vec3, rng: random.Random) -> Vec3:
    return (
        rng.uniform(min_pos[0], max_pos[0]),
        rng.uniform(min_pos[1], max_pos[1]),
        rng.uniform(min_pos[2], max_pos[2]),
    )


def validate_bounds(min_pos: Vec3, max_pos: Vec3) -> None:
    if min_pos[0] > max_pos[0] or min_pos[1] > max_pos[1] or min_pos[2] > max_pos[2]:
        raise ValueError("Each component of min_pos must be <= max_pos.")


def compute_cluster_bounds(
    num_clusters: int, min_pos: Vec3, max_pos: Vec3
) -> List[Tuple[Vec3, Vec3]]:
    """Return per-cluster (min_pos, max_pos) bounding boxes laid out in a grid.

    Clusters are arranged in a grid whose aspect ratio matches the x/y extent of
    the scene.  Each cluster occupies 80% of its grid cell (leaving 20% as gap),
    so clusters are visually separated in the scene.
    """
    x_range = max_pos[0] - min_pos[0]
    y_range = max_pos[1] - min_pos[1]

    ratio = (x_range / y_range) if y_range > 1e-9 else 1.0
    n_x = max(1, round(math.sqrt(num_clusters * ratio)))
    n_y = max(1, math.ceil(num_clusters / n_x))
    n_x = max(1, math.ceil(num_clusters / n_y))

    cell_w = x_range / n_x
    cell_h = y_range / n_y

    cluster_bounds: List[Tuple[Vec3, Vec3]] = []
    for k in range(num_clusters):
        col = k % n_x
        row = k // n_x
        cx = min_pos[0] + (col + 0.5) * cell_w
        cy = min_pos[1] + (row + 0.5) * cell_h
        hw = cell_w * 0.4
        hh = cell_h * 0.4
        c_min: Vec3 = (cx - hw, cy - hh, min_pos[2])
        c_max: Vec3 = (cx + hw, cy + hh, max_pos[2])
        cluster_bounds.append((c_min, c_max))

    return cluster_bounds


def generate_objects_homogeneous(
    n_objects: int,
    mean_dir: Vec3,
    kappa: float,
    min_pos: Vec3,
    max_pos: Vec3,
    reflectance: float,
    roughness: float,
    specular: float,
    metallic: float,
    rng: random.Random,
) -> List[ObjectSpec]:
    material: MaterialSpec = (reflectance, roughness, specular, metallic)
    objects: List[ObjectSpec] = []
    for _ in range(n_objects):
        pos = sample_position(min_pos, max_pos, rng)
        direction = sample_vmf(mean_dir, kappa, rng)
        objects.append((pos, direction, material))
    return objects


def generate_objects_grouped(
    n_objects: int,
    mean_dir: Vec3,
    kappa_within_cluster: float,
    min_pos: Vec3,
    max_pos: Vec3,
    num_clusters: int,
    rng: random.Random,
) -> List[ObjectSpec]:
    """Place quads in spatial clusters; each cluster gets a distinct material from
    MULTI_MATERIAL_SET.  Quads within a cluster share a position sub-box and have
    more tightly concentrated orientations (kappa_within_cluster)."""
    cluster_bounds = compute_cluster_bounds(num_clusters, min_pos, max_pos)

    base = n_objects // num_clusters
    remainder = n_objects % num_clusters
    cluster_counts = [base + (1 if i < remainder else 0) for i in range(num_clusters)]

    objects: List[ObjectSpec] = []
    for c_idx, count in enumerate(cluster_counts):
        material = MULTI_MATERIAL_SET[c_idx % len(MULTI_MATERIAL_SET)]
        c_min, c_max = cluster_bounds[c_idx]
        for _ in range(count):
            pos = sample_position(c_min, c_max, rng)
            direction = sample_vmf(mean_dir, kappa_within_cluster, rng)
            objects.append((pos, direction, material))

    return objects


def generate_objects_heterogeneous(
    n_objects: int,
    mean_dir: Vec3,
    kappa: float,
    min_pos: Vec3,
    max_pos: Vec3,
    rng: random.Random,
) -> List[ObjectSpec]:
    """Place quads randomly (same as homogeneous) but assign each quad an
    independently sampled material from MULTI_MATERIAL_SET, so there is no
    spatial correlation between position and material."""
    objects: List[ObjectSpec] = []
    for _ in range(n_objects):
        pos = sample_position(min_pos, max_pos, rng)
        direction = sample_vmf(mean_dir, kappa, rng)
        material = rng.choice(MULTI_MATERIAL_SET)
        objects.append((pos, direction, material))
    return objects


def get_short_invocation_command() -> str:
    converted_args: List[str] = []
    for token in sys.argv[1:]:
        if token in SHORT_FLAG_MAP:
            converted_args.append(SHORT_FLAG_MAP[token])
            continue

        rewritten = False
        for long_flag, short_flag in SHORT_FLAG_MAP.items():
            prefix = long_flag + "="
            if token.startswith(prefix):
                converted_args.append(short_flag + token[len(long_flag):])
                rewritten = True
                break

        if not rewritten:
            converted_args.append(token)

    return "python " + " ".join(shlex.quote(arg) for arg in [sys.argv[0], *converted_args])


def build_generation_comment(command: str) -> str:
    escaped_command = command.replace("--", "- -")
    return (
        "<!-- This file was generated by the create_geometry_disney script, using command: -->\n"
        f"<!-- \t{escaped_command} -->"
    )


def build_bsdf_lines(
    bsdf_type: str,
    reflectance: float,
    roughness: float,
    specular: float,
    metallic: float,
    indent: str,
) -> List[str]:
    """Return XML lines for the chosen BSDF (no trailing newline)."""
    if bsdf_type == "diffuse":
        return [
            f'{indent}<bsdf type="diffuse">',
            f'{indent}    <rgb name="reflectance" value="{format_float(reflectance)}"/>',
            f'{indent}</bsdf>',
        ]
    else:  # principled
        return [
            f'{indent}<bsdf type="basicprincipled">',
            f'{indent}    <rgb name="reflectance" value="{format_float(reflectance)}"/>',
            f'{indent}    <float name="roughness" value="{format_float(roughness)}"/>',
            f'{indent}    <float name="specular" value="{format_float(specular)}"/>',
            f'{indent}    <float name="metallic" value="{format_float(metallic)}"/>',
            f'{indent}</bsdf>',
        ]


def build_xml(
    objects: List[ObjectSpec],
    obj_filename: str,
    name_prefix: str,
    scale: Vec3,
    up: Vec3,
    bsdf_type: str,
    generation_comment: str,
    instances: bool = False,
) -> str:
    lines: List[str] = ['<?xml version="1.0" encoding="utf-8"?>', generation_comment, '<scene version="0.6.0">']

    if instances:
        # instances mode shares one BSDF; use the first object's material
        first_mat = objects[0][2] if objects else (0.8, 0.5, 0.5, 0.0)
        reflectance, roughness, specular, metallic = first_mat
        group_id = f"{name_prefix}_group"
        bsdf_lines = build_bsdf_lines(bsdf_type, reflectance, roughness, specular, metallic, indent="        ")
        lines.extend(
            [
                f'    <shape type="shapegroup" id="{group_id}">',
                '        <shape type="obj">',
                f'            <string name="filename" value="{obj_filename}"/>',
            ]
            + bsdf_lines
            + [
                '        </shape>',
                '    </shape>',
            ]
        )

    for idx, (origin, direction, material) in enumerate(objects):
        target = (
            origin[0] + direction[0],
            origin[1] + direction[1],
            origin[2] + direction[2],
        )
        reflectance, roughness, specular, metallic = material

        if instances:
            group_id = f"{name_prefix}_group"
            lines.extend(
                [
                    f'    <shape type="instance" name="{name_prefix}_{idx:05d}">',
                    f'        <ref id="{group_id}"/>',
                    '        <transform name="toWorld">',
                    f'            <scale x="{format_float(scale[0])}" y="{format_float(scale[1])}" z="{format_float(scale[2])}"/>',
                    f'            <lookAt origin="{format_vec3(origin)}" target="{format_vec3(target)}" up="{format_vec3(up)}"/>',
                    '        </transform>',
                    '    </shape>',
                ]
            )
        else:
            bsdf_lines = build_bsdf_lines(bsdf_type, reflectance, roughness, specular, metallic, indent="        ")
            lines.extend(
                [
                    f'    <shape type="obj" name="{name_prefix}_{idx:05d}">',
                    f'        <string name="filename" value="{obj_filename}"/>',
                    '        <transform name="toWorld">',
                    f'            <scale x="{format_float(scale[0])}" y="{format_float(scale[1])}" z="{format_float(scale[2])}"/>',
                    f'            <lookAt origin="{format_vec3(origin)}" target="{format_vec3(target)}" up="{format_vec3(up)}"/>',
                    '        </transform>',
                ]
                + bsdf_lines
                + [
                    '    </shape>',
                ]
            )

    lines.append('</scene>')
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Mitsuba XML with randomly placed/oriented obj instances and a Disney (basicprincipled) or diffuse BSDF."
    )
    parser.add_argument("-n", "--n-objects", type=int, required=True, help="Number of objects to instantiate.")
    parser.add_argument("-f", "--obj-filename", type=str, required=True, help="Path to the OBJ mesh.")
    parser.add_argument(
        "-d",
        "--mean-dir",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="Mean orientation direction.",
    )
    parser.add_argument(
        "-k",
        "--kappa",
        type=float,
        required=True,
        help="vMF concentration parameter (0 = uniform sphere, larger = more concentrated).",
    )
    parser.add_argument(
        "-a",
        "--min-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="AABB minimum world position.",
    )
    parser.add_argument(
        "-b",
        "--max-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="AABB maximum world position.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="-",
        help="Output XML file path. Use '-' for stdout.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible output.",
    )
    parser.add_argument(
        "-p",
        "--name-prefix",
        type=str,
        default="quad",
        help="Name prefix used for shape IDs.",
    )
    parser.add_argument(
        "-c",
        "--scale",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(0.05, 0.05, 0.0005),
        help="Scale applied in the toWorld transform.",
    )
    parser.add_argument(
        "-u",
        "--up",
        type=float,
        nargs=3,
        metavar=("UX", "UY", "UZ"),
        default=(0.0, 0.0, 1.0),
        help="Up vector used in lookAt.",
    )
    parser.add_argument(
        "-r",
        "--reflectance",
        type=float,
        default=0.8,
        help="Base diffuse reflectance for principled or diffuse BSDF (homogeneous mode only).",
    )
    parser.add_argument(
        "--bsdf",
        type=str,
        choices=["diffuse", "principled"],
        default="diffuse",
        help="BSDF type: 'diffuse' (Lambertian) or 'principled' (Disney/basicprincipled). Homogeneous mode only.",
    )
    parser.add_argument(
        "--roughness",
        type=float,
        default=0.5,
        help="GGX roughness for principled BSDF (ignored for diffuse). Range [0, 1]. Homogeneous mode only.",
    )
    parser.add_argument(
        "--specular",
        type=float,
        default=0.5,
        help="Dielectric specular weight for principled BSDF (ignored for diffuse). Range [0, 1]. Homogeneous mode only.",
    )
    parser.add_argument(
        "--metallic",
        type=float,
        default=0.0,
        help="Metallic weight for principled BSDF (ignored for diffuse). Range [0, 1]. Homogeneous mode only.",
    )
    parser.add_argument(
        "-i",
        "--instances",
        action="store_true",
        help="Use <instance> instead of <shape> for instancing the same mesh. Homogeneous mode only.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["homogeneous", "grouped", "random_heterogeneous"],
        default="homogeneous",
        help=(
            "Placement/material mode. "
            "'homogeneous': all quads share one BSDF (default, existing behavior). "
            "'grouped': N spatial clusters, each cluster a distinct material from the predefined set. "
            "'random_heterogeneous': random placement, per-quad random material from the predefined set."
        ),
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=6,
        help="Number of spatial clusters (grouped mode only).",
    )
    parser.add_argument(
        "--cluster-kappa",
        type=float,
        default=100.0,
        help="vMF concentration for quad orientations within each cluster (grouped mode only).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.n_objects < 0:
        parser.error("--n-objects must be >= 0")
    if args.kappa < 0.0:
        parser.error("--kappa must be >= 0")
    if not (0.0 <= args.roughness <= 1.0):
        parser.error("--roughness must be in [0, 1]")
    if not (0.0 <= args.specular <= 1.0):
        parser.error("--specular must be in [0, 1]")
    if not (0.0 <= args.metallic <= 1.0):
        parser.error("--metallic must be in [0, 1]")
    if args.instances and args.mode != "homogeneous":
        parser.error("--instances is only valid in homogeneous mode")
    if args.num_clusters < 1:
        parser.error("--num-clusters must be >= 1")

    mean_dir = normalize((args.mean_dir[0], args.mean_dir[1], args.mean_dir[2]))
    min_pos = (args.min_pos[0], args.min_pos[1], args.min_pos[2])
    max_pos = (args.max_pos[0], args.max_pos[1], args.max_pos[2])
    validate_bounds(min_pos, max_pos)

    scale = (args.scale[0], args.scale[1], args.scale[2])
    up = normalize((args.up[0], args.up[1], args.up[2]))

    rng = random.Random(args.seed)
    command = get_short_invocation_command()
    generation_comment = build_generation_comment(command)

    if args.mode == "homogeneous":
        objects = generate_objects_homogeneous(
            n_objects=args.n_objects,
            mean_dir=mean_dir,
            kappa=args.kappa,
            min_pos=min_pos,
            max_pos=max_pos,
            reflectance=args.reflectance,
            roughness=args.roughness,
            specular=args.specular,
            metallic=args.metallic,
            rng=rng,
        )
        bsdf_type = args.bsdf
    elif args.mode == "grouped":
        objects = generate_objects_grouped(
            n_objects=args.n_objects,
            mean_dir=mean_dir,
            kappa_within_cluster=args.cluster_kappa,
            min_pos=min_pos,
            max_pos=max_pos,
            num_clusters=args.num_clusters,
            rng=rng,
        )
        bsdf_type = "principled"
    else:  # random_heterogeneous
        objects = generate_objects_heterogeneous(
            n_objects=args.n_objects,
            mean_dir=mean_dir,
            kappa=args.kappa,
            min_pos=min_pos,
            max_pos=max_pos,
            rng=rng,
        )
        bsdf_type = "principled"

    xml = build_xml(
        objects=objects,
        obj_filename=args.obj_filename,
        name_prefix=args.name_prefix,
        scale=scale,
        up=up,
        bsdf_type=bsdf_type,
        generation_comment=generation_comment,
        instances=args.instances,
    )

    if args.output == "-":
        sys.stdout.write(xml)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml, encoding="utf-8")
        print(f"Generated geometry XML written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
