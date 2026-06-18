#!/usr/bin/env python3
"""
Generate a hemisphere mesh (OBJ) made of independent triangles (duplicated vertices),
with deterministic uniform thinning to approximate a target solid-angle ratio over
the hemisphere (2π sr).

Example:
    python create_hemisphere.py --radius 10.0 --ratio 0.4 --frequency 20 \
        --output-dir scenes/ --reflectance 0.8
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Icosphere generation
# ---------------------------------------------------------------------------

def normalize(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n == 0.0:
        raise ValueError("Zero-length vector.")
    return (v[0] / n, v[1] / n, v[2] / n)


def _icosahedron() -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [normalize(v) for v in raw]
    faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    return verts, faces


def subdivide(verts: List[Vec3], faces: List[Tuple[int,int,int]], freq: int) -> List[Tuple[Vec3, Vec3, Vec3]]:
    """Return list of (v0,v1,v2) triangles on the unit sphere after subdivision."""
    triangles: List[Tuple[Vec3, Vec3, Vec3]] = []
    for ia, ib, ic in faces:
        a, b, c = verts[ia], verts[ib], verts[ic]
        for i in range(freq):
            for j in range(freq - i):
                k = freq - i - j
                # four sub-triangle vertices via barycentric coords
                def pt(wi, wj, wk):
                    fi, fj, fk = wi / freq, wj / freq, wk / freq
                    x = fi * a[0] + fj * b[0] + fk * c[0]
                    y = fi * a[1] + fj * b[1] + fk * c[1]
                    z = fi * a[2] + fj * b[2] + fk * c[2]
                    return normalize((x, y, z))

                p00 = pt(i,   j,   k)
                p10 = pt(i+1, j,   k-1)
                p01 = pt(i,   j+1, k-1)
                triangles.append((p00, p10, p01))
                if j + 1 <= freq - i - 1:
                    p11 = pt(i+1, j+1, k-2)
                    triangles.append((p10, p11, p01))
    return triangles


# ---------------------------------------------------------------------------
# Hemisphere filtering and thinning
# ---------------------------------------------------------------------------

def centroid(tri: Tuple[Vec3, Vec3, Vec3]) -> Vec3:
    a, b, c = tri
    return (
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    )


def theta_of(v: Vec3) -> float:
    """Polar angle from +Z axis, in [0, pi/2] for upper hemisphere."""
    z = max(-1.0, min(1.0, v[2]))
    return math.acos(z)


def thin_uniform(triangles: List[Tuple[Vec3,Vec3,Vec3]], ratio: float) -> List[Tuple[Vec3,Vec3,Vec3]]:
    """
    Deterministically keep ~ratio fraction of triangles, uniformly distributed
    across theta rings.  We bin faces by their theta ring index and apply an
    evenly-spaced stride within each bin, so that the kept faces are spread
    uniformly in azimuth for every elevation.
    """
    if ratio <= 0.0:
        return []
    if ratio >= 1.0:
        return list(triangles)

    # Determine number of theta rings from icosphere structure.
    # We use N_BINS proportional to sqrt(total), capped generously.
    n_bins = max(8, int(round(math.sqrt(len(triangles)))))
    theta_max = math.pi / 2.0

    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for idx, tri in enumerate(triangles):
        ctr = centroid(tri)
        th = theta_of(ctr)
        bin_idx = min(n_bins - 1, int(th / theta_max * n_bins))
        bins[bin_idx].append(idx)

    kept_indices = []
    for bin_faces in bins:
        n = len(bin_faces)
        if n == 0:
            continue
        # How many to keep in this bin
        n_keep = max(0, round(ratio * n))
        if n_keep == 0:
            continue
        if n_keep >= n:
            kept_indices.extend(bin_faces)
            continue
        # Evenly-spaced stride: pick n_keep faces out of n with uniform spacing
        stride = n / n_keep
        for k in range(n_keep):
            pick = int(k * stride + stride / 2.0)  # centred within stride
            kept_indices.append(bin_faces[pick])

    return [triangles[i] for i in sorted(kept_indices)]


# ---------------------------------------------------------------------------
# Solid angle computation
# ---------------------------------------------------------------------------

def triangle_solid_angle(a: Vec3, b: Vec3, c: Vec3) -> float:
    """Solid angle subtended by a spherical triangle (van Oosterom & Strackee)."""
    def cross(u, v):
        return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    def dot(u, v):
        return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]
    def norm(u):
        return math.sqrt(dot(u, u))

    na, nb, nc = normalize(a), normalize(b), normalize(c)
    numerator = abs(dot(na, cross(nb, nc)))
    denom = 1.0 + dot(na, nb) + dot(nb, nc) + dot(na, nc)
    if abs(denom) < 1e-15:
        return 0.0
    return 2.0 * math.atan2(numerator, denom)


def total_solid_angle(triangles: List[Tuple[Vec3,Vec3,Vec3]]) -> float:
    return sum(triangle_solid_angle(a, b, c) for a, b, c in triangles)


# ---------------------------------------------------------------------------
# OBJ writing (duplicated vertices)
# ---------------------------------------------------------------------------

def write_obj(triangles: List[Tuple[Vec3,Vec3,Vec3]], radius: float, path: Path) -> None:
    lines = [
        "# Hemisphere mesh — independent triangles (duplicated vertices)",
        f"# radius={radius}, n_triangles={len(triangles)}",
        "",
    ]
    for idx, (a, b, c) in enumerate(triangles):
        for vx, vy, vz in (a, b, c):
            lines.append(f"v {vx*radius:.10f} {vy*radius:.10f} {vz*radius:.10f}")
        base = idx * 3 + 1
        lines.append(f"f {base} {base+2} {base+1}")  # reversed winding for inward normals
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# XML writing
# ---------------------------------------------------------------------------

def format_float(v: float) -> str:
    return f"{v:.10g}"


def write_xml(
    obj_path: Path,
    radius: float,
    reflectance: float,
    output_path: Path,
) -> None:
    cam_origin = (0.0, -radius * 1.5, radius / 2.0)

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<scene version="0.6.0">',
        '',
        # '    <!-- Integrator -->',
        # '    <integrator type="path">',
        # '        <integer name="max_depth" value="3"/>',
        # '    </integrator>',
        # '    <!-- Camera -->',
        # '    <sensor type="perspective">',
        # '        <string name="fov_axis" value="x"/>',
        # '        <float name="fov" value="60"/>',
        # '        <transform name="toWorld">',
        # f'            <lookAt origin="{format_float(cam_origin[0])}, {format_float(cam_origin[1])}, {format_float(cam_origin[2])}"'
        # f' target="0, 0, 0" up="0, 0, 1"/>',
        # '        </transform>',
        # '        <sampler type="independent">',
        # '            <integer name="sample_count" value="64"/>',
        # '        </sampler>',
        # '        <film type="hdrfilm">',
        # '            <integer name="width" value="800"/>',
        # '            <integer name="height" value="800"/>',
        # '        </film>',
        # '    </sensor>',
        # '',
        '    <!-- Hemisphere mesh -->',
        '    <shape type="obj">',
        f'        <string name="filename" value="{obj_path.as_posix()}"/>',
        '        <bsdf type="diffuse">',
        f'            <rgb name="reflectance" value="{format_float(reflectance)}"/>',
        '        </bsdf>',
        '    </shape>',
        '',
        '    <!-- Spotlight at center, slightly above floor, pointing up, near-180° cone -->',
        '    <emitter type="spot">',
        '        <transform name="toWorld">',
        f'            <lookAt origin="0, 0, {format_float(radius * 0.01)}" target="0, 0, 1" up="0, 1, 0"/>',
        '        </transform>',
        '        <float name="cutoffAngle" value="89.0"/>',
        '        <float name="beamWidth" value="85.0"/>',
        '        <rgb name="intensity" value="1.0"/>',
        '    </emitter>',
        '',
        '    <!-- Floor plane -->',
        '    <shape type="rectangle">',
        '        <transform name="toWorld">',
        f'            <scale x="{format_float(radius * 2.0)}" y="{format_float(radius * 2.0)}" z="1"/>',
        '            <rotate x="1" angle="0"/>',
        '        </transform>',
        '        <bsdf type="diffuse">',
        f'            <rgb name="reflectance" value="{format_float(reflectance)}"/>',
        '        </bsdf>',
        '    </shape>',
        '',
        '</scene>',
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a hemisphere OBJ (independent triangles) + Mitsuba XML."
    )
    p.add_argument("-R", "--radius",      type=float, default=10.0,  help="Hemisphere radius.")
    p.add_argument("-rt","--ratio",       type=float, default=0.5,   help="Target solid-angle ratio over hemisphere (2π sr).")
    p.add_argument("-q", "--frequency",   type=int,   default=20,    help="Icosphere subdivision frequency (controls triangle count).")
    p.add_argument("-o", "--output-dir",  type=str,   default=".",   help="Directory for output files.")
    p.add_argument("-p", "--name-prefix", type=str,   default="hemisphere", help="Filename prefix.")
    p.add_argument("-r", "--reflectance", type=float, default=0.8,   help="Diffuse reflectance.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not (0.0 < args.ratio <= 1.0):
        print("ERROR: --ratio must be in (0, 1]", file=sys.stderr)
        return 1
    if args.frequency < 1:
        print("ERROR: --frequency must be >= 1", file=sys.stderr)
        return 1

    # 1. Build full icosphere triangles
    verts, faces = _icosahedron()
    all_tris = subdivide(verts, faces, args.frequency)

    # 2. Keep only upper-hemisphere faces (all three vertices z >= 0)
    hemi_tris = [
        t for t in all_tris
        if t[0][2] >= 0.0 and t[1][2] >= 0.0 and t[2][2] >= 0.0
    ]

    total_hemi = len(hemi_tris)
    print(f"Icosphere freq={args.frequency}: {len(all_tris)} total faces, "
          f"{total_hemi} upper-hemisphere faces")

    # 3. Thin deterministically to approximate the target ratio
    kept = thin_uniform(hemi_tris, args.ratio)
    actual_n = len(kept)

    # 4. Compute actual solid angle ratio
    sa_kept   = total_solid_angle(kept)
    sa_hemi   = 2.0 * math.pi          # ideal hemisphere
    actual_ratio = sa_kept / sa_hemi

    print(f"Target ratio : {args.ratio:.4f}  ({args.ratio * 2 * math.pi:.4f} sr)")
    print(f"Kept faces   : {actual_n} / {total_hemi}")
    print(f"Actual ratio : {actual_ratio:.4f}  ({sa_kept:.4f} sr)")

    # 5. Build output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_name = f"{args.name_prefix}_n{actual_n}_ratio{actual_ratio:.4f}.obj"
    xml_name = f"{args.name_prefix}_n{actual_n}_ratio{actual_ratio:.4f}.xml"
    obj_path = output_dir / obj_name
    xml_path = output_dir / xml_name

    # 6. Write OBJ
    write_obj(kept, args.radius, obj_path)
    print(f"OBJ written  : {obj_path}")

    # 7. Write XML (obj_path relative to xml_path's directory for portability)
    write_xml(
        obj_path=obj_path,
        radius=args.radius,
        reflectance=args.reflectance,
        output_path=xml_path,
    )
    print(f"XML written  : {xml_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
