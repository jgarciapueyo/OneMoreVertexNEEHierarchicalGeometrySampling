"""Generate a Mitsuba 1 scene that showcases the basic principled BSDF.

The generated XML contains:
- A perspective camera at (-5, 0, 3) looking at (0, 0, 1)
- A checkerboard ground plane
- A 4-row grid of spheres where each row varies one BSDF parameter
  (roughness, metallic, specular, reflectance)
"""

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def prettify(elem):
    rough = tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="\t", encoding="utf-8")


def vec_to_str(v):
    return f"{v[0]}, {v[1]}, {v[2]}"


def add_sensor(scene, origin, target, up, resx=512, fov=45, samples=256):
    sensor = SubElement(scene, "sensor", type="perspective")
    SubElement(sensor, "string", name="fovAxis", value="smaller")
    SubElement(sensor, "float", name="nearClip", value="0.0001")
    SubElement(sensor, "float", name="farClip", value="1000")

    transform = SubElement(sensor, "transform", name="toWorld")
    SubElement(
        transform,
        "lookAt",
        origin=vec_to_str(origin),
        target=vec_to_str(target),
        up=vec_to_str(up),
    )

    SubElement(sensor, "float", name="fov", value=str(fov))

    sampler = SubElement(sensor, "sampler", type="independent")
    SubElement(sampler, "integer", name="sampleCount", value=str(samples))

    film = SubElement(sensor, "film", type="hdrfilm")
    SubElement(film, "integer", name="width", value=str(resx))
    SubElement(film, "integer", name="height", value=str(resx))


def add_floor(scene, checker_uscale=50, checker_vscale=50):
    plane = SubElement(scene, "shape", type="rectangle", name="floor")
    transform = SubElement(plane, "transform", name="toWorld")
    SubElement(transform, "scale", x="10", y="10", z="1")

    bsdf = SubElement(plane, "bsdf", type="diffuse")
    tex = SubElement(bsdf, "texture", type="checkerboard", name="reflectance")
    SubElement(tex, "float", name="uscale", value=str(checker_uscale))
    SubElement(tex, "float", name="vscale", value=str(checker_vscale))


def add_env_light(scene, radiance=1.0):
    emitter = SubElement(scene, "emitter", type="envmap")
    SubElement(emitter, "string", name="filename", value="env.hdr")


def add_principled_sphere(
    scene,
    name,
    center,
    radius,
    roughness,
    metallic,
    specular,
    reflectance,
):
    sphere = SubElement(scene, "shape", type="sphere", name=name)
    transform = SubElement(sphere, "transform", name="toWorld")
    SubElement(transform, "scale", x=str(radius), y=str(radius), z=str(radius))
    SubElement(transform, "translate", x=str(center[0]), y=str(center[1]), z=str(center[2]))

    bsdf = SubElement(sphere, "bsdf", type="basicprincipled")
    SubElement(bsdf, "rgb", name="reflectance", value=f"{reflectance[0]}, {reflectance[1]}, {reflectance[2]}")
    SubElement(bsdf, "float", name="roughness", value=str(roughness))
    SubElement(bsdf, "float", name="metallic", value=str(metallic))
    SubElement(bsdf, "float", name="specular", value=str(specular))


def add_parameter_grid(scene, columns=10, radius=0.2, x_min=-2.5, x_max=2.5, y_start=-3.0, y_step=2.0):
    step = (x_max - x_min) / (columns - 1)
    x_positions = [x_min + i * step for i in range(columns)]

    # Row roughness
    reflectance = (0.46, 0.48, 0.76)
    metallic = 0.0
    specular = 0.2
    roughness = [i/columns for i in range(0, columns)]

    row_idx = 0
    for col_idx, x in enumerate(x_positions):
        add_principled_sphere(
                scene=scene,
                name=f"sphere_r{row_idx:02d}_c{col_idx:02d}_{roughness[col_idx]:.2f}",
                center=(x, row_idx, 1),
                radius=radius,
                roughness=roughness[col_idx],
                metallic=metallic,
                specular=specular,
                reflectance=reflectance,
            )
        
    # Row roughness
    reflectance = (0.46, 0.48, 0.76)
    metallic = 0.0
    specular = 0.8
    roughness = [i/columns for i in range(0, columns)]

    row_idx = 1
    for col_idx, x in enumerate(x_positions):
        add_principled_sphere(
                scene=scene,
                name=f"sphere_r{row_idx:02d}_c{col_idx:02d}_{roughness[col_idx]:.2f}",
                center=(x, row_idx, 1),
                radius=radius,
                roughness=roughness[col_idx],
                metallic=metallic,
                specular=specular,
                reflectance=reflectance,
            )

    # Row metallic
    reflectance = (0.98, 0.92, 0.44)
    metallic = [i/columns for i in range(0, columns)]
    specular = 0.2
    roughness = 0.01

    row_idx = 2
    for col_idx, x in enumerate(x_positions):
        add_principled_sphere(
                scene=scene,
                name=f"sphere_r{row_idx:02d}_c{col_idx:02d}_{metallic[col_idx]:.2f}",
                center=(x, row_idx, 1),
                radius=radius,
                roughness=roughness,
                metallic=metallic[col_idx],
                specular=specular,
                reflectance=reflectance,
            )
        
    # Row metallic
    reflectance = (0.98, 0.92, 0.44)
    metallic = [i/columns for i in range(0, columns)]
    specular = 0.8
    roughness = 0.01

    row_idx = 3
    for col_idx, x in enumerate(x_positions):
        add_principled_sphere(
                scene=scene,
                name=f"sphere_r{row_idx:02d}_c{col_idx:02d}_{metallic[col_idx]:.2f}",
                center=(x, row_idx, 1),
                radius=radius,
                roughness=roughness,
                metallic=metallic[col_idx],
                specular=specular,
                reflectance=reflectance,
            )
        
    # Row specular
    reflectance = (0.89, 0.21, 0.25)
    metallic = 0.0
    specular = [i/columns for i in range(0, columns)]
    roughness = 0.01

    row_idx = 4
    for col_idx, x in enumerate(x_positions):
        add_principled_sphere(
                scene=scene,
                name=f"sphere_r{row_idx:02d}_c{col_idx:02d}_{specular[col_idx]:.2f}",
                center=(x, row_idx, 1),
                radius=radius,
                roughness=roughness,
                metallic=metallic,
                specular=specular[col_idx],
                reflectance=reflectance,
            )
        
    # Row specular
    reflectance = (0.89, 0.21, 0.25)
    metallic = 0.8
    specular = [i/columns for i in range(0, columns)]
    roughness = 0.01

    row_idx = 5
    for col_idx, x in enumerate(x_positions):
        add_principled_sphere(
                scene=scene,
                name=f"sphere_r{row_idx:02d}_c{col_idx:02d}_{specular[col_idx]:.2f}",
                center=(x, row_idx, 1),
                radius=radius,
                roughness=roughness,
                metallic=metallic,
                specular=specular[col_idx],
                reflectance=reflectance,
            )


def add_integrator(scene, samples=256):
    integrator = SubElement(scene, "integrator", type="path")
    SubElement(integrator, "integer", name="maxDepth", value="8")
    SubElement(integrator, "integer", name="rrDepth", value="5")
    SubElement(integrator, "integer", name="sampleCount", value=str(samples))


def build_scene_xml(resx=256, fov=45, samples=256):
    scene = Element("scene", version="0.6.0")
    add_integrator(scene, samples=samples)
    add_sensor(
        scene,
        origin=(0, 2.5, 8),
        target=(0, 2.5, 0),
        up=(-1, 0, 0),
        resx=resx,
        fov=fov,
        samples=samples,
    )
    add_env_light(scene, radiance=1.0)
    add_floor(scene, checker_uscale=50, checker_vscale=50)
    add_parameter_grid(scene)

    return prettify(scene)


def write_scene(path="scene.xml", **kwargs):
    xml_bytes = build_scene_xml(**kwargs)
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8")

    with open(path, "wb") as file_handle:
        file_handle.write(xml_bytes)


if __name__ == "__main__":
    write_scene(path="scene.xml", resx=512, fov=45, samples=512)
