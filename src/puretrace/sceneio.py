"""Declarative JSON scene loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .camera import Camera
from .geometry import Quad, Sphere, Triangle, lathe, make_box
from .materials import PrincipledMaterial
from .math3d import Vec2, Vec3
from .objloader import Transform, load_obj
from .scene import HomogeneousMedium, Scene
from .textures import CheckerTexture, EnvironmentMap, ImageTexture, SolidColor, Texture, load_image


@dataclass(frozen=True, slots=True)
class LoadedScene:
    scene: Scene
    camera: Camera
    render: dict[str, Any]


def _vec3(value: Any, default: Vec3 | None = None) -> Vec3:
    if value is None and default is not None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a three-element vector, got {value!r}")
    return Vec3(float(value[0]), float(value[1]), float(value[2]))


def _vec2(value: Any, default: Vec2 | None = None) -> Vec2:
    if value is None and default is not None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected a two-element vector, got {value!r}")
    return Vec2(float(value[0]), float(value[1]))


def _texture(spec: Any, root: Path) -> Texture:
    if isinstance(spec, (list, tuple)):
        return SolidColor(_vec3(spec))
    if isinstance(spec, str):
        return ImageTexture(load_image(root / spec))
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid texture specification {spec!r}")
    kind = spec.get("type", "solid")
    if kind == "solid":
        return SolidColor(_vec3(spec.get("color", [1, 1, 1])))
    if kind == "checker":
        return CheckerTexture(
            _texture(spec.get("even", [0.8, 0.8, 0.8]), root),
            _texture(spec.get("odd", [0.2, 0.2, 0.2]), root),
            float(spec.get("scale", 4.0)),
            bool(spec.get("use_uv", False)),
        )
    if kind == "image":
        return ImageTexture(
            load_image(root / spec["file"], srgb=bool(spec.get("srgb", True))),
            float(spec.get("scale_u", 1.0)),
            float(spec.get("scale_v", 1.0)),
            float(spec.get("offset_u", 0.0)),
            float(spec.get("offset_v", 0.0)),
            bool(spec.get("flip_v", True)),
            bool(spec.get("wrap_u", True)),
        )
    raise ValueError(f"Unknown texture type {kind!r}")


def _material(name: str, spec: dict[str, Any], root: Path) -> PrincipledMaterial:
    return PrincipledMaterial(
        base_color=_texture(spec.get("base_color", [0.8, 0.8, 0.8]), root),
        metallic=float(spec.get("metallic", 0.0)),
        roughness=float(spec.get("roughness", 0.5)),
        transmission=float(spec.get("transmission", 0.0)),
        ior=float(spec.get("ior", 1.5)),
        emission=_texture(spec.get("emission", [0, 0, 0]), root),
        emission_strength=float(spec.get("emission_strength", 0.0)),
        opacity=float(spec.get("opacity", 1.0)),
        two_sided=bool(spec.get("two_sided", False)),
        name=name,
    )


def load_scene(path: str | Path) -> LoadedScene:
    path = Path(path)
    root = path.parent
    data = json.loads(path.read_text(encoding="utf-8"))
    material_specs = data.get("materials", {})
    materials = {name: _material(name, spec, root) for name, spec in material_specs.items()}

    def get_material(name: str) -> PrincipledMaterial:
        if name not in materials:
            raise ValueError(f"Unknown material {name!r}")
        return materials[name]

    environment_spec = data.get("environment", {})
    if "hdr" in environment_spec:
        environment = EnvironmentMap(
            image=load_image(root / environment_spec["hdr"], srgb=False),
            strength=float(environment_spec.get("strength", 1.0)),
            rotation=float(environment_spec.get("rotation", 0.0)) * 3.141592653589793 / 180.0,
        )
    else:
        environment = EnvironmentMap(
            color=_vec3(environment_spec.get("color", [0, 0, 0])),
            strength=float(environment_spec.get("strength", 1.0)),
        )
    medium_spec = data.get("medium")
    medium = None
    if medium_spec:
        medium = HomogeneousMedium(
            float(medium_spec.get("density", 0.0)),
            _vec3(medium_spec.get("albedo", [0.9, 0.9, 0.9])),
            float(medium_spec.get("anisotropy", 0.0)),
            float(medium_spec.get("max_distance", 100.0)),
        )
    scene = Scene(environment=environment, medium=medium)

    for index, spec in enumerate(data.get("objects", [])):
        kind = spec.get("type")
        try:
            if kind in ("sphere", "moving_sphere"):
                scene.add(
                    Sphere(
                        _vec3(spec["center"]),
                        float(spec["radius"]),
                        get_material(spec["material"]),
                        _vec3(spec.get("center_end")) if "center_end" in spec else None,
                        float(spec.get("time_start", 0.0)),
                        float(spec.get("time_end", 1.0)),
                    )
                )
            elif kind == "quad":
                scene.add(
                    Quad(
                        _vec3(spec["origin"]),
                        _vec3(spec["u"]),
                        _vec3(spec["v"]),
                        get_material(spec["material"]),
                    )
                )
            elif kind == "triangle":
                scene.add(
                    Triangle(
                        _vec3(spec["p0"]),
                        _vec3(spec["p1"]),
                        _vec3(spec["p2"]),
                        get_material(spec["material"]),
                        _vec2(spec.get("uv0"), Vec2()),
                        _vec2(spec.get("uv1"), Vec2(1, 0)),
                        _vec2(spec.get("uv2"), Vec2(0, 1)),
                    )
                )
            elif kind == "box":
                scene.add(*make_box(_vec3(spec["min"]), _vec3(spec["max"]), get_material(spec["material"])))
            elif kind == "lathe":
                profile = [(float(item[0]), float(item[1])) for item in spec["profile"]]
                scene.add(
                    *lathe(
                        profile,
                        get_material(spec["material"]),
                        center=_vec3(spec.get("center"), Vec3()),
                        segments=int(spec.get("segments", 32)),
                    )
                )
            elif kind == "obj":
                scale_value = spec.get("scale", [1, 1, 1])
                if isinstance(scale_value, (int, float)):
                    scale_value = [scale_value] * 3
                transform = Transform(
                    _vec3(scale_value),
                    float(spec.get("rotate_y", 0.0)),
                    _vec3(spec.get("translate"), Vec3()),
                )
                override = get_material(spec["material"]) if "material" in spec else None
                imported = load_obj(root / spec["file"], transform=transform, material_override=override)
                scene.add(*imported.triangles)
            else:
                raise ValueError(f"Unknown object type {kind!r}")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Object {index}: {error}") from error

    camera_spec = data.get("camera", {})
    render = dict(data.get("render", {}))
    width = int(render.get("width", 640))
    height = int(render.get("height", 360))
    camera = Camera(
        _vec3(camera_spec.get("look_from", [0, 1, 5])),
        _vec3(camera_spec.get("look_at", [0, 1, 0])),
        _vec3(camera_spec.get("up", [0, 1, 0])),
        float(camera_spec.get("vertical_fov", 45.0)),
        width / height,
        float(camera_spec.get("aperture", 0.0)),
        float(camera_spec["focus_distance"]) if "focus_distance" in camera_spec else None,
        float(camera_spec.get("shutter_open", 0.0)),
        float(camera_spec.get("shutter_close", 1.0)),
    )
    scene.commit(camera.shutter_open, camera.shutter_close)
    return LoadedScene(scene, camera, render)

