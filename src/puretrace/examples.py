"""Procedural reference scenes exercising the renderer's major features."""

from __future__ import annotations

from collections.abc import Callable

from .camera import Camera
from .geometry import Quad, Sphere, lathe, make_box
from .materials import PrincipledMaterial
from .math3d import Vec3
from .scene import HomogeneousMedium, Scene
from .textures import CheckerTexture, EnvironmentMap, SolidColor


def _mat(color: Vec3, **kwargs) -> PrincipledMaterial:
    return PrincipledMaterial(base_color=color, **kwargs)


def cornell_box(aspect: float = 1.0) -> tuple[Scene, Camera]:
    white = _mat(Vec3(0.73, 0.73, 0.73), roughness=0.7, name="matte white")
    red = _mat(Vec3(0.63, 0.065, 0.05), roughness=0.75, name="red wall")
    green = _mat(Vec3(0.14, 0.45, 0.091), roughness=0.75, name="green wall")
    metal = _mat(Vec3(0.85, 0.65, 0.28), metallic=1.0, roughness=0.16, name="brushed gold")
    glass = _mat(Vec3(0.96, 0.985, 1.0), transmission=1.0, roughness=0.01, ior=1.52, name="glass")
    light = PrincipledMaterial(
        emission=Vec3(1.0, 0.91, 0.72), emission_strength=18.0, name="ceiling light"
    )
    scene = Scene(environment=EnvironmentMap(color=Vec3()))
    scene.add(
        Quad(Vec3(-1, 0, -1), Vec3(2, 0, 0), Vec3(0, 0, 2), white),
        Quad(Vec3(-1, 2, 1), Vec3(2, 0, 0), Vec3(0, 0, -2), white),
        Quad(Vec3(1, 0, -1), Vec3(-2, 0, 0), Vec3(0, 2, 0), white),
        Quad(Vec3(-1, 0, -1), Vec3(0, 0, 2), Vec3(0, 2, 0), red),
        Quad(Vec3(1, 0, 1), Vec3(0, 0, -2), Vec3(0, 2, 0), green),
        Quad(Vec3(-0.35, 1.99, -0.35), Vec3(0.7, 0, 0), Vec3(0, 0, 0.7), light),
        *make_box(Vec3(-0.72, 0, -0.55), Vec3(-0.12, 0.72, 0.12), white),
        Sphere(Vec3(0.48, 0.43, -0.33), 0.43, glass),
        Sphere(Vec3(0.52, 1.18, 0.18), 0.27, metal),
    )
    camera = Camera(Vec3(0, 1.02, 3.7), Vec3(0, 1.0, 0), vertical_fov=40, aspect_ratio=aspect)
    scene.commit()
    return scene, camera


def reflective_spheres(aspect: float = 16 / 9) -> tuple[Scene, Camera]:
    checker = CheckerTexture(
        SolidColor(Vec3(0.72, 0.72, 0.72)), SolidColor(Vec3(0.06, 0.07, 0.085)), 2.0, True
    )
    floor = PrincipledMaterial(base_color=checker, roughness=0.38, name="checker floor")
    chrome = _mat(Vec3(0.92, 0.95, 1.0), metallic=1.0, roughness=0.055, name="chrome")
    copper = _mat(Vec3(0.95, 0.42, 0.18), metallic=1.0, roughness=0.2, name="copper")
    glass = _mat(Vec3(0.94, 0.985, 1.0), transmission=1.0, roughness=0.008, ior=1.5, name="glass")
    ceramic = _mat(Vec3(0.12, 0.32, 0.85), roughness=0.22, name="blue ceramic")
    light = PrincipledMaterial(emission=Vec3(1.0, 0.88, 0.68), emission_strength=28.0, name="softbox")
    environment = EnvironmentMap(color=Vec3(0.055, 0.08, 0.14), strength=1.0)
    scene = Scene(environment=environment)
    scene.add(
        Quad(Vec3(-7, 0, -7), Vec3(14, 0, 0), Vec3(0, 0, 14), floor),
        Sphere(Vec3(-1.55, 0.83, 0.1), 0.83, chrome),
        Sphere(Vec3(0.0, 0.72, -0.35), 0.72, glass),
        Sphere(Vec3(1.45, 0.62, 0.25), 0.62, copper),
        Sphere(Vec3(0.55, 0.35, 1.2), 0.35, ceramic, Vec3(0.9, 0.35, 1.2)),
        Quad(Vec3(-1.8, 4.8, -1.5), Vec3(3.6, 0, 0), Vec3(0, 0, 2.3), light),
    )
    camera = Camera(
        Vec3(4.8, 2.7, 6.5),
        Vec3(0, 0.75, 0),
        vertical_fov=35,
        aspect_ratio=aspect,
        aperture=0.07,
        focus_distance=8.25,
    )
    scene.commit()
    return scene, camera


def glass_chess(aspect: float = 4 / 3) -> tuple[Scene, Camera]:
    board_texture = CheckerTexture(
        SolidColor(Vec3(0.78, 0.72, 0.61)), SolidColor(Vec3(0.055, 0.07, 0.09)), 8.0, True
    )
    board = PrincipledMaterial(base_color=board_texture, roughness=0.22, name="chess board")
    clear = _mat(Vec3(0.90, 0.97, 1.0), transmission=1.0, roughness=0.018, ior=1.52, name="clear glass")
    smoke = _mat(Vec3(0.22, 0.34, 0.48), transmission=1.0, roughness=0.025, ior=1.52, name="smoked glass")
    frame = _mat(Vec3(0.18, 0.09, 0.035), metallic=0.55, roughness=0.18, name="board frame")
    warm_light = PrincipledMaterial(emission=Vec3(1.0, 0.78, 0.55), emission_strength=22.0, name="warm softbox")
    cool_light = PrincipledMaterial(emission=Vec3(0.58, 0.72, 1.0), emission_strength=14.0, name="cool softbox")
    scene = Scene(environment=EnvironmentMap(color=Vec3(0.018, 0.024, 0.04), strength=1.0))
    scene.add(
        Quad(Vec3(-4, 0.06, -4), Vec3(8, 0, 0), Vec3(0, 0, 8), board),
        *make_box(Vec3(-4.22, -0.12, -4.22), Vec3(4.22, 0.055, 4.22), frame),
        Quad(Vec3(-2.7, 5.5, -2.2), Vec3(4.2, 0, 0), Vec3(0, 0, 3.0), warm_light),
        Quad(Vec3(4.4, 3.6, 2.8), Vec3(0, 0, -3.4), Vec3(0, -2.2, 0), cool_light),
    )
    pawn_profile = [
        (0.29, 0.0), (0.34, 0.07), (0.31, 0.14), (0.20, 0.22),
        (0.13, 0.48), (0.20, 0.58), (0.13, 0.66), (0.24, 0.78),
        (0.25, 0.90), (0.17, 1.02), (0.0, 1.07),
    ]
    king_profile = [
        (0.40, 0.0), (0.46, 0.09), (0.40, 0.17), (0.25, 0.28),
        (0.18, 0.72), (0.28, 0.86), (0.20, 0.98), (0.25, 1.13),
        (0.12, 1.25), (0.12, 1.46), (0.24, 1.46), (0.24, 1.55),
        (0.12, 1.55), (0.12, 1.68), (0.0, 1.68),
    ]
    for x in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
        scene.add(*lathe(pawn_profile, clear, center=Vec3(x, 0.06, -2.5), segments=20))
    for x in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
        scene.add(*lathe(pawn_profile, smoke, center=Vec3(x, 0.06, 2.5), segments=20))
    scene.add(*lathe(king_profile, clear, center=Vec3(-0.5, 0.06, -3.45), segments=24))
    scene.add(*lathe(king_profile, smoke, center=Vec3(0.5, 0.06, 3.45), segments=24))
    camera = Camera(
        Vec3(7.1, 5.1, 8.4),
        Vec3(0, 0.55, 0),
        vertical_fov=37,
        aspect_ratio=aspect,
        aperture=0.085,
        focus_distance=11.4,
    )
    scene.commit()
    return scene, camera


def caustics(aspect: float = 4 / 3) -> tuple[Scene, Camera]:
    floor = _mat(Vec3(0.72, 0.74, 0.77), roughness=0.78, name="diffuse receiver")
    wall = _mat(Vec3(0.45, 0.52, 0.62), roughness=0.7, name="backdrop")
    glass = _mat(Vec3(0.85, 0.97, 1.0), transmission=1.0, roughness=0.004, ior=1.55, name="dense glass")
    metal = _mat(Vec3(0.92, 0.58, 0.22), metallic=1.0, roughness=0.07, name="gold")
    light = PrincipledMaterial(emission=Vec3(1.0, 0.83, 0.56), emission_strength=45.0, name="focused area light")
    scene = Scene(environment=EnvironmentMap(color=Vec3(0.008, 0.012, 0.02)))
    scene.add(
        Quad(Vec3(-4, 0, -4), Vec3(8, 0, 0), Vec3(0, 0, 8), floor),
        Quad(Vec3(4, 0, -3), Vec3(-8, 0, 0), Vec3(0, 5, 0), wall),
        Sphere(Vec3(-0.65, 0.95, -0.25), 0.95, glass),
        Sphere(Vec3(1.25, 0.7, 0.25), 0.7, metal),
        Quad(Vec3(-1.6, 4.8, -0.8), Vec3(1.3, 0, 0), Vec3(0, 0, 1.0), light),
    )
    camera = Camera(Vec3(4.8, 2.8, 6.8), Vec3(0, 0.8, 0), vertical_fov=37, aspect_ratio=aspect)
    scene.commit()
    return scene, camera


def volumetric_fog(aspect: float = 16 / 9) -> tuple[Scene, Camera]:
    floor = _mat(Vec3(0.18, 0.19, 0.21), roughness=0.65, name="floor")
    metal = _mat(Vec3(0.78, 0.83, 0.9), metallic=1.0, roughness=0.12, name="metal")
    red = _mat(Vec3(0.65, 0.04, 0.025), roughness=0.28, name="red")
    light = PrincipledMaterial(emission=Vec3(1.0, 0.64, 0.36), emission_strength=35.0, name="fog light")
    scene = Scene(
        environment=EnvironmentMap(color=Vec3(0.006, 0.01, 0.018)),
        medium=HomogeneousMedium(0.09, Vec3(0.93, 0.95, 1.0), 0.25, 40.0),
    )
    scene.add(
        Quad(Vec3(-8, 0, -8), Vec3(16, 0, 0), Vec3(0, 0, 16), floor),
        Sphere(Vec3(-1.2, 1.0, -0.6), 1.0, metal),
        Sphere(Vec3(1.25, 0.7, 0.4), 0.7, red),
        Quad(Vec3(-0.6, 3.8, -3.5), Vec3(1.2, 0, 0), Vec3(0, 0, 1.0), light),
    )
    camera = Camera(Vec3(5.8, 2.4, 7.2), Vec3(0, 0.8, -0.4), vertical_fov=38, aspect_ratio=aspect)
    scene.commit()
    return scene, camera


EXAMPLES: dict[str, Callable[[float], tuple[Scene, Camera]]] = {
    "cornell": cornell_box,
    "spheres": reflective_spheres,
    "glass-chess": glass_chess,
    "caustics": caustics,
    "fog": volumetric_fog,
}


def build_example(name: str, aspect: float) -> tuple[Scene, Camera]:
    try:
        return EXAMPLES[name](aspect)
    except KeyError as error:
        raise ValueError(f"Unknown example {name!r}; choose from {', '.join(EXAMPLES)}") from error
