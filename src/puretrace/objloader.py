"""Wavefront OBJ/MTL importer with triangulation and material conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from .geometry import Triangle
from .materials import PrincipledMaterial
from .math3d import Vec2, Vec3
from .textures import ImageTexture, load_image


@dataclass(frozen=True, slots=True)
class Transform:
    scale: Vec3 = Vec3(1.0, 1.0, 1.0)
    rotate_y: float = 0.0
    translate: Vec3 = Vec3()

    def point(self, point: Vec3) -> Vec3:
        value = point * self.scale
        angle = math.radians(self.rotate_y)
        cosine, sine = math.cos(angle), math.sin(angle)
        value = Vec3(cosine * value.x + sine * value.z, value.y, -sine * value.x + cosine * value.z)
        return value + self.translate

    def normal(self, normal: Vec3) -> Vec3:
        value = Vec3(
            normal.x / self.scale.x if self.scale.x != 0.0 else 0.0,
            normal.y / self.scale.y if self.scale.y != 0.0 else 0.0,
            normal.z / self.scale.z if self.scale.z != 0.0 else 0.0,
        )
        angle = math.radians(self.rotate_y)
        cosine, sine = math.cos(angle), math.sin(angle)
        return Vec3(cosine * value.x + sine * value.z, value.y, -sine * value.x + cosine * value.z).normalized()


@dataclass(frozen=True, slots=True)
class OBJResult:
    triangles: tuple[Triangle, ...]
    materials: dict[str, PrincipledMaterial]


def _numbers(parts: list[str], count: int) -> list[float]:
    if len(parts) < count + 1:
        raise ValueError(f"Expected {count} values after {parts[0]}")
    return [float(value) for value in parts[1 : count + 1]]


def load_mtl(path: str | Path) -> dict[str, PrincipledMaterial]:
    path = Path(path)
    records: dict[str, dict[str, object]] = {}
    current: dict[str, object] | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].lower()
        if key == "newmtl":
            name = line[len(parts[0]) :].strip()
            current = records.setdefault(name, {})
        elif current is not None:
            if key in ("kd", "ks", "ke"):
                current[key] = Vec3(*_numbers(parts, 3))
            elif key in ("ns", "ni", "d", "tr", "illum"):
                current[key] = float(parts[1])
            elif key == "map_kd":
                # Options are uncommon; using the final token handles paths from
                # the vast majority of exporters while keeping parsing predictable.
                current[key] = parts[-1]

    materials: dict[str, PrincipledMaterial] = {}
    for name, values in records.items():
        base = values.get("kd", Vec3(0.8, 0.8, 0.8))
        texture_path = values.get("map_kd")
        if isinstance(texture_path, str):
            base = ImageTexture(load_image(path.parent / texture_path))
        specular = values.get("ks", Vec3())
        shininess = float(values.get("ns", 40.0))
        roughness = math.sqrt(2.0 / max(2.0, shininess + 2.0))
        illum = int(float(values.get("illum", 2)))
        dissolve = float(values.get("d", 1.0))
        if "tr" in values:
            dissolve = 1.0 - float(values["tr"])
        glass = illum in (4, 6, 7, 9)
        metallic = 1.0 if isinstance(specular, Vec3) and specular.luminance() > 0.65 and illum in (2, 3, 5) else 0.0
        emission = values.get("ke", Vec3())
        materials[name] = PrincipledMaterial(
            base_color=base,
            metallic=metallic,
            roughness=roughness,
            transmission=1.0 if glass else 0.0,
            ior=float(values.get("ni", 1.5)),
            opacity=1.0 if glass else dissolve,
            emission=emission,
            emission_strength=1.0 if isinstance(emission, Vec3) and emission.max_component() > 0.0 else 0.0,
            name=name,
        )
    return materials


def _resolve_index(token: str, length: int) -> int:
    value = int(token)
    index = value - 1 if value > 0 else length + value
    if index < 0 or index >= length:
        raise IndexError(f"OBJ index {value} is out of range for {length} elements")
    return index


def load_obj(
    path: str | Path,
    *,
    transform: Transform = Transform(),
    material_override: PrincipledMaterial | None = None,
) -> OBJResult:
    path = Path(path)
    vertices: list[Vec3] = []
    texcoords: list[Vec2] = []
    normals: list[Vec3] = []
    faces: list[tuple[list[tuple[int, int | None, int | None]], str | None]] = []
    materials: dict[str, PrincipledMaterial] = {}
    active_material: str | None = None

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].lower()
        try:
            if key == "v":
                vertices.append(Vec3(*_numbers(parts, 3)))
            elif key == "vt":
                values = _numbers(parts, 2)
                texcoords.append(Vec2(values[0], values[1]))
            elif key == "vn":
                normals.append(Vec3(*_numbers(parts, 3)).normalized())
            elif key == "mtllib":
                for filename in parts[1:]:
                    materials.update(load_mtl(path.parent / filename))
            elif key == "usemtl":
                active_material = line[len(parts[0]) :].strip()
            elif key == "f":
                corners = []
                for corner in parts[1:]:
                    fields = corner.split("/")
                    vertex = _resolve_index(fields[0], len(vertices))
                    uv = _resolve_index(fields[1], len(texcoords)) if len(fields) > 1 and fields[1] else None
                    normal = _resolve_index(fields[2], len(normals)) if len(fields) > 2 and fields[2] else None
                    corners.append((vertex, uv, normal))
                if len(corners) < 3:
                    raise ValueError("A face needs at least three vertices")
                faces.append((corners, active_material))
        except (ValueError, IndexError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error

    fallback = material_override or PrincipledMaterial(name="OBJ default")
    triangles: list[Triangle] = []
    for corners, material_name in faces:
        material = material_override or materials.get(material_name or "", fallback)
        for index in range(1, len(corners) - 1):
            selected = (corners[0], corners[index], corners[index + 1])
            positions = [transform.point(vertices[item[0]]) for item in selected]
            uvs = [texcoords[item[1]] if item[1] is not None else Vec2() for item in selected]
            transformed_normals = [
                transform.normal(normals[item[2]]) if item[2] is not None else None for item in selected
            ]
            triangles.append(
                Triangle(
                    positions[0],
                    positions[1],
                    positions[2],
                    material,
                    uvs[0],
                    uvs[1],
                    uvs[2],
                    transformed_normals[0],
                    transformed_normals[1],
                    transformed_normals[2],
                )
            )
    return OBJResult(tuple(triangles), materials)

