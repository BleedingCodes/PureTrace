"""Geometric primitives, intersections, light sampling, and mesh generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .math3d import AABB, EPSILON, PI, Ray, Vec2, Vec3, clamp, lerp
from .rng import RNG

if TYPE_CHECKING:
    from .materials import PrincipledMaterial


@dataclass(frozen=True, slots=True)
class HitRecord:
    t: float
    point: Vec3
    normal: Vec3
    geom_normal: Vec3
    uv: Vec2
    material: "PrincipledMaterial"
    front_face: bool
    primitive: "Primitive"


@dataclass(frozen=True, slots=True)
class SurfaceSample:
    point: Vec3
    normal: Vec3
    uv: Vec2
    material: "PrincipledMaterial"
    pdf_area: float
    primitive: "Primitive"


class Primitive:
    material: "PrincipledMaterial"

    def bounds(self, time0: float = 0.0, time1: float = 1.0) -> AABB:
        raise NotImplementedError

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        raise NotImplementedError

    def surface_area(self) -> float:
        return 0.0

    def sample_surface(self, rng: RNG, time: float = 0.0) -> SurfaceSample:
        raise NotImplementedError

    def pdf_direction(self, origin: Vec3, direction: Vec3, time: float = 0.0) -> float:
        hit = self.hit(Ray(origin, direction, time), 1.0e-5, math.inf)
        if hit is None:
            return 0.0
        area = self.surface_area()
        if area <= 0.0:
            return 0.0
        distance_squared = hit.t * hit.t * direction.length_squared()
        cosine = abs(hit.geom_normal.dot(-direction.normalized()))
        if cosine <= 1.0e-10:
            return 0.0
        return distance_squared / (cosine * area)


@dataclass(slots=True)
class Sphere(Primitive):
    center0: Vec3
    radius: float
    material: "PrincipledMaterial"
    center1: Vec3 | None = None
    time0: float = 0.0
    time1: float = 1.0

    def center(self, time: float) -> Vec3:
        if self.center1 is None or self.time1 == self.time0:
            return self.center0
        amount = clamp((time - self.time0) / (self.time1 - self.time0), 0.0, 1.0)
        return lerp(self.center0, self.center1, amount)

    def bounds(self, time0: float = 0.0, time1: float = 1.0) -> AABB:
        radius = Vec3(self.radius, self.radius, self.radius)
        box0 = AABB(self.center(time0) - radius, self.center(time0) + radius)
        box1 = AABB(self.center(time1) - radius, self.center(time1) + radius)
        return box0.union(box1)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        center = self.center(ray.time)
        oc = ray.origin - center
        a = ray.direction.length_squared()
        half_b = oc.dot(ray.direction)
        c = oc.length_squared() - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        if discriminant < 0.0:
            return None
        root_term = math.sqrt(discriminant)
        root = (-half_b - root_term) / a
        if root <= t_min or root >= t_max:
            root = (-half_b + root_term) / a
            if root <= t_min or root >= t_max:
                return None
        point = ray.at(root)
        outward = (point - center) / self.radius
        front_face = ray.direction.dot(outward) < 0.0
        normal = outward if front_face else -outward
        phi = math.atan2(outward.z, outward.x)
        theta = math.acos(clamp(outward.y, -1.0, 1.0))
        uv = Vec2(0.5 + phi / (2.0 * PI), 1.0 - theta / PI)
        return HitRecord(root, point, normal, normal, uv, self.material, front_face, self)

    def surface_area(self) -> float:
        return 4.0 * PI * self.radius * self.radius

    def sample_surface(self, rng: RNG, time: float = 0.0) -> SurfaceSample:
        normal = rng.uniform_sphere()
        point = self.center(time) + normal * self.radius
        phi = math.atan2(normal.z, normal.x)
        theta = math.acos(clamp(normal.y, -1.0, 1.0))
        uv = Vec2(0.5 + phi / (2.0 * PI), 1.0 - theta / PI)
        return SurfaceSample(point, normal, uv, self.material, 1.0 / self.surface_area(), self)


@dataclass(slots=True)
class Triangle(Primitive):
    p0: Vec3
    p1: Vec3
    p2: Vec3
    material: "PrincipledMaterial"
    uv0: Vec2 = Vec2()
    uv1: Vec2 = Vec2(1.0, 0.0)
    uv2: Vec2 = Vec2(0.0, 1.0)
    n0: Vec3 | None = None
    n1: Vec3 | None = None
    n2: Vec3 | None = None

    def _outward_normal(self) -> Vec3:
        return (self.p1 - self.p0).cross(self.p2 - self.p0).normalized()

    def bounds(self, time0: float = 0.0, time1: float = 1.0) -> AABB:
        minimum = Vec3(
            min(self.p0.x, self.p1.x, self.p2.x),
            min(self.p0.y, self.p1.y, self.p2.y),
            min(self.p0.z, self.p1.z, self.p2.z),
        )
        maximum = Vec3(
            max(self.p0.x, self.p1.x, self.p2.x),
            max(self.p0.y, self.p1.y, self.p2.y),
            max(self.p0.z, self.p1.z, self.p2.z),
        )
        return AABB(minimum, maximum).padded()

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        edge1 = self.p1 - self.p0
        edge2 = self.p2 - self.p0
        pvec = ray.direction.cross(edge2)
        determinant = edge1.dot(pvec)
        if abs(determinant) < EPSILON:
            return None
        inv_det = 1.0 / determinant
        tvec = ray.origin - self.p0
        b1 = tvec.dot(pvec) * inv_det
        if b1 < 0.0 or b1 > 1.0:
            return None
        qvec = tvec.cross(edge1)
        b2 = ray.direction.dot(qvec) * inv_det
        if b2 < 0.0 or b1 + b2 > 1.0:
            return None
        t = edge2.dot(qvec) * inv_det
        if t <= t_min or t >= t_max:
            return None
        b0 = 1.0 - b1 - b2
        outward = edge1.cross(edge2).normalized()
        front_face = ray.direction.dot(outward) < 0.0
        oriented_geom = outward if front_face else -outward
        if self.n0 is not None and self.n1 is not None and self.n2 is not None:
            shading = (self.n0 * b0 + self.n1 * b1 + self.n2 * b2).normalized()
            if shading.dot(outward) < 0.0:
                shading = -shading
            if not front_face:
                shading = -shading
        else:
            shading = oriented_geom
        uv = self.uv0 * b0 + self.uv1 * b1 + self.uv2 * b2
        return HitRecord(t, ray.at(t), shading, oriented_geom, uv, self.material, front_face, self)

    def surface_area(self) -> float:
        return 0.5 * (self.p1 - self.p0).cross(self.p2 - self.p0).length()

    def sample_surface(self, rng: RNG, time: float = 0.0) -> SurfaceSample:
        r1 = math.sqrt(rng.random())
        r2 = rng.random()
        b0 = 1.0 - r1
        b1 = r1 * (1.0 - r2)
        b2 = r1 * r2
        point = self.p0 * b0 + self.p1 * b1 + self.p2 * b2
        uv = self.uv0 * b0 + self.uv1 * b1 + self.uv2 * b2
        normal = self._outward_normal()
        return SurfaceSample(point, normal, uv, self.material, 1.0 / self.surface_area(), self)


@dataclass(slots=True)
class Quad(Primitive):
    origin: Vec3
    u: Vec3
    v: Vec3
    material: "PrincipledMaterial"

    def _normal(self) -> Vec3:
        return self.u.cross(self.v).normalized()

    def bounds(self, time0: float = 0.0, time1: float = 1.0) -> AABB:
        points = (self.origin, self.origin + self.u, self.origin + self.v, self.origin + self.u + self.v)
        minimum = Vec3(
            min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)
        )
        maximum = Vec3(
            max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)
        )
        return AABB(minimum, maximum).padded()

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        normal = self._normal()
        denominator = normal.dot(ray.direction)
        if abs(denominator) < EPSILON:
            return None
        t = normal.dot(self.origin - ray.origin) / denominator
        if t <= t_min or t >= t_max:
            return None
        point = ray.at(t)
        relative = point - self.origin
        uu = self.u.dot(self.u)
        uv_cross = self.u.dot(self.v)
        vv = self.v.dot(self.v)
        ru = relative.dot(self.u)
        rv = relative.dot(self.v)
        determinant = uu * vv - uv_cross * uv_cross
        if abs(determinant) < EPSILON:
            return None
        a = (ru * vv - rv * uv_cross) / determinant
        b = (rv * uu - ru * uv_cross) / determinant
        if a < 0.0 or a > 1.0 or b < 0.0 or b > 1.0:
            return None
        front_face = ray.direction.dot(normal) < 0.0
        oriented = normal if front_face else -normal
        return HitRecord(t, point, oriented, oriented, Vec2(a, b), self.material, front_face, self)

    def surface_area(self) -> float:
        return self.u.cross(self.v).length()

    def sample_surface(self, rng: RNG, time: float = 0.0) -> SurfaceSample:
        a, b = rng.random(), rng.random()
        point = self.origin + self.u * a + self.v * b
        return SurfaceSample(
            point, self._normal(), Vec2(a, b), self.material, 1.0 / self.surface_area(), self
        )


def make_box(minimum: Vec3, maximum: Vec3, material: "PrincipledMaterial") -> list[Quad]:
    d = maximum - minimum
    return [
        Quad(Vec3(minimum.x, minimum.y, maximum.z), Vec3(d.x, 0, 0), Vec3(0, d.y, 0), material),
        Quad(Vec3(maximum.x, minimum.y, minimum.z), Vec3(-d.x, 0, 0), Vec3(0, d.y, 0), material),
        Quad(Vec3(maximum.x, minimum.y, maximum.z), Vec3(0, 0, -d.z), Vec3(0, d.y, 0), material),
        Quad(Vec3(minimum.x, minimum.y, minimum.z), Vec3(0, 0, d.z), Vec3(0, d.y, 0), material),
        Quad(Vec3(minimum.x, maximum.y, maximum.z), Vec3(d.x, 0, 0), Vec3(0, 0, -d.z), material),
        Quad(Vec3(minimum.x, minimum.y, minimum.z), Vec3(d.x, 0, 0), Vec3(0, 0, d.z), material),
    ]


def lathe(
    profile: list[tuple[float, float]],
    material: "PrincipledMaterial",
    *,
    center: Vec3 = Vec3(),
    segments: int = 32,
) -> list[Triangle]:
    """Revolve a radius/y profile into a smooth triangle mesh."""
    if len(profile) < 2 or segments < 3:
        raise ValueError("A lathe needs at least two profile points and three segments")
    normals: list[tuple[float, float]] = []
    for i, (radius, y) in enumerate(profile):
        before = profile[max(0, i - 1)]
        after = profile[min(len(profile) - 1, i + 1)]
        dr = after[0] - before[0]
        dy = after[1] - before[1]
        length = math.hypot(dy, dr)
        normals.append((dy / length if length else 1.0, -dr / length if length else 0.0))

    vertices: list[list[Vec3]] = []
    vertex_normals: list[list[Vec3]] = []
    for j, (radius, y) in enumerate(profile):
        ring: list[Vec3] = []
        normal_ring: list[Vec3] = []
        radial, vertical = normals[j]
        for i in range(segments):
            angle = 2.0 * PI * i / segments
            cosine, sine = math.cos(angle), math.sin(angle)
            ring.append(center + Vec3(radius * cosine, y, radius * sine))
            normal_ring.append(Vec3(radial * cosine, vertical, radial * sine).normalized())
        vertices.append(ring)
        vertex_normals.append(normal_ring)

    triangles: list[Triangle] = []
    height_segments = len(profile) - 1
    for j in range(height_segments):
        for i in range(segments):
            ni = (i + 1) % segments
            p00, p10 = vertices[j][i], vertices[j][ni]
            p01, p11 = vertices[j + 1][i], vertices[j + 1][ni]
            n00, n10 = vertex_normals[j][i], vertex_normals[j][ni]
            n01, n11 = vertex_normals[j + 1][i], vertex_normals[j + 1][ni]
            u0, u1 = i / segments, (i + 1) / segments
            v0, v1 = j / height_segments, (j + 1) / height_segments
            triangles.append(
                Triangle(p00, p01, p11, material, Vec2(u0, v0), Vec2(u0, v1), Vec2(u1, v1), n00, n01, n11)
            )
            triangles.append(
                Triangle(p00, p11, p10, material, Vec2(u0, v0), Vec2(u1, v1), Vec2(u1, v0), n00, n11, n10)
            )
    return triangles

