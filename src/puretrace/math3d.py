"""Small allocation-conscious vector, ray, and sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math


EPSILON = 1.0e-7
PI = math.pi
TAU = math.tau


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, value: float) -> "Vec2":
        return Vec2(self.x * value, self.y * value)

    def __rmul__(self, value: float) -> "Vec2":
        return Vec2(self.x * value, self.y * value)

    def __truediv__(self, value: float) -> "Vec2":
        inv = 1.0 / value
        return Vec2(self.x * inv, self.y * inv)


@dataclass(frozen=True, slots=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, other: float | "Vec3") -> "Vec3":
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float | "Vec3") -> "Vec3":
        return self * other

    def __truediv__(self, value: float) -> "Vec3":
        inv = 1.0 / value
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def __getitem__(self, axis: int) -> float:
        if axis == 0:
            return self.x
        if axis == 1:
            return self.y
        if axis == 2:
            return self.z
        raise IndexError(axis)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> "Vec3":
        length = self.length()
        if length <= EPSILON:
            return Vec3()
        return self / length

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)

    def min_component(self) -> float:
        return min(self.x, self.y, self.z)

    def luminance(self) -> float:
        return 0.2126 * self.x + 0.7152 * self.y + 0.0722 * self.z

    def is_finite(self) -> bool:
        return math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.z)

    def near_zero(self) -> bool:
        return abs(self.x) < EPSILON and abs(self.y) < EPSILON and abs(self.z) < EPSILON

    def clamped(self, low: float = 0.0, high: float = 1.0) -> "Vec3":
        return Vec3(
            min(high, max(low, self.x)),
            min(high, max(low, self.y)),
            min(high, max(low, self.z)),
        )


ZERO = Vec3(0.0, 0.0, 0.0)
ONE = Vec3(1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class Ray:
    origin: Vec3
    direction: Vec3
    time: float = 0.0

    def at(self, t: float) -> Vec3:
        return self.origin + self.direction * t


@dataclass(frozen=True, slots=True)
class AABB:
    minimum: Vec3
    maximum: Vec3

    @classmethod
    def empty(cls) -> "AABB":
        inf = math.inf
        return cls(Vec3(inf, inf, inf), Vec3(-inf, -inf, -inf))

    @classmethod
    def from_points(cls, a: Vec3, b: Vec3) -> "AABB":
        return cls(
            Vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)),
            Vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)),
        )

    def union(self, other: "AABB") -> "AABB":
        return AABB(
            Vec3(
                min(self.minimum.x, other.minimum.x),
                min(self.minimum.y, other.minimum.y),
                min(self.minimum.z, other.minimum.z),
            ),
            Vec3(
                max(self.maximum.x, other.maximum.x),
                max(self.maximum.y, other.maximum.y),
                max(self.maximum.z, other.maximum.z),
            ),
        )

    def padded(self, amount: float = 1.0e-5) -> "AABB":
        d = Vec3(amount, amount, amount)
        return AABB(self.minimum - d, self.maximum + d)

    def centroid(self) -> Vec3:
        return (self.minimum + self.maximum) * 0.5

    def extent(self) -> Vec3:
        return self.maximum - self.minimum

    def surface_area(self) -> float:
        d = self.extent()
        return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)

    def longest_axis(self) -> int:
        d = self.extent()
        if d.x >= d.y and d.x >= d.z:
            return 0
        return 1 if d.y >= d.z else 2

    def hit(self, ray: Ray, t_min: float, t_max: float) -> bool:
        for axis in range(3):
            direction = ray.direction[axis]
            origin = ray.origin[axis]
            lo = self.minimum[axis]
            hi = self.maximum[axis]
            if abs(direction) < EPSILON:
                if origin < lo or origin > hi:
                    return False
                continue
            inv = 1.0 / direction
            near = (lo - origin) * inv
            far = (hi - origin) * inv
            if inv < 0.0:
                near, far = far, near
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_max <= t_min:
                return False
        return True


def lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a * (1.0 - t) + b * t


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    return direction - normal * (2.0 * direction.dot(normal))


def refract(direction: Vec3, normal: Vec3, eta: float) -> Vec3:
    cos_theta = min((-direction).dot(normal), 1.0)
    perpendicular = (direction + normal * cos_theta) * eta
    parallel = normal * -math.sqrt(max(0.0, 1.0 - perpendicular.length_squared()))
    return perpendicular + parallel


def orthonormal_basis(normal: Vec3) -> tuple[Vec3, Vec3]:
    """Frisvad-style stable basis around a unit normal."""
    if normal.z < -0.9999999:
        return Vec3(0.0, -1.0, 0.0), Vec3(-1.0, 0.0, 0.0)
    a = 1.0 / (1.0 + normal.z)
    b = -normal.x * normal.y * a
    tangent = Vec3(1.0 - normal.x * normal.x * a, b, -normal.x)
    bitangent = Vec3(b, 1.0 - normal.y * normal.y * a, -normal.y)
    return tangent, bitangent


def to_world(local: Vec3, normal: Vec3) -> Vec3:
    tangent, bitangent = orthonormal_basis(normal)
    return tangent * local.x + bitangent * local.y + normal * local.z


def offset_point(point: Vec3, normal: Vec3, direction: Vec3) -> Vec3:
    sign = 1.0 if normal.dot(direction) >= 0.0 else -1.0
    scale = 1.0e-5 * max(1.0, abs(point.x), abs(point.y), abs(point.z))
    return point + normal * (sign * scale)
