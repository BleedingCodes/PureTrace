"""Scene assembly, BVH ownership, lights, and homogeneous participating media."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .bvh import BVHNode
from .geometry import HitRecord, Primitive
from .math3d import Ray, Vec2, Vec3, ZERO, offset_point
from .rng import RNG
from .textures import EnvironmentMap


@dataclass(frozen=True, slots=True)
class HomogeneousMedium:
    density: float = 0.0
    albedo: Vec3 = Vec3(0.9, 0.9, 0.9)
    anisotropy: float = 0.0
    max_distance: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "density", max(0.0, self.density))
        object.__setattr__(self, "anisotropy", min(0.99, max(-0.99, self.anisotropy)))


@dataclass(frozen=True, slots=True)
class LightSample:
    direction: Vec3
    radiance: Vec3
    pdf: float
    distance: float


@dataclass(slots=True)
class Scene:
    primitives: list[Primitive] = field(default_factory=list)
    environment: EnvironmentMap = field(default_factory=EnvironmentMap)
    medium: HomogeneousMedium | None = None
    _bvh: BVHNode | None = field(default=None, init=False, repr=False)
    _lights: list[Primitive] = field(default_factory=list, init=False, repr=False)

    def add(self, *primitives: Primitive) -> None:
        self.primitives.extend(primitives)
        self._bvh = None

    def commit(self, time0: float = 0.0, time1: float = 1.0) -> None:
        self._bvh = BVHNode.build(self.primitives, time0, time1) if self.primitives else None
        self._lights = [obj for obj in self.primitives if obj.material.is_emissive]

    @property
    def light_count(self) -> int:
        return len(self._lights) + int(self.environment.enabled)

    def hit(self, ray: Ray, t_min: float = 1.0e-5, t_max: float = math.inf) -> HitRecord | None:
        if self._bvh is None and self.primitives:
            self.commit()
        return self._bvh.hit(ray, t_min, t_max) if self._bvh is not None else None

    def sample_light(self, point: Vec3, rng: RNG, time: float) -> LightSample | None:
        count = self.light_count
        if count == 0:
            return None
        choice = min(count - 1, int(rng.random() * count))
        if choice == len(self._lights):
            direction, radiance, pdf = self.environment.sample(rng)
            return LightSample(direction, radiance, pdf / count, math.inf)
        primitive = self._lights[choice]
        sample = primitive.sample_surface(rng, time)
        delta = sample.point - point
        distance_squared = delta.length_squared()
        if distance_squared <= 1.0e-12:
            return None
        distance = math.sqrt(distance_squared)
        direction = delta / distance
        light_cosine = sample.normal.dot(-direction)
        if sample.material.two_sided:
            light_cosine = abs(light_cosine)
        if light_cosine <= 1.0e-10:
            return None
        pdf = sample.pdf_area * distance_squared / light_cosine / count
        radiance = sample.material.emitted(
            sample.uv, sample.point, sample.normal.dot(-direction) > 0.0
        )
        return LightSample(direction, radiance, pdf, distance)

    def light_pdf(self, point: Vec3, direction: Vec3, hit: HitRecord | None) -> float:
        count = self.light_count
        if count == 0:
            return 0.0
        if hit is None:
            return self.environment.pdf(direction) / count if self.environment.enabled else 0.0
        if not hit.material.is_emissive:
            return 0.0
        return hit.primitive.pdf_direction(point, direction, 0.0) / count

    def shadow_transmittance(
        self, origin: Vec3, direction: Vec3, distance: float, time: float
    ) -> Vec3:
        max_distance = distance
        medium_distance = distance
        if not math.isfinite(medium_distance):
            medium_distance = self.medium.max_distance if self.medium else 1.0e6
        medium_t = 1.0
        if self.medium is not None and self.medium.density > 0.0:
            medium_t = math.exp(-self.medium.density * medium_distance)
        transmittance = Vec3(medium_t, medium_t, medium_t)
        ray_origin = origin
        remaining = max_distance
        for _ in range(16):
            hit = self.hit(Ray(ray_origin, direction, time), 1.0e-5, remaining)
            if hit is None:
                return transmittance
            opacity = hit.material.opacity
            # Refractive interfaces do not transmit a straight shadow ray; their
            # focused contribution is discovered by actual specular paths.
            if opacity >= 0.999 or hit.material.transmission > 0.0:
                return ZERO
            transmittance = transmittance * (1.0 - opacity)
            if transmittance.max_component() <= 1.0e-6:
                return ZERO
            travelled = hit.t
            if math.isfinite(remaining):
                remaining -= travelled
                if remaining <= 1.0e-5:
                    return transmittance
            ray_origin = offset_point(hit.point, hit.geom_normal, direction)
        return ZERO

