"""Scene graph, light sampling, and homogeneous participating medium."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

from .math3d import Ray, Vec3, ZERO
from .rng import RNG
from .textures import EnvironmentMap

if TYPE_CHECKING:
    from .geometry import HitRecord, Primitive


@dataclass(frozen=True, slots=True)
class LightSample:
    direction: Vec3
    radiance: Vec3
    pdf: float
    distance: float


@dataclass(frozen=True, slots=True)
class HomogeneousMedium:
    density: float
    albedo: Vec3
    anisotropy: float
    max_distance: float = 100.0


@dataclass(slots=True)
class Scene:
    environment: EnvironmentMap = field(default_factory=EnvironmentMap)
    medium: HomogeneousMedium | None = None
    _primitives: list["Primitive"] = field(default_factory=list, init=False, repr=False)
    _lights: list["Primitive"] = field(default_factory=list, init=False, repr=False)
    _bvh: object = field(default=None, init=False, repr=False)

    def add(self, *primitives: "Primitive") -> None:
        for primitive in primitives:
            self._primitives.append(primitive)

    def commit(self, time0: float = 0.0, time1: float = 1.0) -> None:
        from .bvh import BVHNode
        self._lights = [p for p in self._primitives if p.material.is_emissive]
        if self._primitives:
            self._bvh = BVHNode.build(self._primitives, time0, time1)
        else:
            self._bvh = None

    def hit(self, ray: Ray, t_min: float = 1.0e-5, t_max: float = math.inf) -> "HitRecord | None":
        if self._bvh is None:
            return None
        return self._bvh.hit(ray, t_min, t_max)

    def sample_light(
        self, origin: Vec3, rng: RNG, time: float = 0.0
    ) -> LightSample | None:
        has_lights = bool(self._lights)
        has_env = self.environment.enabled

        if not has_lights and not has_env:
            return None

        if has_lights and has_env:
            use_primitive = rng.random() < 0.5
            mix_pdf = 0.5
        elif has_lights:
            use_primitive = True
            mix_pdf = 1.0
        else:
            use_primitive = False
            mix_pdf = 1.0

        if use_primitive:
            primitive = self._lights[int(rng.random() * len(self._lights)) % len(self._lights)]
            sample = primitive.sample_surface(rng, time)
            to_light = sample.point - origin
            distance_sq = to_light.length_squared()
            if distance_sq <= 0.0:
                return None
            distance = math.sqrt(distance_sq)
            direction = to_light / distance
            cosine = abs(sample.normal.dot(-direction))
            if cosine <= 1.0e-10 or sample.pdf_area <= 0.0:
                return None
            pdf_primitive = sample.pdf_area * distance_sq / cosine / len(self._lights)
            radiance = sample.material.emitted(sample.uv, sample.point, sample.normal.dot(-direction) >= 0.0)
            return LightSample(direction, radiance, pdf_primitive * mix_pdf, distance)
        else:
            direction, radiance, pdf_env = self.environment.sample(rng)
            if pdf_env <= 0.0:
                return None
            return LightSample(direction, radiance, pdf_env * mix_pdf, math.inf)

    def light_pdf(
        self, origin: Vec3, direction: Vec3, hit: "HitRecord | None"
    ) -> float:
        has_lights = bool(self._lights)
        has_env = self.environment.enabled
        if not has_lights and not has_env:
            return 0.0

        mix_weight = 0.5 if (has_lights and has_env) else 1.0
        pdf = 0.0

        if has_lights and hit is not None and hit.primitive in self._lights:
            area = hit.primitive.surface_area()
            if area > 0.0:
                distance_sq = hit.t * hit.t
                cosine = abs(hit.geom_normal.dot(-direction.normalized()))
                if cosine > 1.0e-10:
                    pdf += mix_weight * (distance_sq / (cosine * area)) / len(self._lights)

        if has_env and hit is None:
            pdf += mix_weight * self.environment.pdf(direction)

        return pdf

    def shadow_transmittance(
        self, origin: Vec3, direction: Vec3, distance: float, time: float = 0.0
    ) -> Vec3:
        ray = Ray(origin, direction, time)
        t_max = distance
        limit = t_max
        hit = self.hit(ray, 1.0e-5, limit)
        if hit is not None:
            # Opaque or semi-transparent blocker.
            if hit.material.opacity >= 1.0:
                return ZERO
            transmittance = Vec3(1.0 - hit.material.opacity, 1.0 - hit.material.opacity, 1.0 - hit.material.opacity)
            return transmittance
        return Vec3(1.0, 1.0, 1.0)
