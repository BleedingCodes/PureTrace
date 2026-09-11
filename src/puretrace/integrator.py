"""MIS path tracer with next-event estimation and homogeneous volumes."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .math3d import PI, Ray, Vec3, ZERO, offset_point, to_world
from .rng import RNG
from .scene import HomogeneousMedium, Scene


def power_heuristic(pdf_a: float, pdf_b: float) -> float:
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    return a2 / (a2 + b2) if a2 + b2 > 0.0 else 0.0


def henyey_greenstein(cosine: float, g: float) -> float:
    denominator = 1.0 + g * g - 2.0 * g * cosine
    return (1.0 - g * g) / (4.0 * PI * denominator ** 1.5)


def sample_henyey_greenstein(incoming: Vec3, g: float, rng: RNG) -> Vec3:
    u = rng.random()
    if abs(g) < 1.0e-3:
        cosine = 1.0 - 2.0 * u
    else:
        ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
        cosine = (1.0 + g * g - ratio * ratio) / (2.0 * g)
        cosine = min(1.0, max(-1.0, cosine))
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    phi = 2.0 * PI * rng.random()
    local = Vec3(sine * math.cos(phi), sine * math.sin(phi), cosine)
    return to_world(local, incoming).normalized()


@dataclass(frozen=True, slots=True)
class PathIntegrator:
    max_depth: int = 12
    russian_roulette_depth: int = 5
    sample_clamp: float = 0.0

    def _direct_surface(self, scene: Scene, hit, wo: Vec3, rng: RNG, time: float) -> Vec3:
        light = scene.sample_light(hit.point, rng, time)
        if light is None or light.pdf <= 0.0 or light.radiance.max_component() <= 0.0:
            return ZERO
        cosine = max(0.0, hit.normal.dot(light.direction))
        if cosine <= 0.0:
            return ZERO
        value = hit.material.evaluate(wo, light.direction, hit.normal, hit.uv, hit.point)
        if value.max_component() <= 0.0:
            return ZERO
        origin = offset_point(hit.point, hit.geom_normal, light.direction)
        limit = light.distance - 2.0e-5 if math.isfinite(light.distance) else math.inf
        visibility = scene.shadow_transmittance(origin, light.direction, limit, time)
        if visibility.max_component() <= 0.0:
            return ZERO
        bsdf_pdf = hit.material.pdf(wo, light.direction, hit.normal, hit.uv, hit.point)
        weight = power_heuristic(light.pdf, bsdf_pdf)
        return value * light.radiance * visibility * (cosine * weight / light.pdf)

    def _direct_medium(
        self, scene: Scene, point: Vec3, incoming: Vec3, medium: HomogeneousMedium, rng: RNG, time: float
    ) -> Vec3:
        light = scene.sample_light(point, rng, time)
        if light is None or light.pdf <= 0.0:
            return ZERO
        phase = henyey_greenstein(incoming.dot(light.direction), medium.anisotropy)
        origin = point + light.direction * 1.0e-5
        limit = light.distance - 2.0e-5 if math.isfinite(light.distance) else math.inf
        visibility = scene.shadow_transmittance(origin, light.direction, limit, time)
        weight = power_heuristic(light.pdf, phase)
        return light.radiance * visibility * (phase * weight / light.pdf)

    def radiance(self, initial_ray: Ray, scene: Scene, rng: RNG) -> Vec3:
        ray = initial_ray
        throughput = Vec3(1.0, 1.0, 1.0)
        result = ZERO
        depth = 0
        previous_point: Vec3 | None = None
        previous_pdf = 0.0
        previous_delta = True

        while depth < self.max_depth:
            hit = scene.hit(ray)
            surface_distance = hit.t if hit is not None else math.inf

            medium = scene.medium
            if medium is not None and medium.density > 0.0:
                segment_limit = min(surface_distance, medium.max_distance)
                scatter_distance = -math.log(max(1.0e-15, 1.0 - rng.random())) / medium.density
                if scatter_distance < segment_limit:
                    point = ray.at(scatter_distance)
                    result = result + throughput * self._direct_medium(
                        scene, point, ray.direction, medium, rng, ray.time
                    )
                    direction = sample_henyey_greenstein(ray.direction, medium.anisotropy, rng)
                    previous_pdf = henyey_greenstein(ray.direction.dot(direction), medium.anisotropy)
                    previous_point = point
                    previous_delta = False
                    throughput = throughput * medium.albedo
                    ray = Ray(point + direction * 1.0e-5, direction, ray.time)
                    depth += 1
                    continue

            if hit is None:
                environment = scene.environment.radiance(ray.direction)
                if environment.max_component() > 0.0:
                    weight = 1.0
                    if not previous_delta and previous_point is not None:
                        light_pdf = scene.light_pdf(previous_point, ray.direction, None)
                        weight = power_heuristic(previous_pdf, light_pdf)
                    result = result + throughput * environment * weight
                break

            emission = hit.material.emitted(hit.uv, hit.point, hit.front_face)
            if emission.max_component() > 0.0:
                weight = 1.0
                if not previous_delta and previous_point is not None:
                    light_pdf = scene.light_pdf(previous_point, ray.direction, hit)
                    weight = power_heuristic(previous_pdf, light_pdf)
                result = result + throughput * emission * weight

            wo = -ray.direction
            result = result + throughput * self._direct_surface(scene, hit, wo, rng, ray.time)
            sample = hit.material.sample(wo, hit.normal, hit.uv, hit.point, hit.front_face, rng)
            if sample is None or sample.weight.max_component() <= 0.0:
                break
            throughput = throughput * sample.weight
            if not throughput.is_finite():
                break
            previous_point = hit.point
            previous_pdf = sample.pdf
            previous_delta = sample.delta
            ray = Ray(
                offset_point(hit.point, hit.geom_normal, sample.direction), sample.direction, ray.time
            )
            if not sample.null:
                depth += 1

            if not sample.null and depth >= self.russian_roulette_depth:
                survival = min(0.95, max(0.05, throughput.max_component()))
                if rng.random() > survival:
                    break
                throughput = throughput / survival

        if self.sample_clamp > 0.0:
            maximum = result.max_component()
            if maximum > self.sample_clamp:
                result = result * (self.sample_clamp / maximum)
        return result
