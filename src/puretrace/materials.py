"""Disney-inspired metallic/roughness materials and BSDF sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .math3d import PI, ONE, Vec2, Vec3, ZERO, clamp, reflect, refract, to_world
from .rng import RNG
from .textures import SolidColor, Texture, as_texture


@dataclass(frozen=True, slots=True)
class BSDFSample:
    direction: Vec3
    weight: Vec3
    pdf: float
    delta: bool = False
    null: bool = False


def _fresnel_schlick(cosine: float, f0: Vec3) -> Vec3:
    factor = (1.0 - clamp(cosine, 0.0, 1.0)) ** 5
    return f0 + (ONE - f0) * factor


def _dielectric_fresnel(cosine: float, eta_i: float, eta_t: float) -> float:
    cosine = clamp(cosine, -1.0, 1.0)
    entering = cosine > 0.0
    if not entering:
        eta_i, eta_t = eta_t, eta_i
        cosine = abs(cosine)
    sin_t = eta_i / eta_t * math.sqrt(max(0.0, 1.0 - cosine * cosine))
    if sin_t >= 1.0:
        return 1.0
    cos_t = math.sqrt(max(0.0, 1.0 - sin_t * sin_t))
    parallel = ((eta_t * cosine) - (eta_i * cos_t)) / (
        (eta_t * cosine) + (eta_i * cos_t)
    )
    perpendicular = ((eta_i * cosine) - (eta_t * cos_t)) / (
        (eta_i * cosine) + (eta_t * cos_t)
    )
    return 0.5 * (parallel * parallel + perpendicular * perpendicular)


def _ggx_distribution(no_h: float, alpha: float) -> float:
    a2 = alpha * alpha
    denominator = no_h * no_h * (a2 - 1.0) + 1.0
    return a2 / max(1.0e-12, PI * denominator * denominator)


def _smith_g1(no_v: float, alpha: float) -> float:
    if no_v <= 0.0:
        return 0.0
    a2 = alpha * alpha
    return 2.0 * no_v / (no_v + math.sqrt(a2 + (1.0 - a2) * no_v * no_v))


def _sample_ggx(normal: Vec3, alpha: float, rng: RNG) -> Vec3:
    u1 = rng.random()
    u2 = rng.random()
    a2 = alpha * alpha
    cos_theta = math.sqrt((1.0 - u1) / max(1.0e-12, 1.0 + (a2 - 1.0) * u1))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    phi = 2.0 * PI * u2
    return to_world(Vec3(sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta), normal)


@dataclass(slots=True)
class PrincipledMaterial:
    base_color: Texture | Vec3 | tuple[float, float, float] = field(
        default_factory=lambda: SolidColor(Vec3(0.8, 0.8, 0.8))
    )
    metallic: float = 0.0
    roughness: float = 0.5
    transmission: float = 0.0
    ior: float = 1.5
    emission: Texture | Vec3 | tuple[float, float, float] = field(
        default_factory=lambda: SolidColor(ZERO)
    )
    emission_strength: float = 0.0
    opacity: float = 1.0
    two_sided: bool = False
    name: str = "material"

    def __post_init__(self) -> None:
        self.base_color = as_texture(self.base_color)
        self.emission = as_texture(self.emission)
        self.metallic = clamp(float(self.metallic), 0.0, 1.0)
        self.roughness = clamp(float(self.roughness), 0.001, 1.0)
        self.transmission = clamp(float(self.transmission), 0.0, 1.0)
        self.ior = max(1.0001, float(self.ior))
        self.opacity = clamp(float(self.opacity), 0.0, 1.0)

    @property
    def is_emissive(self) -> bool:
        return self.emission_strength > 0.0

    def color(self, uv: Vec2, point: Vec3) -> Vec3:
        return self.base_color.value(uv, point)

    def emitted(self, uv: Vec2, point: Vec3, front_face: bool = True) -> Vec3:
        if self.emission_strength <= 0.0 or (not front_face and not self.two_sided):
            return ZERO
        return self.emission.value(uv, point) * self.emission_strength

    def _opaque_lobes(self, uv: Vec2, point: Vec3) -> tuple[Vec3, Vec3, float]:
        base = self.color(uv, point)
        dielectric_f0 = ((self.ior - 1.0) / (self.ior + 1.0)) ** 2
        f0 = Vec3(dielectric_f0, dielectric_f0, dielectric_f0) * (1.0 - self.metallic)
        f0 = f0 + base * self.metallic
        diffuse = base * (1.0 - self.metallic)
        diffuse_energy = diffuse.luminance()
        specular_energy = max(0.02, f0.luminance())
        if diffuse_energy <= 1.0e-8:
            specular_probability = 1.0
        else:
            specular_probability = clamp(
                specular_energy / (specular_energy + diffuse_energy), 0.1, 0.9
            )
        return diffuse, f0, specular_probability

    def evaluate(self, wo: Vec3, wi: Vec3, normal: Vec3, uv: Vec2, point: Vec3) -> Vec3:
        no_v = max(0.0, normal.dot(wo))
        no_l = max(0.0, normal.dot(wi))
        if no_v <= 0.0 or no_l <= 0.0 or self.opacity <= 0.0:
            return ZERO
        # Delta transmission is sampled separately and has no finite solid-angle value.
        opaque_weight = 1.0 - self.transmission
        if opaque_weight <= 0.0:
            return ZERO
        diffuse, f0, _ = self._opaque_lobes(uv, point)
        half_vector = (wo + wi).normalized()
        if half_vector.near_zero():
            return diffuse * (opaque_weight / PI)
        no_h = max(0.0, normal.dot(half_vector))
        vo_h = max(0.0, wo.dot(half_vector))
        alpha = max(0.002, self.roughness * self.roughness)
        d = _ggx_distribution(no_h, alpha)
        g = _smith_g1(no_v, alpha) * _smith_g1(no_l, alpha)
        fresnel = _fresnel_schlick(vo_h, f0)
        specular = fresnel * (d * g / max(1.0e-8, 4.0 * no_v * no_l))
        # Fresnel-reflected energy is unavailable to the diffuse substrate.
        diffuse_brdf = diffuse * (ONE - fresnel) * (1.0 / PI)
        return (diffuse_brdf + specular) * opaque_weight * self.opacity

    def pdf(self, wo: Vec3, wi: Vec3, normal: Vec3, uv: Vec2, point: Vec3) -> float:
        no_l = max(0.0, normal.dot(wi))
        no_v = max(0.0, normal.dot(wo))
        opaque_probability = self.opacity * (1.0 - self.transmission)
        if no_l <= 0.0 or no_v <= 0.0 or opaque_probability <= 0.0:
            return 0.0
        _, _, specular_probability = self._opaque_lobes(uv, point)
        diffuse_pdf = no_l / PI
        half_vector = (wo + wi).normalized()
        if half_vector.near_zero():
            specular_pdf = 0.0
        else:
            alpha = max(0.002, self.roughness * self.roughness)
            no_h = max(0.0, normal.dot(half_vector))
            vo_h = max(1.0e-8, wo.dot(half_vector))
            specular_pdf = _ggx_distribution(no_h, alpha) * no_h / (4.0 * vo_h)
        mixture = (1.0 - specular_probability) * diffuse_pdf + specular_probability * specular_pdf
        return opaque_probability * mixture

    def sample(
        self,
        wo: Vec3,
        normal: Vec3,
        uv: Vec2,
        point: Vec3,
        front_face: bool,
        rng: RNG,
    ) -> BSDFSample | None:
        # Alpha transparency is a null event; matching selection probabilities
        # cancel the lobe coefficient in the Monte Carlo weight.
        if rng.random() > self.opacity:
            return BSDFSample(-wo, ONE, 1.0 - self.opacity, True, True)

        if self.transmission > 0.0 and rng.random() < self.transmission:
            incident = -wo
            eta_i, eta_t = (1.0, self.ior) if front_face else (self.ior, 1.0)
            eta = eta_i / eta_t
            cos_theta = min(1.0, wo.dot(normal))
            sin2_theta = max(0.0, 1.0 - cos_theta * cos_theta)
            cannot_refract = eta * eta * sin2_theta > 1.0
            fresnel = _dielectric_fresnel(cos_theta, eta_i, eta_t)
            if cannot_refract or rng.random() < fresnel:
                direction = reflect(incident, normal).normalized()
                weight = ONE
            else:
                direction = refract(incident, normal, eta).normalized()
                weight = self.color(uv, point)
            return BSDFSample(direction, weight, self.opacity * self.transmission, True)

        opaque_probability = self.opacity * (1.0 - self.transmission)
        if opaque_probability <= 0.0:
            return None
        _, _, specular_probability = self._opaque_lobes(uv, point)
        if rng.random() < specular_probability:
            alpha = max(0.002, self.roughness * self.roughness)
            half_vector = _sample_ggx(normal, alpha, rng)
            if wo.dot(half_vector) < 0.0:
                half_vector = -half_vector
            direction = reflect(-wo, half_vector).normalized()
            if direction.dot(normal) <= 0.0:
                return None
        else:
            direction = to_world(rng.cosine_hemisphere(), normal).normalized()
        pdf = self.pdf(wo, direction, normal, uv, point)
        if pdf <= 1.0e-12:
            return None
        value = self.evaluate(wo, direction, normal, uv, point)
        weight = value * (max(0.0, normal.dot(direction)) / pdf)
        # GGX remains a finite-density lobe even at very low roughness. Only
        # ideal dielectric reflection/refraction and alpha null events are
        # marked delta for MIS purposes.
        return BSDFSample(direction, weight, pdf, False)
