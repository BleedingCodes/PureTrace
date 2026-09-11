"""Fast deterministic random numbers, independent of process scheduling."""

from __future__ import annotations

import math

from .math3d import PI, TAU, Vec2, Vec3


MASK64 = (1 << 64) - 1


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def pixel_seed(base: int, x: int, y: int, sample: int) -> int:
    value = base & MASK64
    value = splitmix64(value ^ ((x + 1) * 0xD6E8FEB86659FD93))
    value = splitmix64(value ^ ((y + 1) * 0xA5A3564E27F8862B))
    return splitmix64(value ^ ((sample + 1) * 0x9E3779B97F4A7C15))


class RNG:
    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = splitmix64(seed)

    def uint64(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK64
        x ^= x >> 27
        self.state = x & MASK64
        return (x * 0x2545F4914F6CDD1D) & MASK64

    def random(self) -> float:
        return (self.uint64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.random()

    def cosine_hemisphere(self) -> Vec3:
        r = math.sqrt(self.random())
        phi = TAU * self.random()
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        return Vec3(x, y, math.sqrt(max(0.0, 1.0 - x * x - y * y)))

    def uniform_sphere(self) -> Vec3:
        z = 1.0 - 2.0 * self.random()
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = TAU * self.random()
        return Vec3(r * math.cos(phi), z, r * math.sin(phi))

    def uniform_disk(self) -> Vec2:
        # Shirley-Chiu concentric disk mapping.
        sx = 2.0 * self.random() - 1.0
        sy = 2.0 * self.random() - 1.0
        if sx == 0.0 and sy == 0.0:
            return Vec2()
        if abs(sx) > abs(sy):
            radius = sx
            theta = (PI / 4.0) * (sy / sx)
        else:
            radius = sy
            theta = PI / 2.0 - (PI / 4.0) * (sx / sy)
        return Vec2(radius * math.cos(theta), radius * math.sin(theta))

