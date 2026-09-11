"""Thin-lens perspective camera with shutter-time sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .math3d import Ray, Vec3
from .rng import RNG


@dataclass(slots=True)
class Camera:
    look_from: Vec3
    look_at: Vec3
    up: Vec3 = Vec3(0.0, 1.0, 0.0)
    vertical_fov: float = 45.0
    aspect_ratio: float = 16.0 / 9.0
    aperture: float = 0.0
    focus_distance: float | None = None
    shutter_open: float = 0.0
    shutter_close: float = 1.0
    _origin: Vec3 = field(init=False, repr=False)
    _lower_left: Vec3 = field(init=False, repr=False)
    _horizontal: Vec3 = field(init=False, repr=False)
    _vertical: Vec3 = field(init=False, repr=False)
    _u: Vec3 = field(init=False, repr=False)
    _v: Vec3 = field(init=False, repr=False)
    _lens_radius: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        theta = math.radians(self.vertical_fov)
        viewport_height = 2.0 * math.tan(theta * 0.5)
        viewport_width = self.aspect_ratio * viewport_height
        w = (self.look_from - self.look_at).normalized()
        u = self.up.cross(w).normalized()
        if u.near_zero():
            raise ValueError("Camera up vector is parallel to its viewing direction")
        v = w.cross(u)
        focus = self.focus_distance
        if focus is None:
            focus = (self.look_from - self.look_at).length()
        self._origin = self.look_from
        self._horizontal = u * (viewport_width * focus)
        self._vertical = v * (viewport_height * focus)
        self._lower_left = (
            self._origin - self._horizontal * 0.5 - self._vertical * 0.5 - w * focus
        )
        self._u = u
        self._v = v
        self._lens_radius = self.aperture * 0.5

    def ray(self, s: float, t: float, rng: RNG) -> Ray:
        if self._lens_radius > 0.0:
            disk = rng.uniform_disk()
            offset = (self._u * disk.x + self._v * disk.y) * self._lens_radius
        else:
            offset = Vec3()
        if self.shutter_close > self.shutter_open:
            time = rng.uniform(self.shutter_open, self.shutter_close)
        else:
            time = self.shutter_open
        direction = (
            self._lower_left + self._horizontal * s + self._vertical * t - self._origin - offset
        ).normalized()
        return Ray(self._origin + offset, direction, time)

