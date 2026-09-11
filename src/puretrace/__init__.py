"""PureTrace: a standard-library-only, CPU path tracer."""

from .camera import Camera
from .materials import PrincipledMaterial
from .math3d import Vec2, Vec3
from .renderer import RenderConfig, Renderer
from .scene import Scene

__all__ = [
    "Camera",
    "PrincipledMaterial",
    "RenderConfig",
    "Renderer",
    "Scene",
    "Vec2",
    "Vec3",
]

__version__ = "1.0.0"

