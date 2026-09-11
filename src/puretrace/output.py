"""Pure-Python PNG and OpenEXR output."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import zlib

from .math3d import Vec3, clamp


def _linear_to_srgb(value: float) -> float:
    value = max(0.0, value)
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1.0 / 2.4) - 0.055


def _aces(value: float) -> float:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return clamp((value * (a * value + b)) / (value * (c * value + d) + e), 0.0, 1.0)


def tone_map(color: Vec3, exposure: float = 0.0, operator: str = "aces") -> Vec3:
    scale = 2.0**exposure
    color = color * scale
    if operator == "aces":
        mapped = Vec3(_aces(color.x), _aces(color.y), _aces(color.z))
    elif operator == "reinhard":
        mapped = Vec3(color.x / (1.0 + color.x), color.y / (1.0 + color.y), color.z / (1.0 + color.z))
    elif operator == "linear":
        mapped = color.clamped()
    else:
        raise ValueError(f"Unknown tone mapper {operator!r}")
    return Vec3(_linear_to_srgb(mapped.x), _linear_to_srgb(mapped.y), _linear_to_srgb(mapped.z))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(
    path: str | Path,
    width: int,
    height: int,
    pixels: list[Vec3] | tuple[Vec3, ...],
    *,
    exposure: float = 0.0,
    tone_mapper: str = "aces",
) -> None:
    if len(pixels) != width * height:
        raise ValueError("Pixel count does not match image dimensions")
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter: None
        for x in range(width):
            color = tone_map(pixels[y * width + x], exposure, tone_mapper)
            rows.extend(
                (
                    round(clamp(color.x, 0.0, 1.0) * 255.0),
                    round(clamp(color.y, 0.0, 1.0) * 255.0),
                    round(clamp(color.z, 0.0, 1.0) * 255.0),
                )
            )
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"sRGB", b"\x00")
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + _png_chunk(b"IEND", b"")
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)


def _exr_attribute(name: str, kind: str, value: bytes) -> bytes:
    return name.encode("ascii") + b"\x00" + kind.encode("ascii") + b"\x00" + struct.pack("<I", len(value)) + value


def write_exr(
    path: str | Path,
    width: int,
    height: int,
    pixels: list[Vec3] | tuple[Vec3, ...],
    *,
    half: bool = True,
) -> None:
    """Write an uncompressed scanline OpenEXR containing linear RGB values."""
    if len(pixels) != width * height:
        raise ValueError("Pixel count does not match image dimensions")
    pixel_type = 1 if half else 2  # HALF or FLOAT
    channel_list = bytearray()
    for name in ("B", "G", "R"):
        channel_list.extend(name.encode("ascii") + b"\x00")
        channel_list.extend(struct.pack("<iB3xii", pixel_type, 0, 1, 1))
    channel_list.append(0)
    window = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    header = bytearray()
    header.extend(_exr_attribute("channels", "chlist", bytes(channel_list)))
    header.extend(_exr_attribute("compression", "compression", b"\x00"))
    header.extend(_exr_attribute("dataWindow", "box2i", window))
    header.extend(_exr_attribute("displayWindow", "box2i", window))
    header.extend(_exr_attribute("lineOrder", "lineOrder", b"\x00"))
    header.extend(_exr_attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)))
    header.extend(_exr_attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)))
    header.extend(_exr_attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)))
    header.append(0)

    bytes_per_sample = 2 if half else 4
    scanline_size = width * 3 * bytes_per_sample
    chunk_size = 8 + scanline_size
    prefix_size = 8 + len(header) + height * 8
    offsets = [prefix_size + y * chunk_size for y in range(height)]
    result = bytearray(struct.pack("<II", 20000630, 2))
    result.extend(header)
    result.extend(struct.pack(f"<{height}Q", *offsets))
    sample_format = "<e" if half else "<f"
    for y in range(height):
        result.extend(struct.pack("<ii", y, scanline_size))
        row = pixels[y * width : (y + 1) * width]
        for component in ("z", "y", "x"):
            for color in row:
                value = getattr(color, component)
                if half:
                    value = min(65504.0, max(-65504.0, value))
                result.extend(struct.pack(sample_format, value))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(result))

