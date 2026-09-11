"""Texture loading and sampling without third-party dependencies.

Supported inputs are PNG (8-bit, non-interlaced), PPM, and Radiance RGBE HDR.
PNG colors are converted from sRGB to linear; HDR and PPM can be treated as
linear or sRGB at load time.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import zlib

from .math3d import PI, TAU, Vec2, Vec3, ZERO, clamp, lerp
from .rng import RNG


def srgb_channel_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def srgb_to_linear(color: Vec3) -> Vec3:
    return Vec3(
        srgb_channel_to_linear(color.x),
        srgb_channel_to_linear(color.y),
        srgb_channel_to_linear(color.z),
    )


class Texture:
    def value(self, uv: Vec2, point: Vec3) -> Vec3:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SolidColor(Texture):
    color: Vec3

    def value(self, uv: Vec2, point: Vec3) -> Vec3:
        return self.color


@dataclass(frozen=True, slots=True)
class CheckerTexture(Texture):
    even: Texture
    odd: Texture
    scale: float = 4.0
    use_uv: bool = False

    def value(self, uv: Vec2, point: Vec3) -> Vec3:
        if self.use_uv:
            cell = math.floor(uv.x * self.scale) + math.floor(uv.y * self.scale)
        else:
            cell = (
                math.floor(point.x * self.scale)
                + math.floor(point.y * self.scale)
                + math.floor(point.z * self.scale)
            )
        return (self.even if cell % 2 == 0 else self.odd).value(uv, point)


@dataclass(frozen=True, slots=True)
class ImageData:
    width: int
    height: int
    pixels: tuple[Vec3, ...]

    def at(self, x: int, y: int) -> Vec3:
        return self.pixels[(y % self.height) * self.width + (x % self.width)]

    def bilinear(self, u: float, v: float, wrap_u: bool = True) -> Vec3:
        if wrap_u:
            u %= 1.0
        else:
            u = clamp(u, 0.0, 1.0)
        v = clamp(v, 0.0, 1.0)
        x = u * self.width - 0.5
        y = v * self.height - 0.5
        x0 = math.floor(x)
        y0 = math.floor(y)
        tx = x - x0
        ty = y - y0
        if wrap_u:
            xa, xb = x0 % self.width, (x0 + 1) % self.width
        else:
            xa = min(self.width - 1, max(0, x0))
            xb = min(self.width - 1, max(0, x0 + 1))
        ya = min(self.height - 1, max(0, y0))
        yb = min(self.height - 1, max(0, y0 + 1))
        top = lerp(self.at(xa, ya), self.at(xb, ya), tx)
        bottom = lerp(self.at(xa, yb), self.at(xb, yb), tx)
        return lerp(top, bottom, ty)


@dataclass(frozen=True, slots=True)
class ImageTexture(Texture):
    image: ImageData
    scale_u: float = 1.0
    scale_v: float = 1.0
    offset_u: float = 0.0
    offset_v: float = 0.0
    flip_v: bool = True
    wrap_u: bool = True

    def value(self, uv: Vec2, point: Vec3) -> Vec3:
        u = uv.x * self.scale_u + self.offset_u
        v = uv.y * self.scale_v + self.offset_v
        if self.flip_v:
            v = 1.0 - v
        return self.image.bilinear(u, v, self.wrap_u)


def _paeth(a: int, b: int, c: int) -> int:
    prediction = a + b - c
    pa = abs(prediction - a)
    pb = abs(prediction - b)
    pc = abs(prediction - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def load_png(path: str | Path, *, srgb: bool = True) -> ImageData:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    compressed = bytearray()
    palette: list[tuple[int, int, int]] = []
    transparency: bytes | None = None
    while pos + 12 <= len(data):
        size = struct.unpack_from(">I", data, pos)[0]
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + size]
        expected_crc = struct.unpack_from(">I", data, pos + 8 + size)[0]
        actual_crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"PNG CRC error in {kind!r}: {path}")
        pos += 12 + size
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif kind == b"PLTE":
            palette = [tuple(payload[i : i + 3]) for i in range(0, len(payload), 3)]
        elif kind == b"tRNS":
            transparency = payload
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    if bit_depth != 8 or interlace != 0:
        raise ValueError("PureTrace supports 8-bit, non-interlaced PNG textures")
    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported PNG color type {color_type}")
    channels = channels_by_type[color_type]
    stride = width * channels
    raw = zlib.decompress(bytes(compressed))
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"Malformed PNG data: expected {expected} bytes, got {len(raw)}")
    rows: list[bytes] = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        source = raw[cursor + 1 : cursor + 1 + stride]
        cursor += stride + 1
        row = bytearray(stride)
        for i, byte in enumerate(source):
            left = row[i - channels] if i >= channels else 0
            up = previous[i]
            upper_left = previous[i - channels] if i >= channels else 0
            if filter_type == 0:
                value = byte
            elif filter_type == 1:
                value = byte + left
            elif filter_type == 2:
                value = byte + up
            elif filter_type == 3:
                value = byte + ((left + up) >> 1)
            elif filter_type == 4:
                value = byte + _paeth(left, up, upper_left)
            else:
                raise ValueError(f"Unsupported PNG filter {filter_type}")
            row[i] = value & 0xFF
        rows.append(bytes(row))
        previous = row

    pixels: list[Vec3] = []
    for row in rows:
        for x in range(width):
            i = x * channels
            if color_type == 0 or color_type == 4:
                r = g = b = row[i]
            elif color_type == 3:
                index = row[i]
                if index >= len(palette):
                    raise ValueError("PNG palette index out of range")
                r, g, b = palette[index]
                if transparency is not None and index < len(transparency):
                    alpha = transparency[index] / 255.0
                    r, g, b = round(r * alpha), round(g * alpha), round(b * alpha)
            else:
                r, g, b = row[i], row[i + 1], row[i + 2]
            color = Vec3(r / 255.0, g / 255.0, b / 255.0)
            pixels.append(srgb_to_linear(color) if srgb else color)
    return ImageData(width, height, tuple(pixels))


def _ppm_tokens(data: bytes):
    token = bytearray()
    i = 0
    while i < len(data):
        byte = data[i]
        if byte == 35:  # '#'
            while i < len(data) and data[i] not in (10, 13):
                i += 1
        elif byte in b" \t\r\n":
            if token:
                yield bytes(token), i + 1
                token.clear()
        else:
            token.append(byte)
        i += 1
    if token:
        yield bytes(token), i


def load_ppm(path: str | Path, *, srgb: bool = True) -> ImageData:
    data = Path(path).read_bytes()
    tokens = _ppm_tokens(data)
    magic, _ = next(tokens)
    width_token, _ = next(tokens)
    height_token, _ = next(tokens)
    max_token, payload_start = next(tokens)
    width, height, maximum = int(width_token), int(height_token), int(max_token)
    if maximum <= 0 or maximum > 255:
        raise ValueError("Only 8-bit PPM files are supported")
    values: list[int]
    if magic == b"P3":
        values = [int(token) for token, _ in tokens]
    elif magic == b"P6":
        values = list(data[payload_start : payload_start + width * height * 3])
    else:
        raise ValueError(f"Unsupported PPM type {magic!r}")
    if len(values) < width * height * 3:
        raise ValueError("Truncated PPM file")
    pixels = []
    for i in range(0, width * height * 3, 3):
        color = Vec3(values[i] / maximum, values[i + 1] / maximum, values[i + 2] / maximum)
        pixels.append(srgb_to_linear(color) if srgb else color)
    return ImageData(width, height, tuple(pixels))


def load_hdr(path: str | Path) -> ImageData:
    data = Path(path).read_bytes()
    stream = memoryview(data)
    pos = 0

    def line() -> bytes:
        nonlocal pos
        end = data.find(b"\n", pos)
        if end < 0:
            raise ValueError("Truncated Radiance HDR header")
        result = bytes(stream[pos:end]).rstrip(b"\r")
        pos = end + 1
        return result

    signature = line()
    if signature not in (b"#?RADIANCE", b"#?RGBE"):
        raise ValueError(f"Not a Radiance HDR file: {path}")
    while line():
        pass
    resolution = line().split()
    if len(resolution) != 4 or resolution[0] not in (b"-Y", b"+Y"):
        raise ValueError("Unsupported Radiance HDR orientation")
    height = int(resolution[1])
    width = int(resolution[3])
    flip_y = resolution[0] == b"+Y"
    flip_x = resolution[2] == b"-X"
    scanlines: list[list[tuple[int, int, int, int]]] = []
    for _ in range(height):
        if pos + 4 > len(data):
            raise ValueError("Truncated Radiance HDR pixels")
        header = bytes(stream[pos : pos + 4])
        pos += 4
        if width >= 8 and width <= 32767 and header[:2] == b"\x02\x02":
            encoded_width = (header[2] << 8) | header[3]
            if encoded_width != width:
                raise ValueError("Radiance HDR scanline width mismatch")
            channels: list[list[int]] = []
            for _channel in range(4):
                values: list[int] = []
                while len(values) < width:
                    if pos >= len(data):
                        raise ValueError("Truncated Radiance HDR RLE data")
                    count = data[pos]
                    pos += 1
                    if count > 128:
                        run = count - 128
                        if run == 0 or pos >= len(data):
                            raise ValueError("Invalid Radiance HDR RLE run")
                        values.extend([data[pos]] * run)
                        pos += 1
                    else:
                        if count == 0 or pos + count > len(data):
                            raise ValueError("Invalid Radiance HDR literal run")
                        values.extend(data[pos : pos + count])
                        pos += count
                channels.append(values)
            scanline = list(zip(*channels))
        else:
            needed = width * 4
            flat = header + bytes(stream[pos : pos + needed - 4])
            pos += needed - 4
            if len(flat) != needed:
                raise ValueError("Truncated flat Radiance HDR data")
            scanline = [tuple(flat[i : i + 4]) for i in range(0, needed, 4)]
        if flip_x:
            scanline.reverse()
        scanlines.append(scanline)
    if flip_y:
        scanlines.reverse()

    pixels: list[Vec3] = []
    for row in scanlines:
        for r, g, b, exponent in row:
            if exponent == 0:
                pixels.append(ZERO)
            else:
                scale = math.ldexp(1.0, exponent - (128 + 8))
                pixels.append(Vec3(r * scale, g * scale, b * scale))
    return ImageData(width, height, tuple(pixels))


def load_image(path: str | Path, *, srgb: bool = True) -> ImageData:
    extension = Path(path).suffix.lower()
    if extension == ".png":
        return load_png(path, srgb=srgb)
    if extension in (".ppm", ".pnm"):
        return load_ppm(path, srgb=srgb)
    if extension in (".hdr", ".rgbe"):
        return load_hdr(path)
    raise ValueError(f"Unsupported texture format {extension!r}; use PNG, PPM, or HDR")


@dataclass(slots=True)
class EnvironmentMap:
    image: ImageData | None = None
    color: Vec3 = ZERO
    strength: float = 1.0
    rotation: float = 0.0
    _cdf: tuple[float, ...] = ()
    _weights: tuple[float, ...] = ()
    _total: float = 0.0

    def __post_init__(self) -> None:
        if self.image is None:
            return
        weights: list[float] = []
        cumulative: list[float] = []
        total = 0.0
        for y in range(self.image.height):
            theta = PI * ((y + 0.5) / self.image.height)
            sin_theta = max(1.0e-6, math.sin(theta))
            for x in range(self.image.width):
                weight = max(0.0, self.image.at(x, y).luminance()) * sin_theta
                total += weight
                weights.append(weight)
                cumulative.append(total)
        if total <= 0.0:
            weights = [1.0] * (self.image.width * self.image.height)
            cumulative = []
            total = 0.0
            for weight in weights:
                total += weight
                cumulative.append(total)
        self._weights = tuple(weights)
        self._cdf = tuple(cumulative)
        self._total = total

    @property
    def enabled(self) -> bool:
        if self.image is not None:
            return self.strength > 0.0
        return self.strength > 0.0 and self.color.max_component() > 0.0

    def radiance(self, direction: Vec3) -> Vec3:
        if self.image is None:
            return self.color * self.strength
        d = direction.normalized()
        u = 0.5 + math.atan2(d.z, d.x) / TAU + self.rotation / TAU
        v = math.acos(clamp(d.y, -1.0, 1.0)) / PI
        return self.image.bilinear(u, v, True) * self.strength

    def sample(self, rng: RNG) -> tuple[Vec3, Vec3, float]:
        if self.image is None:
            direction = rng.uniform_sphere()
            return direction, self.color * self.strength, 1.0 / (4.0 * PI)
        target = rng.random() * self._total
        index = min(len(self._cdf) - 1, bisect_left(self._cdf, target))
        x = index % self.image.width
        y = index // self.image.width
        u = (x + rng.random()) / self.image.width
        v = (y + rng.random()) / self.image.height
        theta = PI * v
        phi = TAU * (u - 0.5) - self.rotation
        sin_theta = math.sin(theta)
        direction = Vec3(math.cos(phi) * sin_theta, math.cos(theta), math.sin(phi) * sin_theta)
        probability_pixel = self._weights[index] / self._total
        pdf = probability_pixel * self.image.width * self.image.height
        pdf /= max(1.0e-8, 2.0 * PI * PI * sin_theta)
        return direction, self.radiance(direction), pdf

    def pdf(self, direction: Vec3) -> float:
        if self.image is None:
            return 1.0 / (4.0 * PI)
        d = direction.normalized()
        u = (0.5 + math.atan2(d.z, d.x) / TAU + self.rotation / TAU) % 1.0
        v = math.acos(clamp(d.y, -1.0, 1.0)) / PI
        x = min(self.image.width - 1, int(u * self.image.width))
        y = min(self.image.height - 1, int(v * self.image.height))
        probability_pixel = self._weights[y * self.image.width + x] / self._total
        sin_theta = max(1.0e-8, math.sin(PI * v))
        return probability_pixel * self.image.width * self.image.height / (
            2.0 * PI * PI * sin_theta
        )


def as_texture(value: Texture | Vec3 | tuple[float, float, float] | list[float]) -> Texture:
    if isinstance(value, Texture):
        return value
    if isinstance(value, Vec3):
        return SolidColor(value)
    return SolidColor(Vec3(float(value[0]), float(value[1]), float(value[2])))

