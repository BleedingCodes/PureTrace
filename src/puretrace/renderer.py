"""Progressive tile scheduling, multiprocessing, and resumable accumulation."""

from __future__ import annotations

from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import struct
import sys
import time
from typing import Callable
import zlib

from .camera import Camera
from .integrator import PathIntegrator
from .math3d import Vec3
from .output import write_exr, write_png
from .rng import RNG, pixel_seed
from .scene import Scene


CHECKPOINT_MAGIC = b"PTRCHK1\x00"


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 640
    height: int = 360
    samples_per_pixel: int = 128
    max_depth: int = 12
    tile_size: int = 16
    workers: int = 0
    samples_per_pass: int = 1
    seed: int = 1337
    russian_roulette_depth: int = 5
    sample_clamp: float = 0.0
    checkpoint_interval: float = 30.0

    def __post_init__(self) -> None:
        for name in ("width", "height", "samples_per_pixel", "max_depth", "tile_size", "samples_per_pass"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class Tile:
    index: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def pixel_count(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True, slots=True)
class TileJob:
    tile: Tile
    sample_start: int
    sample_count: int


@dataclass(frozen=True, slots=True)
class TileResult:
    tile: Tile
    sample_count: int
    sums: tuple[float, ...]


@dataclass(slots=True)
class RenderState:
    width: int
    height: int
    tile_size: int
    sums: list[float]
    tile_samples: list[int]
    scene_digest: str
    seed: int
    max_depth: int

    @classmethod
    def fresh(cls, config: RenderConfig, tile_count: int, scene_digest: str) -> "RenderState":
        return cls(
            config.width,
            config.height,
            config.tile_size,
            [0.0] * (config.width * config.height * 3),
            [0] * tile_count,
            scene_digest,
            config.seed,
            config.max_depth,
        )

    def pixels(self, tiles: list[Tile]) -> list[Vec3]:
        result = [Vec3()] * (self.width * self.height)
        for tile in tiles:
            samples = self.tile_samples[tile.index]
            inv = 1.0 / samples if samples > 0 else 0.0
            for y in range(tile.y0, tile.y1):
                for x in range(tile.x0, tile.x1):
                    pixel = y * self.width + x
                    base = pixel * 3
                    result[pixel] = Vec3(
                        self.sums[base] * inv,
                        self.sums[base + 1] * inv,
                        self.sums[base + 2] * inv,
                    )
        return result

    def average_spp(self, tiles: list[Tile]) -> float:
        total = sum(self.tile_samples[t.index] * t.pixel_count for t in tiles)
        return total / (self.width * self.height)


@dataclass(frozen=True, slots=True)
class RenderResult:
    width: int
    height: int
    pixels: tuple[Vec3, ...]
    samples_per_pixel: int
    elapsed_seconds: float

    def save_png(self, path: str | Path, *, exposure: float = 0.0, tone_mapper: str = "aces") -> None:
        write_png(path, self.width, self.height, self.pixels, exposure=exposure, tone_mapper=tone_mapper)

    def save_exr(self, path: str | Path, *, half: bool = True) -> None:
        write_exr(path, self.width, self.height, self.pixels, half=half)


_WORKER_SCENE: Scene | None = None
_WORKER_CAMERA: Camera | None = None
_WORKER_INTEGRATOR: PathIntegrator | None = None
_WORKER_WIDTH = 0
_WORKER_HEIGHT = 0
_WORKER_SEED = 0


def _worker_init(
    scene: Scene, camera: Camera, integrator: PathIntegrator, width: int, height: int, seed: int
) -> None:
    global _WORKER_SCENE, _WORKER_CAMERA, _WORKER_INTEGRATOR
    global _WORKER_WIDTH, _WORKER_HEIGHT, _WORKER_SEED
    _WORKER_SCENE = scene
    _WORKER_CAMERA = camera
    _WORKER_INTEGRATOR = integrator
    _WORKER_WIDTH = width
    _WORKER_HEIGHT = height
    _WORKER_SEED = seed


def _render_tile(job: TileJob) -> TileResult:
    if _WORKER_SCENE is None or _WORKER_CAMERA is None or _WORKER_INTEGRATOR is None:
        raise RuntimeError("Render worker is not initialized")
    tile = job.tile
    sums: list[float] = []
    for y in range(tile.y0, tile.y1):
        for x in range(tile.x0, tile.x1):
            red = green = blue = 0.0
            for sample in range(job.sample_start, job.sample_start + job.sample_count):
                rng = RNG(pixel_seed(_WORKER_SEED, x, y, sample))
                u = (x + rng.random()) / _WORKER_WIDTH
                v = 1.0 - (y + rng.random()) / _WORKER_HEIGHT
                ray = _WORKER_CAMERA.ray(u, v, rng)
                color = _WORKER_INTEGRATOR.radiance(ray, _WORKER_SCENE, rng)
                red += color.x
                green += color.y
                blue += color.z
            sums.extend((red, green, blue))
    return TileResult(tile, job.sample_count, tuple(sums))


def make_tiles(width: int, height: int, tile_size: int) -> list[Tile]:
    tiles: list[Tile] = []
    index = 0
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tiles.append(Tile(index, x, y, min(width, x + tile_size), min(height, y + tile_size)))
            index += 1
    return tiles


def _merge_tile(state: RenderState, result: TileResult) -> None:
    cursor = 0
    tile = result.tile
    for y in range(tile.y0, tile.y1):
        for x in range(tile.x0, tile.x1):
            target = (y * state.width + x) * 3
            state.sums[target] += result.sums[cursor]
            state.sums[target + 1] += result.sums[cursor + 1]
            state.sums[target + 2] += result.sums[cursor + 2]
            cursor += 3
    state.tile_samples[tile.index] += result.sample_count


def save_checkpoint(path: str | Path, state: RenderState) -> None:
    metadata = {
        "width": state.width,
        "height": state.height,
        "tile_size": state.tile_size,
        "tile_samples": state.tile_samples,
        "scene_digest": state.scene_digest,
        "seed": state.seed,
        "max_depth": state.max_depth,
        "byteorder": sys.byteorder,
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    values = array("d", state.sums)
    if sys.byteorder != "little":
        values.byteswap()
    compressed = zlib.compress(values.tobytes(), 3)
    payload = CHECKPOINT_MAGIC + struct.pack("<I", len(metadata_bytes)) + metadata_bytes + compressed
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)


def load_checkpoint(path: str | Path) -> RenderState:
    data = Path(path).read_bytes()
    if not data.startswith(CHECKPOINT_MAGIC):
        raise ValueError("Not a PureTrace checkpoint")
    offset = len(CHECKPOINT_MAGIC)
    metadata_size = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    metadata = json.loads(data[offset : offset + metadata_size])
    raw = zlib.decompress(data[offset + metadata_size :])
    values = array("d")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    expected = metadata["width"] * metadata["height"] * 3
    if len(values) != expected:
        raise ValueError("Checkpoint accumulation buffer has the wrong size")
    return RenderState(
        metadata["width"],
        metadata["height"],
        metadata["tile_size"],
        list(values),
        list(metadata["tile_samples"]),
        metadata["scene_digest"],
        metadata["seed"],
        metadata["max_depth"],
    )


class Renderer:
    def __init__(self, config: RenderConfig):
        self.config = config

    def _validate_resume(self, state: RenderState, digest: str, tile_count: int) -> None:
        config = self.config
        checks = {
            "width": (state.width, config.width),
            "height": (state.height, config.height),
            "tile size": (state.tile_size, config.tile_size),
            "seed": (state.seed, config.seed),
            "max depth": (state.max_depth, config.max_depth),
            "scene": (state.scene_digest, digest),
            "tile count": (len(state.tile_samples), tile_count),
        }
        mismatches = [name for name, (old, new) in checks.items() if old != new]
        if mismatches:
            raise ValueError("Checkpoint does not match this render: " + ", ".join(mismatches))

    def render(
        self,
        scene: Scene,
        camera: Camera,
        *,
        checkpoint: str | Path | None = None,
        resume: bool = False,
        preview: str | Path | None = None,
        exposure: float = 0.0,
        tone_mapper: str = "aces",
        progress: Callable[[float, float, float], None] | None = None,
    ) -> RenderResult:
        config = self.config
        scene.commit(camera.shutter_open, camera.shutter_close)
        digest = hashlib.sha256(pickle.dumps((scene, camera), protocol=5)).hexdigest()
        tiles = make_tiles(config.width, config.height, config.tile_size)
        if resume:
            if checkpoint is None or not Path(checkpoint).exists():
                raise FileNotFoundError("Resume requested, but no checkpoint exists")
            state = load_checkpoint(checkpoint)
            self._validate_resume(state, digest, len(tiles))
        else:
            state = RenderState.fresh(config, len(tiles), digest)

        integrator = PathIntegrator(
            config.max_depth, config.russian_roulette_depth, config.sample_clamp
        )
        workers = config.workers if config.workers > 0 else max(1, os.cpu_count() or 1)
        center_x, center_y = config.width * 0.5, config.height * 0.5
        dispatch_tiles = sorted(
            tiles,
            key=lambda tile: (
                ((tile.x0 + tile.x1) * 0.5 - center_x) ** 2
                + ((tile.y0 + tile.y1) * 0.5 - center_y) ** 2
            ),
        )
        start_time = time.monotonic()
        last_checkpoint = start_time

        def report() -> None:
            if progress is not None:
                elapsed = time.monotonic() - start_time
                average = state.average_spp(tiles)
                fraction = min(1.0, average / config.samples_per_pixel)
                progress(fraction, average, elapsed)

        def persist() -> None:
            nonlocal last_checkpoint
            if checkpoint is not None:
                save_checkpoint(checkpoint, state)
                last_checkpoint = time.monotonic()

        def jobs_for_round() -> list[TileJob]:
            jobs = []
            for tile in dispatch_tiles:
                completed = state.tile_samples[tile.index]
                if completed < config.samples_per_pixel:
                    count = min(config.samples_per_pass, config.samples_per_pixel - completed)
                    jobs.append(TileJob(tile, completed, count))
            return jobs

        try:
            if workers == 1:
                _worker_init(scene, camera, integrator, config.width, config.height, config.seed)
                while True:
                    jobs = jobs_for_round()
                    if not jobs:
                        break
                    for job in jobs:
                        _merge_tile(state, _render_tile(job))
                        if checkpoint is not None and time.monotonic() - last_checkpoint >= config.checkpoint_interval:
                            persist()
                        report()
                    if preview is not None:
                        write_png(preview, config.width, config.height, state.pixels(tiles), exposure=exposure, tone_mapper=tone_mapper)
                    persist()
            else:
                pool = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_worker_init,
                    initargs=(scene, camera, integrator, config.width, config.height, config.seed),
                )
                active_futures = []
                try:
                    while True:
                        jobs = jobs_for_round()
                        if not jobs:
                            break
                        active_futures = [pool.submit(_render_tile, job) for job in jobs]
                        for future in as_completed(active_futures):
                            _merge_tile(state, future.result())
                            if checkpoint is not None and time.monotonic() - last_checkpoint >= config.checkpoint_interval:
                                persist()
                            report()
                        if preview is not None:
                            write_png(preview, config.width, config.height, state.pixels(tiles), exposure=exposure, tone_mapper=tone_mapper)
                        persist()
                    pool.shutdown(wait=True)
                except BaseException:
                    for future in active_futures:
                        future.cancel()
                    # Running tiles are bounded by tile_size; do not wait for
                    # every queued tile before the interrupt checkpoint is saved.
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        except KeyboardInterrupt:
            persist()
            if preview is not None:
                write_png(preview, config.width, config.height, state.pixels(tiles), exposure=exposure, tone_mapper=tone_mapper)
            raise

        persist()
        elapsed = time.monotonic() - start_time
        pixels = tuple(state.pixels(tiles))
        return RenderResult(config.width, config.height, pixels, config.samples_per_pixel, elapsed)
