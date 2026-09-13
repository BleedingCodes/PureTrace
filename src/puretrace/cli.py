"""Command-line interface for PureTrace."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from .examples import EXAMPLES, build_example
from .renderer import RenderConfig, Renderer
from .sceneio import load_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puretrace",
        description="Pure-Python CPU path tracer with PBR, BVH, HDR, volumes, and resumable tiles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("examples", help="list built-in reference scenes")
    render = subparsers.add_parser("render", help="render a JSON scene or built-in example")
    render.add_argument("scene", help="scene.json or one of the built-in example names")
    render.add_argument("-o", "--output", default="render.png", help="output .png or .exr")
    render.add_argument("--exr", action="store_true", help="also write a linear .exr beside PNG output")
    render.add_argument("--png", action="store_true", help="also write a tone-mapped .png beside EXR output")
    render.add_argument("-W", "--width", type=int)
    render.add_argument("-H", "--height", type=int)
    render.add_argument("-s", "--samples", type=int)
    render.add_argument("-d", "--depth", type=int)
    render.add_argument("--tile-size", type=int)
    render.add_argument("-j", "--workers", type=int)
    render.add_argument("--samples-per-pass", type=int)
    render.add_argument("--seed", type=int)
    render.add_argument("--sample-clamp", type=float)
    render.add_argument("--checkpoint", help="checkpoint path (defaults beside output)")
    render.add_argument("--resume", action="store_true", help="resume the matching checkpoint")
    render.add_argument("--exposure", type=float, default=0.0, help="PNG exposure in stops")
    render.add_argument("--tone-map", choices=("aces", "reinhard", "linear"), default="aces")
    return parser


def _setting(cli_value, settings: dict, key: str, default):
    return cli_value if cli_value is not None else settings.get(key, default)


def _render(args: argparse.Namespace) -> int:
    if args.scene in EXAMPLES:
        settings = {}
        width = args.width or 640
        height = args.height or (640 if args.scene == "cornell" else 480)
        scene, camera = build_example(args.scene, width / height)
    else:
        loaded = load_scene(args.scene)
        settings = loaded.render
        width = _setting(args.width, settings, "width", 640)
        height = _setting(args.height, settings, "height", 360)
        scene, camera = loaded.scene, loaded.camera
        if camera.aspect_ratio != width / height:
            # Rebuild after CLI resolution overrides so projection stays correct.
            from .camera import Camera

            camera = Camera(
                camera.look_from,
                camera.look_at,
                camera.up,
                camera.vertical_fov,
                width / height,
                camera.aperture,
                camera.focus_distance,
                camera.shutter_open,
                camera.shutter_close,
            )
    config = RenderConfig(
        width=int(width),
        height=int(height),
        samples_per_pixel=int(_setting(args.samples, settings, "samples_per_pixel", 128)),
        max_depth=int(_setting(args.depth, settings, "max_depth", 12)),
        tile_size=int(_setting(args.tile_size, settings, "tile_size", 16)),
        workers=int(_setting(args.workers, settings, "workers", 0)),
        samples_per_pass=int(_setting(args.samples_per_pass, settings, "samples_per_pass", 1)),
        seed=int(_setting(args.seed, settings, "seed", 1337)),
        sample_clamp=float(_setting(args.sample_clamp, settings, "sample_clamp", 0.0)),
    )
    output = Path(args.output)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output.with_suffix(".ptrchk")
    preview = output if output.suffix.lower() == ".png" else output.with_suffix(".preview.png")
    last_print = 0.0

    def progress(fraction: float, spp: float, elapsed: float) -> None:
        nonlocal last_print
        now = time.monotonic()
        if now - last_print < 0.25 and fraction < 1.0:
            return
        last_print = now
        rate = spp / elapsed if elapsed > 0.0 else 0.0
        remaining = (config.samples_per_pixel - spp) / rate if rate > 0.0 else 0.0
        print(
            f"\r{fraction * 100:6.2f}%  {spp:7.2f}/{config.samples_per_pixel} spp  "
            f"elapsed {elapsed:7.1f}s  ETA {remaining:7.1f}s",
            end="",
            flush=True,
        )

    print(
        f"PureTrace: {config.width}x{config.height}, {config.samples_per_pixel} spp, "
        f"depth {config.max_depth}, workers {config.workers or 'auto'}"
    )
    renderer = Renderer(config)
    try:
        result = renderer.render(
            scene,
            camera,
            checkpoint=checkpoint,
            resume=args.resume,
            preview=preview,
            exposure=args.exposure,
            tone_mapper=args.tone_map,
            progress=progress,
        )
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume with --resume (checkpoint: {checkpoint})")
        return 130
    print()
    suffix = output.suffix.lower()
    if suffix == ".png":
        result.save_png(output, exposure=args.exposure, tone_mapper=args.tone_map)
        if args.exr:
            result.save_exr(output.with_suffix(".exr"))
    elif suffix == ".exr":
        result.save_exr(output)
        if args.png:
            result.save_png(output.with_suffix(".png"), exposure=args.exposure, tone_mapper=args.tone_map)
    else:
        raise ValueError("Output filename must end in .png or .exr")
    print(f"Saved {output} in {result.elapsed_seconds:.1f}s; checkpoint {checkpoint}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "examples":
        print("\n".join(EXAMPLES))
        return 0
    try:
        return _render(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
