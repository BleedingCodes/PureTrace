# PureTrace

A CPU-only, offline path tracer written in pure Python — no GPU, no native
extensions, no third-party dependencies. Everything is standard library:
geometry, BVH construction, physically based scattering, PNG encoding,
OpenEXR encoding, multiprocessing, and checkpointing.

Built as a systems programming showcase — implementing a production-quality
rendering pipeline entirely within Python's standard library constraints.

---

## Feature Set

| Rendering | Geometry and Assets | Production Workflow |
|---|---|---|
| Monte Carlo global illumination | Wavefront OBJ + MTL importer | Progressive passes |
| Multiple-importance-sampled path tracing | N-gon triangulation | Multi-process CPU rendering |
| Metallic/roughness PBR (GGX) | Smooth vertex normals and UVs | Center-first tile scheduler |
| Diffuse, metal, dielectric glass | Moving spheres | Atomic render checkpoints |
| Reflection and refraction | Quads, triangles, boxes, spheres | Resume interrupted renders |
| Area lights and soft shadows | Procedural lathed meshes | Deterministic per-pixel sampling |
| HDR environment lighting | 12-bin SAH BVH | Tone-mapped PNG output |
| Texture and environment maps | PNG, PPM, Radiance HDR input | Linear half/float OpenEXR output |
| Thin-lens depth of field | JSON scene description | No GPU and no native extension |
| Shutter-time motion blur | Homogeneous participating medium | Single-core or multi-core mode |
| Volumetric fog and anisotropy | Emissive geometry | Built-in reference scenes |

---

## Requirements

- Python 3.11+
- No external packages — zero `pip install` dependencies

---

## Quick Start

```bash
git clone https://github.com/BleedingCodes/PureTrace.git
cd PureTrace
pip install -e .
puretrace examples
puretrace render cornell -W 512 -H 512 -s 256 -j 8 -o cornell.png --exr
```

Without installing:

```bash
PYTHONPATH=src python -m puretrace render spheres -W 640 -H 400 -s 128 -o spheres.png
```

**Built-in scenes:**

| Scene | Description |
|---|---|
| `cornell` | Colored diffuse walls, rough metal, glass, soft ceiling light |
| `spheres` | Chrome, copper, glass, depth of field, and motion blur |
| `glass-chess` | Smooth lathed glass pieces on a reflective checkerboard |
| `caustics` | Glass and polished metal under a compact area light |
| `fog` | Anisotropic participating media and visible light transport |

A first clean render worth waiting for is 256–512 samples per pixel. Glass
caustics converge slowly — 1,000+ samples per pixel is normal for that scene.

---

## Progressive Rendering and Resume

Every CLI render writes an atomic `.ptrchk` checkpoint file beside the output.
Ctrl-C saves the current tiles and preview. Resume with the same settings:

```bash
# Start
puretrace render glass-chess -W 900 -H 675 -s 1200 -j 8 \
  --samples-per-pass 4 -o chess.png

# Resume later
puretrace render glass-chess -W 900 -H 675 -s 1200 -j 8 \
  --samples-per-pass 4 -o chess.png --resume
```

Resolution, tile size, seed, max depth, and scene must match on resume. The
target sample count may be increased.

---

## JSON Scenes

Render the included OBJ example:

```bash
puretrace render scenes/mesh-demo.json -s 128 -o mesh-demo.png --exr
```

Scene structure:

```json
{
  "render": {
    "width": 640,
    "height": 360,
    "samples_per_pixel": 256,
    "max_depth": 12,
    "tile_size": 16,
    "samples_per_pass": 4
  },
  "camera": {
    "look_from": [4, 2, 6],
    "look_at": [0, 1, 0],
    "vertical_fov": 40,
    "aperture": 0.05,
    "focus_distance": 7,
    "shutter_open": 0,
    "shutter_close": 1
  },
  "environment": {
    "hdr": "studio.hdr",
    "strength": 1.2,
    "rotation": 35
  },
  "materials": {
    "glass": {
      "base_color": [0.95, 0.99, 1.0],
      "roughness": 0.01,
      "transmission": 1.0,
      "ior": 1.52
    },
    "metal": {
      "base_color": [0.95, 0.55, 0.2],
      "metallic": 1.0,
      "roughness": 0.16
    }
  },
  "objects": [
    {
      "type": "obj",
      "file": "room.obj",
      "scale": 1.0,
      "rotate_y": 20,
      "translate": [0, 0, 0]
    }
  ]
}
```

**Material fields:** `base_color`, `metallic`, `roughness`, `transmission`,
`ior`, `opacity`, `emission_strength`, `two_sided`. Textures accept solid,
checker, or image types (PNG, PPM, Radiance `.hdr` — no JPEG by design).

**Object types:** `sphere`, `moving_sphere`, `quad`, `triangle`, `box`,
`lathe`, `obj`.

---

## Python API

```python
from puretrace.examples import build_example
from puretrace.renderer import RenderConfig, Renderer

scene, camera = build_example("cornell", aspect=1.0)
renderer = Renderer(RenderConfig(
    width=512,
    height=512,
    samples_per_pixel=256,
    max_depth=12,
    workers=8,
    tile_size=16,
    samples_per_pass=4,
))
image = renderer.render(scene, camera, checkpoint="cornell.ptrchk")
image.save_png("cornell.png", exposure=0.0)
image.save_exr("cornell.exr")
```

---

## Output Formats

- **PNG** — 8-bit sRGB with ACES, Reinhard, or linear tone mapping and exposure control
- **OpenEXR** — uncompressed scanline RGB in linear scene space, half-float default (32-bit float available via API)
- A PNG render can also write EXR with `--exr`; an EXR render can also write PNG with `--png`

---

## Rendering Architecture

1. Scene primitives are converted into a binned surface-area-heuristic BVH
2. The camera samples pixel area, lens aperture, and shutter time
3. The integrator traces BSDF/phase-function paths and explicitly samples area/environment lights
4. Power-heuristic MIS combines light and BSDF sampling
5. Homogeneous free-flight sampling adds volumetric scattering and fog attenuation
6. Independent tiles run in worker processes and return double-precision RGB sums
7. The main process merges tiles, refreshes progressive output, and atomically checkpoints

This is a deliberately readable renderer. Pure Python object traversal is
expensive — use all physical CPU cores, sensible image sizes, and progressive
sampling. The implementation favors correctness, deterministic behavior, and
hackability over SIMD tricks.

---

## Project Structure

```
PureTrace/
├── src/
│   └── puretrace/
│       ├── __main__.py
│       ├── cli.py          # Command-line interface
│       ├── renderer.py     # Core render loop, multiprocessing, checkpointing
│       ├── integrator.py   # Path tracing integrator, MIS, volumetrics
│       ├── geometry.py     # Primitives, intersection math
│       ├── bvh.py          # SAH BVH construction and traversal
│       ├── materials.py    # PBR BSDF (GGX), glass, diffuse
│       ├── textures.py     # Solid, checker, image textures
│       ├── camera.py       # Thin-lens camera, DOF, motion blur
│       ├── scene.py        # Scene graph
│       ├── sceneio.py      # JSON scene loader
│       ├── objloader.py    # Wavefront OBJ/MTL importer
│       ├── output.py       # PNG and OpenEXR output
│       ├── math3d.py       # Vector/matrix math
│       ├── rng.py          # Deterministic per-pixel RNG
│       └── examples.py     # Built-in reference scenes
├── scenes/
│   ├── mesh-demo.json
│   ├── mesh-demo.obj
│   └── mesh-demo.mtl
├── renders/
│   └── cornell-preview.png
├── tests/
│   ├── test_math_geometry.py
│   ├── test_renderer.py
│   └── test_io.py
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Running Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests cover intersection math, BVH closest-hit behavior, UV interpolation,
PNG round-tripping, OpenEXR structure, OBJ triangulation and transforms,
checkpoint loading, and bit-stable resume behavior.

---

## Built by MainbyteLabs

Python tooling for electronics labs, hardware shops, and Linux-based tech teams.

[MainbyteLabs](https://github.com/MR-MainbyteLabs) ·
[LinkedIn](https://linkedin.com/in/michael-rivera-c0ding) ·
mr.mainbytelabs@gmail.com

---

## License

MIT License — see `LICENSE`.
