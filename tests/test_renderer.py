import tempfile
from pathlib import Path
import unittest

from puretrace.camera import Camera
from puretrace.geometry import Quad, Sphere
from puretrace.materials import PrincipledMaterial
from puretrace.math3d import Vec3
from puretrace.renderer import RenderConfig, Renderer, load_checkpoint
from puretrace.scene import Scene
from puretrace.textures import EnvironmentMap


def tiny_scene():
    diffuse = PrincipledMaterial(base_color=Vec3(0.7, 0.2, 0.1), roughness=0.6)
    light = PrincipledMaterial(emission=Vec3(1, 1, 1), emission_strength=5)
    scene = Scene(environment=EnvironmentMap(color=Vec3(0.05, 0.07, 0.1)))
    scene.add(
        Sphere(Vec3(0, 0, -1), 0.5, diffuse),
        Quad(Vec3(-1, 2, -2), Vec3(2, 0, 0), Vec3(0, 0, 2), light),
    )
    camera = Camera(Vec3(), Vec3(0, 0, -1), aspect_ratio=1)
    return scene, camera


class RendererTests(unittest.TestCase):
    def test_checkpoint_resume_matches_uninterrupted_render(self):
        scene, camera = tiny_scene()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "render.ptrchk"
            first = Renderer(
                RenderConfig(width=8, height=8, samples_per_pixel=1, max_depth=3, tile_size=4, workers=1)
            )
            first.render(scene, camera, checkpoint=checkpoint)
            state = load_checkpoint(checkpoint)
            self.assertEqual(state.tile_samples, [1, 1, 1, 1])

            resumed = Renderer(
                RenderConfig(width=8, height=8, samples_per_pixel=2, max_depth=3, tile_size=4, workers=1)
            ).render(scene, camera, checkpoint=checkpoint, resume=True)
            fresh = Renderer(
                RenderConfig(width=8, height=8, samples_per_pixel=2, max_depth=3, tile_size=4, workers=1)
            ).render(scene, camera)
            for a, b in zip(resumed.pixels, fresh.pixels):
                self.assertAlmostEqual(a.x, b.x, places=12)
                self.assertAlmostEqual(a.y, b.y, places=12)
                self.assertAlmostEqual(a.z, b.z, places=12)


if __name__ == "__main__":
    unittest.main()

