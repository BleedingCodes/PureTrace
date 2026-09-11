import struct
import tempfile
from pathlib import Path
import unittest

from puretrace.materials import PrincipledMaterial
from puretrace.math3d import Vec3
from puretrace.objloader import Transform, load_obj
from puretrace.output import write_exr, write_png
from puretrace.textures import load_png


class IOTests(unittest.TestCase):
    def test_png_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.png"
            pixels = [Vec3(0.0, 0.25, 1.0), Vec3(2.0, 0.5, 0.1)]
            write_png(path, 2, 1, pixels)
            loaded = load_png(path)
            self.assertEqual((loaded.width, loaded.height), (2, 1))
            self.assertEqual(len(loaded.pixels), 2)

    def test_openexr_header_and_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.exr"
            write_exr(path, 2, 2, [Vec3(1, 2, 3)] * 4)
            data = path.read_bytes()
            magic, version = struct.unpack_from("<II", data)
            self.assertEqual(magic, 20000630)
            self.assertEqual(version, 2)
            self.assertIn(b"channels\x00chlist\x00", data)

    def test_obj_triangulation_negative_indices_and_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quad.obj"
            path.write_text(
                "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
                "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                "f -4/-4 -3/-3 -2/-2 -1/-1\n",
                encoding="utf-8",
            )
            result = load_obj(
                path,
                transform=Transform(scale=Vec3(2, 2, 2), translate=Vec3(1, 0, 0)),
                material_override=PrincipledMaterial(),
            )
            self.assertEqual(len(result.triangles), 2)
            self.assertEqual(result.triangles[0].p0, Vec3(1, 0, 0))
            self.assertEqual(result.triangles[0].p1, Vec3(3, 0, 0))


if __name__ == "__main__":
    unittest.main()

