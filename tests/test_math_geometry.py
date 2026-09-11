import math
import unittest

from puretrace.bvh import BVHNode
from puretrace.geometry import Sphere, Triangle
from puretrace.materials import PrincipledMaterial
from puretrace.math3d import AABB, Ray, Vec3, reflect, refract


class MathGeometryTests(unittest.TestCase):
    def test_vector_reflection_and_refraction(self):
        reflected = reflect(Vec3(1, -1, 0).normalized(), Vec3(0, 1, 0))
        self.assertGreater(reflected.y, 0)
        transmitted = refract(Vec3(0, -1, 0), Vec3(0, 1, 0), 1 / 1.5)
        self.assertAlmostEqual(transmitted.y, -1.0)

    def test_aabb_parallel_ray(self):
        box = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        self.assertTrue(box.hit(Ray(Vec3(0, 0, 2), Vec3(0, 0, -1)), 0, math.inf))
        self.assertFalse(box.hit(Ray(Vec3(2, 0, 2), Vec3(0, 0, -1)), 0, math.inf))

    def test_bvh_finds_closest(self):
        material = PrincipledMaterial()
        far = Sphere(Vec3(0, 0, -4), 1, material)
        near = Sphere(Vec3(0, 0, -2), 0.5, material)
        side = Sphere(Vec3(4, 0, -2), 0.5, material)
        bvh = BVHNode.build([far, near, side], leaf_size=1)
        hit = bvh.hit(Ray(Vec3(), Vec3(0, 0, -1)), 1e-5, math.inf)
        self.assertIsNotNone(hit)
        self.assertIs(hit.primitive, near)
        self.assertAlmostEqual(hit.t, 1.5)

    def test_triangle_interpolates_uv(self):
        material = PrincipledMaterial()
        triangle = Triangle(Vec3(0, 0, -1), Vec3(1, 0, -1), Vec3(0, 1, -1), material)
        hit = triangle.hit(Ray(Vec3(0.25, 0.25, 0), Vec3(0, 0, -1)), 0, math.inf)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.uv.x, 0.25)
        self.assertAlmostEqual(hit.uv.y, 0.25)


if __name__ == "__main__":
    unittest.main()

