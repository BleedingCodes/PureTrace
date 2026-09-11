"""Binned surface-area-heuristic bounding volume hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import HitRecord, Primitive
from .math3d import AABB, Ray


@dataclass(slots=True)
class BVHNode:
    box: AABB
    left: "BVHNode | None" = None
    right: "BVHNode | None" = None
    objects: tuple[Primitive, ...] = ()

    @classmethod
    def build(
        cls,
        primitives: list[Primitive],
        time0: float = 0.0,
        time1: float = 1.0,
        *,
        leaf_size: int = 4,
        bins: int = 12,
    ) -> "BVHNode":
        if not primitives:
            raise ValueError("Cannot build an empty BVH")
        boxes = {id(obj): obj.bounds(time0, time1) for obj in primitives}

        def recurse(items: list[Primitive]) -> BVHNode:
            node_box = AABB.empty()
            centroid_box = AABB.empty()
            for obj in items:
                box = boxes[id(obj)]
                node_box = node_box.union(box)
                centroid = box.centroid()
                centroid_box = centroid_box.union(AABB(centroid, centroid))
            if len(items) <= leaf_size:
                return cls(node_box, objects=tuple(items))

            best_axis = -1
            best_split = -1
            best_cost = math.inf
            centroid_extent = centroid_box.extent()
            for axis in range(3):
                extent = centroid_extent[axis]
                if extent <= 1.0e-10:
                    continue
                counts = [0] * bins
                bin_boxes: list[AABB | None] = [None] * bins
                low = centroid_box.minimum[axis]
                for obj in items:
                    index = min(bins - 1, int((boxes[id(obj)].centroid()[axis] - low) / extent * bins))
                    counts[index] += 1
                    box = boxes[id(obj)]
                    bin_boxes[index] = box if bin_boxes[index] is None else bin_boxes[index].union(box)

                left_counts = [0] * bins
                right_counts = [0] * bins
                left_boxes: list[AABB | None] = [None] * bins
                right_boxes: list[AABB | None] = [None] * bins
                count = 0
                aggregate: AABB | None = None
                for i in range(bins):
                    count += counts[i]
                    if bin_boxes[i] is not None:
                        aggregate = bin_boxes[i] if aggregate is None else aggregate.union(bin_boxes[i])
                    left_counts[i] = count
                    left_boxes[i] = aggregate
                count = 0
                aggregate = None
                for i in range(bins - 1, -1, -1):
                    count += counts[i]
                    if bin_boxes[i] is not None:
                        aggregate = bin_boxes[i] if aggregate is None else aggregate.union(bin_boxes[i])
                    right_counts[i] = count
                    right_boxes[i] = aggregate
                for split in range(bins - 1):
                    left_box = left_boxes[split]
                    right_box = right_boxes[split + 1]
                    if left_box is None or right_box is None:
                        continue
                    cost = (
                        left_box.surface_area() * left_counts[split]
                        + right_box.surface_area() * right_counts[split + 1]
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_axis = axis
                        best_split = split

            if best_axis < 0:
                axis = centroid_box.longest_axis()
                items.sort(key=lambda obj: boxes[id(obj)].centroid()[axis])
                middle = len(items) // 2
                left_items, right_items = items[:middle], items[middle:]
            else:
                low = centroid_box.minimum[best_axis]
                extent = centroid_extent[best_axis]
                left_items = []
                right_items = []
                for obj in items:
                    index = min(
                        bins - 1,
                        int((boxes[id(obj)].centroid()[best_axis] - low) / extent * bins),
                    )
                    (left_items if index <= best_split else right_items).append(obj)
                if not left_items or not right_items:
                    items.sort(key=lambda obj: boxes[id(obj)].centroid()[best_axis])
                    middle = len(items) // 2
                    left_items, right_items = items[:middle], items[middle:]
            return cls(node_box, recurse(left_items), recurse(right_items))

        return recurse(list(primitives))

    def bounds(self, time0: float = 0.0, time1: float = 1.0) -> AABB:
        return self.box

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord | None:
        if not self.box.hit(ray, t_min, t_max):
            return None
        if self.objects:
            closest = t_max
            result = None
            for obj in self.objects:
                candidate = obj.hit(ray, t_min, closest)
                if candidate is not None:
                    closest = candidate.t
                    result = candidate
            return result
        left_hit = self.left.hit(ray, t_min, t_max) if self.left is not None else None
        if left_hit is not None:
            t_max = left_hit.t
        right_hit = self.right.hit(ray, t_min, t_max) if self.right is not None else None
        return right_hit if right_hit is not None else left_hit

