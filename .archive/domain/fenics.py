from __future__ import annotations

from typing import Any

import numpy as np
from dolfin import DOLFIN_EPS, Point, SubDomain, UserExpression, near


class BoundaryPoint(SubDomain):
    def __init__(self, coordinate, tolerance=DOLFIN_EPS, **kwargs):
        super().__init__(**kwargs)
        self.coordinate = coordinate
        self.tolerance = tolerance

    def inside(self, x, on_boundary: bool):
        return (
                on_boundary
                and near(x[0], self.coordinate[0], self.tolerance)
                and near(x[1], self.coordinate[1], self.tolerance)
                and near(x[2], self.coordinate[2], self.tolerance)
        )


class AxisPlane(SubDomain):
    def __init__(self, axis, coordinate, tolerance=DOLFIN_EPS, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.coordinate = coordinate
        self.tolerance = tolerance

    def inside(self, x, on_boundary):
        return on_boundary and near(x[self.axis], self.coordinate, self.tolerance)


class CubeSubBoundary(SubDomain):
    def __init__(self, corner_1, corner_2):
        super().__init__()
        self.corner_1 = corner_1
        self.corner_2 = corner_2

    def inside(self, x, on_boundary):
        return (
                self.corner_1[0] <= x[0] <= self.corner_2[0]
                and self.corner_1[1] <= x[1] <= self.corner_2[1]
                and self.corner_1[2] <= x[2] <= self.corner_2[2]
        )

class AveragingRadius(UserExpression):
    def __init__(self, tree: Any, G: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.tree = tree
        self.G = G

    def eval(self, value, x):
        p = Point(x[0], x[1], x[2])
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
            return
        edge_ix = self.G.mf[cell]
        edge = list(self.G.edges())[edge_ix]
        value[0] = float(self.G.edges()[edge]["radius"])


class SegmentLength(UserExpression):
    def __init__(self, tree: Any, G: Any, **kwargs: Any):
        super().__init__(**kwargs)
        self.tree = tree
        self.G = G

    def eval(self, value, x):
        p = Point(*x)
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
            return
        edge_ix = self.G.mf[cell]
        u, v = list(self.G.edges())[edge_ix]
        pos_u = np.array(self.G.nodes[u]["pos"], dtype=float)
        pos_v = np.array(self.G.nodes[v]["pos"], dtype=float)
        value[0] = float(np.linalg.norm(pos_v - pos_u))


__all__ = [
    "BoundaryPoint",
    "AxisPlane",
    "CubeSubBoundary",
    "AveragingRadius",
    "SegmentLength",
]
