# tissue/geometry.py

import numpy as np
from dolfin import SubDomain, near

class AxisPlane(SubDomain):
    def __init__(self, axis: int, coordinate: float, tolerance: float = 1e-8):
        super().__init__()
        self.axis = axis
        self.coordinate = coordinate
        self.tolerance = tolerance

    def inside(self, x, on_boundary) -> bool:
        return on_boundary and near(x[self.axis], self.coordinate, self.tolerance)

def point_in_cylinder(point, pos_u, pos_v, radius):
    p = np.array(point)
    u = np.array(pos_u)
    v = np.array(pos_v)
    line = v - u
    line_length_sq = np.dot(line, line)
    if line_length_sq == 0:
        return np.linalg.norm(p - u) <= radius

    t = np.dot(p - u, line) / line_length_sq
    t = np.clip(t, 0.0, 1.0)
    projection = u + t * line
    return np.linalg.norm(p - projection) <= radius