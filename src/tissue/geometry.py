import numpy as np
from dolfin import SubDomain, near

class BoundaryPoint(SubDomain):
    def __init__(self, coordinate, tolerance: float = 1e-8):
        super().__init__()
        self.coordinate = coordinate
        self.tolerance = tolerance

    def inside(self, x, on_boundary: bool) -> bool:
        return (
            on_boundary
            and near(x[0], self.coordinate[0], self.tolerance)
            and near(x[1], self.coordinate[1], self.tolerance)
            and near(x[2], self.coordinate[2], self.tolerance)
        )

class AxisPlane(SubDomain):
    def __init__(self, axis: int, coordinate: float, tolerance: float = 1e-8):
        super().__init__()
        self.axis = axis
        self.coordinate = coordinate
        self.tolerance = tolerance

    def inside(self, x, on_boundary: bool) -> bool:
        return on_boundary and near(x[self.axis], self.coordinate, self.tolerance)