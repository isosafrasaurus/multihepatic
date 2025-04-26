import numpy as np
from dolfin import SubDomain, near, UserExpression, Point, DOLFIN_EPS

class BoundaryPoint(SubDomain):
    def __init__(self, coordinate, tolerance = DOLFIN_EPS):
        super().__init__()
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
    def __init__(self, axis: int, coordinate: float, tolerance: float = DOLFIN_EPS):
        super().__init__()
        self.axis = axis
        self.coordinate = coordinate
        self.tolerance = tolerance

    def inside(self, x, on_boundary):
        return on_boundary and near(x[self.axis], self.coordinate, self.tolerance)