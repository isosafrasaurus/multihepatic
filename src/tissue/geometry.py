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
        self.tolerance = tolerance):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[self.axis], self.coordinate, self.tolerance)

class AveragingRadius(UserExpression):
    def __init__(self, domain, **kwargs):
        self.G = domain.fenics_graph
        self.tree = domain.Lambda.bounding_box_tree()
        self.tree.build(domain.Lambda)
        super().__init__(**kwargs)

    def eval(self, value, x):
        p = Point(x[0], x[1], x[2])
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
        else:
            edge_ix = self.G.mf[cell]
            edge = list(self.G.edges())[edge_ix]
            value[0] = self.G.edges()[edge]['radius']