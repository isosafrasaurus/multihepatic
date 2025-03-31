# tissue/domain_build.py

import warnings, numpy as np
from dolfin import SubDomain, MeshFunction, Measure, facets, near
from .mesh_build import MeshBuild

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

class DomainBuild(MeshBuild):
    def __init__(self, fenics_graph,
        Omega_bounds = None,
        Omega_mesh_voxel_dim = (16, 16, 16),
        Lambda_padding = 0.008,
        Lambda_num_nodes_exp = 5,
        Lambda_inlet = None, 
        Omega_sink = None
    ):
        super().__init__(fenics_graph, Omega_bounds, Omega_mesh_voxel_dim, Lambda_padding, Lambda_num_nodes_exp)
        self.boundary_Omega = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1, 0)
        self.boundary_Lambda = MeshFunction("size_t", self.Lambda, self.Lambda.topology().dim() - 1, 0)

        if Omega_sink is not None:
            Omega_sink.mark(self.boundary_Omega, 1)
        self.dsOmega = Measure("ds", domain=self.Omega, subdomain_data=self.boundary_Omega)

        if Lambda_inlet is not None:
            lambda_coords = self.Lambda.coordinates()
            for node_id in Lambda_inlet:
                if not (0 <= node_id < len(lambda_coords)):
                    raise ValueError(
                        f"Lambda_inlet node_id {node_id} is out of bounds "
                        f"(valid range is [0, {len(lambda_coords)-1}])."
                    )
                coordinate = lambda_coords[node_id]
                inlet_subdomain = BoundaryPoint(coordinate)
                inlet_subdomain.mark(self.boundary_Lambda, 1)

        self.dsLambda = Measure("ds", domain=self.Lambda, subdomain_data=self.boundary_Lambda)
        self.dxOmega = Measure("dx", domain=self.Omega)
        self.dxLambda = Measure("dx", domain=self.Lambda)
        self.dsOmegaNeumann = self.dsOmega(0)
        self.dsOmegaSink    = self.dsOmega(1)
        self.dsLambdaRobin  = self.dsLambda(0)
        self.dsLambdaInlet  = self.dsLambda(1)