

import numpy as np
import warnings
from dolfin import MeshFunction, Measure
from .geometry import BoundaryPoint

class MeasureBuild:
    def __init__(self, mesh_build, Lambda_inlet=None, Omega_sink=None):
        self.mesh_build = mesh_build
        self.Omega = mesh_build.Omega
        self.Lambda = mesh_build.Lambda
        self.radius_map = mesh_build.radius_map

        self.Lambda_inlet = Lambda_inlet
        self.Omega_sink = Omega_sink

        
        self.boundary_Omega = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1, 0)
        self.boundary_Lambda = MeshFunction("size_t", self.Lambda, self.Lambda.topology().dim() - 1, 0)

        
        if self.Omega_sink is not None:
            self.Omega_sink.mark(self.boundary_Omega, 1)
        
        self.dsOmega = Measure("ds", domain=self.Omega, subdomain_data=self.boundary_Omega)

        
        if self.Lambda_inlet is not None:
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
        self.dsOmega_neumann = self.dsOmega(0)
        self.dsOmega_sink    = self.dsOmega(1)
        self.dsLambda_robin  = self.dsLambda(0)
        self.dsLambda_inlet  = self.dsLambda(1)