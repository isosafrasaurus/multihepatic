from typing import Optional, List
from dolfin import (
    MeshFunction, SubDomain, Measure, near, DOLFIN_EPS, Mesh
)
from graphnics import FenicsGraph
from xii import *
import numpy as np
import importlib
import MeshCreator
import os

importlib.reload(MeshCreator)

class Face(SubDomain):
    pass

class XZeroPlane(Face):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -0.008)

class MeasureMeshCreator(MeshCreator.MeshCreator):
    def __init__(
        self,
        G: FenicsGraph,
        Lambda_inlet: List[int],
        Omega_sink: SubDomain,
        **kwargs
    ):
    
        importlib.reload(MeshCreator)

        super_kwargs = mm_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(G, **super_kwargs)

        
        boundary_Omega = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1, 0)
        Omega_sink.mark(boundary_Omega, 1)

        
        Lambda_boundary_markers = MeshFunction("size_t", self.Lambda, self.Lambda.topology().dim() - 1, 0)

        
        if Lambda_inlet is not None:
            lambda_coordinates = self.Lambda.coordinates()

            for node_id in Lambda_inlet:
                
                if node_id < 0 or node_id >= len(lambda_coordinates):
                    raise ValueError(f"Lambda_inlet node_id {node_id} is out of bounds for the Lambda mesh.")

                pos = lambda_coordinates[node_id]

                class InletEndpoint(SubDomain):
                    def __init__(self, point):
                        super().__init__()
                        self.point = point

                    def inside(self, x, on_boundary):
                        return (
                            on_boundary
                            and near(x[0], self.point[0])
                            and near(x[1], self.point[1])
                            and near(x[2], self.point[2])
                        )

                inlet_subdomain = InletEndpoint(pos)
                inlet_subdomain.mark(Lambda_boundary_markers, 1)

        
        dxOmega = Measure("dx", domain=self.Omega)
        dxLambda = Measure("dx", domain=self.Lambda)

        
        dsOmega = Measure("ds", domain=self.Omega, subdomain_data=boundary_Omega)
        dsOmegaNeumann = dsOmega(0)  
        dsOmegaSink = dsOmega(1)     

        
        dsLambda = Measure("ds", domain=self.Lambda, subdomain_data=Lambda_boundary_markers)
        dsLambdaRobin = dsLambda(0)    
        dsLambdaInlet = dsLambda(1)    

        
        self.boundary_Omega = boundary_Omega
        self.Lambda_boundary_markers = Lambda_boundary_markers
        self.dxOmega = dxOmega
        self.dxLambda = dxLambda
        self.dsOmega = dsOmega
        self.dsLambda = dsLambda
        self.dsOmegaNeumann = dsOmegaNeumann
        self.dsOmegaSink = dsOmegaSink
        self.dsLambdaRobin = dsLambdaRobin
        self.dsLambdaInlet = dsLambdaInlet