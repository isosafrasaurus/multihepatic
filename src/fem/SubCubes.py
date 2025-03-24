import tissue
import numpy as np
from dolfin import *
from typing import List
from .Velo import Velo


class CubeSubBoundary(SubDomain):
    def __init__(self, lower, upper):
        
        super().__init__()
        self.lower = lower
        self.upper = upper

    def inside(self, x, on_boundary):
        return (
            self.lower[0] <= x[0] <= self.upper[0] and
            self.lower[1] <= x[1] <= self.upper[1] and
            self.lower[2] <= x[2] <= self.upper[2]
        )

class SubCubes(Velo):
    def __init__(
        self,
        domain: tissue.DomainBuild,
        gamma: float,
        gamma_a: float,
        gamma_R: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        p_cvp: float,
        lower_cube_bounds: List[List[float]],
        upper_cube_bounds: List[List[float]]
    ):
        
        super().__init__(domain, gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, p_cvp)
        
        self.lower_cube_bounds = lower_cube_bounds
        self.upper_cube_bounds = upper_cube_bounds
        
        
        
        self.lower_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.lower_boundaries.set_all(0)
        self.upper_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.upper_boundaries.set_all(0)
        
        
        self.lower_cube = CubeSubBoundary(self.lower_cube_bounds[0], self.lower_cube_bounds[1])
        self.upper_cube = CubeSubBoundary(self.upper_cube_bounds[0], self.upper_cube_bounds[1])
        
        
        self.lower_cube.mark(self.lower_boundaries, 1)
        self.upper_cube.mark(self.upper_boundaries, 1)
        
        
        self.dS_lower = Measure("dS", domain=self.Omega, subdomain_data=self.lower_boundaries)
        self.dS_upper = Measure("dS", domain=self.Omega, subdomain_data=self.upper_boundaries)
    
    def compute_lower_cube_flux(self):
        
        n = FacetNormal(self.Omega)
        flux_lower = assemble(dot(avg(self.velocity), n('-')) * self.dS_lower(1))
        return flux_lower

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.Omega)
        flux_upper = assemble(dot(avg(self.velocity), n('-')) * self.dS_upper(1))
        return flux_upper

    def compute_lower_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        
        flux_lower_in = assemble(conditional(lt(dot(avg(self.velocity), n('-')), 0),
                                             dot(avg(self.velocity), n('-')), 0.0) * self.dS_lower(1))
        return flux_lower_in

    def compute_lower_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        
        flux_lower_out = assemble(conditional(gt(dot(avg(self.velocity), n('-')), 0),
                                              dot(avg(self.velocity), n('-')), 0.0) * self.dS_lower(1))
        return flux_lower_out

    def compute_upper_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        flux_upper_in = assemble(conditional(lt(dot(avg(self.velocity), n('-')), 0),
                                             dot(avg(self.velocity), n('-')), 0.0) * self.dS_upper(1))
        return flux_upper_in

    def compute_upper_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        flux_upper_out = assemble(conditional(gt(dot(avg(self.velocity), n('-')), 0),
                                              dot(avg(self.velocity), n('-')), 0.0) * self.dS_upper(1))
        return flux_upper_out

    def print_cube_diagnostics(self):
        
        print(f"Total Sink Flow (m^3/s): {self.compute_net_flow_sink()}")
        print(f"Total Flow (m^3/s): {self.compute_net_flow_all()}")
        print("--------------------------------------------------")
        print(f"Net flux through lower: {self.compute_lower_cube_flux()}")
        print(f"Inflow through lower: {self.compute_lower_cube_flux_in()}")
        print(f"Outflow through lower: {self.compute_lower_cube_flux_out()}")
        print("--------------------------------------------------")
        print(f"Net flux through upper: {self.compute_upper_cube_flux()}")
        print(f"Inflow through upper: {self.compute_upper_cube_flux_in()}")
        print(f"Outflow through upper: {self.compute_upper_cube_flux_out()}")