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
        # Remove the on_boundary check to mark all facets of the cube,
        # interior or exterior.
        return (near(x[0], self.lower[0]) or near(x[0], self.upper[0]) or
                near(x[1], self.lower[1]) or near(x[1], self.upper[1]) or
                near(x[2], self.lower[2]) or near(x[2], self.upper[2]))

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
        
        # Obtain mesh coordinates (for any extra geometry setup if needed)
        coords = self.Omega.coordinates()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
            
        self.lower_cube_bounds = lower_cube_bounds
        self.upper_cube_bounds = upper_cube_bounds
        
        # Create MeshFunctions for marking facets.
        self.lower_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.lower_boundaries.set_all(0)
        self.upper_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.upper_boundaries.set_all(0)
        
        self.lower_cube = CubeSubBoundary(self.lower_cube_bounds[0], self.lower_cube_bounds[1])
        self.upper_cube = CubeSubBoundary(self.upper_cube_bounds[0], self.upper_cube_bounds[1])
        self.lower_cube.mark(self.lower_boundaries, 1)
        self.upper_cube.mark(self.upper_boundaries, 1)
        
        # Use dS (capital S) for interior facets.
        self.dS_lower = Measure("dS", domain=self.Omega, subdomain_data=self.lower_boundaries)
        self.dS_upper = Measure("dS", domain=self.Omega, subdomain_data=self.upper_boundaries)
    
    def compute_lower_cube_flux(self):
        n = FacetNormal(self.Omega)
        # Use avg(velocity) and n('+') to restrict the discontinuous geometry.
        flux_lower = assemble(dot(avg(self.velocity), n('+')) * self.dS_lower(1))
        return flux_lower

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.Omega)
        flux_upper = assemble(dot(avg(self.velocity), n('+')) * self.dS_upper(1))
        return flux_upper

    def compute_lower_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        flux_lower_in = assemble(conditional(lt(dot(avg(self.velocity), n('+')), 0),
                                             dot(avg(self.velocity), n('+')), 0.0) * self.dS_lower(1))
        return flux_lower_in

    def compute_lower_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        flux_lower_out = assemble(conditional(gt(dot(avg(self.velocity), n('+')), 0),
                                              dot(avg(self.velocity), n('+')), 0.0) * self.dS_lower(1))
        return flux_lower_out

    def compute_upper_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        flux_upper_in = assemble(conditional(lt(dot(avg(self.velocity), n('+')), 0),
                                             dot(avg(self.velocity), n('+')), 0.0) * self.dS_upper(1))
        return flux_upper_in

    def compute_upper_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        flux_upper_out = assemble(conditional(gt(dot(avg(self.velocity), n('+')), 0),
                                              dot(avg(self.velocity), n('+')), 0.0) * self.dS_upper(1))
        return flux_upper_out
