import os, numpy as np
from typing import List
from dolfin import SubDomain, MeshFunction, Measure, FacetNormal, conditional, lt, gt, dot, avg, assemble
from .velo import Velo

class CubeSubBoundary(SubDomain):
    def __init__(self, lower: List[float], upper: List[float]):
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
    def __init__(self, domain, lower_cube_bounds: List[List[float]], upper_cube_bounds: List[List[float]], order = 2):
        super().__init__(domain, order)
        self.lower_cube_bounds = lower_cube_bounds
        self.upper_cube_bounds = upper_cube_bounds
        mesh = self.domain.Omega
        facet_dim = mesh.topology().dim() - 1

        self.lower_boundaries = MeshFunction("size_t", mesh, facet_dim)
        self.lower_boundaries.set_all(0)
        self.upper_boundaries = MeshFunction("size_t", mesh, facet_dim)
        self.upper_boundaries.set_all(0)

        self.lower_cube = CubeSubBoundary(self.lower_cube_bounds[0], self.lower_cube_bounds[1])
        self.upper_cube = CubeSubBoundary(self.upper_cube_bounds[0], self.upper_cube_bounds[1])
        self.lower_cube.mark(self.lower_boundaries, 1)
        self.upper_cube.mark(self.upper_boundaries, 1)
        self.dS_lower = Measure("dS", domain=mesh, subdomain_data=self.lower_boundaries)
        self.dS_upper = Measure("dS", domain=mesh, subdomain_data=self.upper_boundaries)

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp):
        super().solve(gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp)

    def compute_lower_cube_flux(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(avg(self.velocity), n('-')) * self.dS_lower(1))

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(avg(self.velocity), n('-')) * self.dS_upper(1))

    def compute_lower_cube_flux_in(self):
        n = FacetNormal(self.domain.Omega)
        expr = conditional(lt(dot(avg(self.velocity), n('-')), 0), dot(avg(self.velocity), n('-')), 0.0)
        return assemble(expr * self.dS_lower(1))

    def compute_lower_cube_flux_out(self):
        n = FacetNormal(self.domain.Omega)
        expr = conditional(gt(dot(avg(self.velocity), n('-')), 0), dot(avg(self.velocity), n('-')), 0.0)
        return assemble(expr * self.dS_lower(1))

    def compute_upper_cube_flux_in(self):
        n = FacetNormal(self.domain.Omega)
        expr = conditional(lt(dot(avg(self.velocity), n('-')), 0), dot(avg(self.velocity), n('-')), 0.0)
        return assemble(expr * self.dS_upper(1))

    def compute_upper_cube_flux_out(self):
        n = FacetNormal(self.domain.Omega)
        expr = conditional(gt(dot(avg(self.velocity), n('-')), 0), dot(avg(self.velocity), n('-')), 0.0)
        return assemble(expr * self.dS_upper(1))
