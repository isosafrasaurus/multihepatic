import os
import numpy as np
from dolfin import SubDomain, MeshFunction, Measure, FacetNormal, conditional, lt, gt, dot, avg, assemble
from .velo import Velo
from tissue import CubeSubBoundary

class SubCubes(Velo):
    def __init__(self, G, Omega, lower_cube_bounds, upper_cube_bounds,
        Lambda_num_nodes_exp = 5,
        Lambda_inlet_nodes = None,
        Omega_sink_subdomain = None,
        order = 2
    ):
        super().__init__(G, Omega, Lambda_num_nodes_exp, Lambda_inlet_nodes, Omega_sink_subdomain, order)
        self.lower_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1, 0)
        self.upper_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1, 0)
        self.lower_cube = CubeSubBoundary(self.lower_cube_bounds[0], self.lower_cube_bounds[1])
        self.upper_cube = CubeSubBoundary(self.upper_cube_bounds[0], self.upper_cube_bounds[1])
        self.lower_cube.mark(self.lower_boundaries, 1)
        self.upper_cube.mark(self.upper_boundaries, 1)
        self.dS_lower = Measure("dS", domain = self.Omega, subdomain_data = self.lower_boundaries)
        self.dS_upper = Measure("dS", domain = self.Omega, subdomain_data = self.upper_boundaries)

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp):
        super().solve(gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp)

    def compute_lower_cube_flux(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(avg(self.velocity), n('-')) * self.dS_lower(1))

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(avg(self.velocity), n('-')) * self.dS_upper(1))

    def compute_lower_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        expr = conditional(
            lt(dot(avg(self.velocity), n('-')), 0),
            dot(avg(self.velocity), n('-')),
            0.0
        )
        return assemble(expr * self.dS_lower(1))

    def compute_lower_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        expr = conditional(
            gt(dot(avg(self.velocity), n('-')), 0),
            dot(avg(self.velocity), n('-')),
            0.0
        )
        return assemble(expr * self.dS_lower(1))

    def compute_upper_cube_flux_in(self):
        n = FacetNormal(self.Omega)
        expr = conditional(
            lt(dot(avg(self.velocity), n('-')), 0),
            dot(avg(self.velocity), n('-')),
            0.0
        )
        return assemble(expr * self.dS_upper(1))

    def compute_upper_cube_flux_out(self):
        n = FacetNormal(self.Omega)
        expr = conditional(
            gt(dot(avg(self.velocity), n('-')), 0),
            dot(avg(self.velocity), n('-')),
            0.0
        )
        return assemble(expr * self.dS_upper(1))