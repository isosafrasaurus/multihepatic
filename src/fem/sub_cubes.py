import os, numpy as np
from typing import List
from dolfin import (
    SubDomain,
    MeshFunction,
    Measure,
    FacetNormal,
    conditional,
    lt,
    gt,
    dot,
    avg,
    assemble
)
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
    def __init__(
        self,
        G,
        Omega,
        lower_cube_bounds: List[List[float]],
        upper_cube_bounds: List[List[float]],
        Lambda_num_nodes_exp: int = 5,
        Lambda_inlet_nodes=None,
        Omega_sink_subdomain=None,
        order: int = 2
    ):
        # initialize the coupled Sink/Velo system
        super().__init__(
            G,
            Omega,
            Lambda_num_nodes_exp,
            Lambda_inlet_nodes,
            Omega_sink_subdomain,
            order
        )

        # store the cube bounds
        self.lower_cube_bounds = lower_cube_bounds
        self.upper_cube_bounds = upper_cube_bounds

        # build and mark the facet‐wise subdomains
        mesh = self.Omega
        facet_dim = mesh.topology().dim() - 1

        self.lower_boundaries = MeshFunction("size_t", mesh, facet_dim, 0)
        self.upper_boundaries = MeshFunction("size_t", mesh, facet_dim, 0)

        self.lower_cube = CubeSubBoundary(
            self.lower_cube_bounds[0],
            self.lower_cube_bounds[1]
        )
        self.upper_cube = CubeSubBoundary(
            self.upper_cube_bounds[0],
            self.upper_cube_bounds[1]
        )
        self.lower_cube.mark(self.lower_boundaries, 1)
        self.upper_cube.mark(self.upper_boundaries, 1)

        # measures on the interior facets of each cube region
        self.dS_lower = Measure(
            "dS",
            domain=mesh,
            subdomain_data=self.lower_boundaries
        )
        self.dS_upper = Measure(
            "dS",
            domain=mesh,
            subdomain_data=self.upper_boundaries
        )

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp):
        # reuse the parent solve (pressures + velocity projection)
        super().solve(gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp)

    def compute_lower_cube_flux(self):
        n = FacetNormal(self.Omega)
        return assemble(
            dot(avg(self.velocity), n('-')) * self.dS_lower(1)
        )

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.Omega)
        return assemble(
            dot(avg(self.velocity), n('-')) * self.dS_upper(1)
        )

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
