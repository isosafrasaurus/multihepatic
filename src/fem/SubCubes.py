import numpy as np
from dolfin import *
from .Velo import Velo

class SubCubes(Velo):
    def __init__(
        self,
        domain,
        gamma: float,
        gamma_a: float,
        gamma_R: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        p_cvp: float,
        lower_cube_bounds,
        upper_cube_bounds
    ):
        # Call parent constructor (solves PDE, sets self.velocity_expr, self.Omega, etc.)
        super().__init__(domain, gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, p_cvp)
        self.mesh = self.Omega  # for shorter variable name

        # Build separate internal-facet measures for each "cube region"
        self.dS_lower = self._build_interior_measure(lower_cube_bounds)
        self.dS_upper = self._build_interior_measure(upper_cube_bounds)

    def _build_interior_measure(self, cube_bounds):
        low = np.array(cube_bounds[0])
        high = np.array(cube_bounds[1])

        # 1) Mark cells
        cell_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim(), 0)
        for cell in cells(self.mesh):
            midpoint = cell.midpoint().array()
            if all(midpoint >= low) and all(midpoint <= high):
                cell_markers[cell.index()] = 1

        # 2) Mark internal facets between marker=1 and marker=0
        facet_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)
        for facet in facets(self.mesh):
            # Usually in 3D, each facet can have up to 2 adjacent cells
            c = list(facet.entities(3))
            if len(c) == 2:  # Only interior facets have 2 adjacent cells
                c0, c1 = c
                if cell_markers[c0] != cell_markers[c1]:
                    # This facet separates "inside=1" from "outside=0"
                    facet_markers[facet.index()] = 1

        # 3) Build the dS measure
        dS_interior = Measure("dS", domain=self.mesh, subdomain_data=facet_markers)
        return dS_interior

    def compute_lower_cube_flux(self):
        n = FacetNormal(self.mesh)
        return assemble(dot(self.velocity_expr("+"), n("+")) * self.dS_lower(1))

    def compute_lower_cube_flux_in(self):
        n = FacetNormal(self.mesh)
        expr = conditional(dot(self.velocity_expr("+"), n("+")) < 0,
                           dot(self.velocity_expr("+"), n("+")), 
                           0.0)
        return assemble(expr * self.dS_lower(1))

    def compute_lower_cube_flux_out(self):
        n = FacetNormal(self.mesh)
        expr = conditional(dot(self.velocity_expr("+"), n("+")) > 0,
                           dot(self.velocity_expr("+"), n("+")), 
                           0.0)
        return assemble(expr * self.dS_lower(1))

    def compute_upper_cube_flux(self):
        n = FacetNormal(self.mesh)
        return assemble(dot(self.velocity_expr("+"), n("+")) * self.dS_upper(1))

    def compute_upper_cube_flux_in(self):
        n = FacetNormal(self.mesh)
        expr = conditional(dot(self.velocity_expr("+"), n("+")) < 0,
                           dot(self.velocity_expr("+"), n("+")),
                           0.0)
        return assemble(expr * self.dS_upper(1))

    def compute_upper_cube_flux_out(self):
        n = FacetNormal(self.mesh)
        expr = conditional(dot(self.velocity_expr("+"), n("+")) > 0,
                           dot(self.velocity_expr("+"), n("+")),
                           0.0)
        return assemble(expr * self.dS_upper(1))