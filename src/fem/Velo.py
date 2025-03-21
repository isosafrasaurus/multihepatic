import os, tissue
import numpy as np
from dolfin import *
from .Sink import Sink

class Velo(Sink):
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
        p_cvp: float
    ):
        super().__init__(domain, gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, p_cvp)
        self.velocity_expr = - (self.k_t / self.mu) * grad(self.uh3d)

    def compute_outflow_sink(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity_expr, n) * self.dsOmegaSink)

    def compute_outflow_all(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity_expr, n) * self.dsOmega)

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok=True)

        super().save_vtk(directory)
        V_vec = VectorFunctionSpace(self.Omega, "DG", 0)

        velocity_dg = project(self.velocity_expr, V_vec)
        velocity_dg.rename("3D Velocity (m/s)", "3D Velocity Distribution")

        velocity_file = File(os.path.join(directory, "velocity3d.pvd"))
        velocity_file << velocity_dg