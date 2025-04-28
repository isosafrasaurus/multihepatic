import os, gc, numpy as np
from dolfin import VectorFunctionSpace, Function, TrialFunction, TestFunction, Constant, inner, grad, FacetNormal, conditional, lt, gt, dot, assemble, solve, File
from .sink import Sink

class Velo(Sink):
    def __init__(self, domain, order = 2):
        super().__init__(domain, order)
        self.k_v     = None
        self.k_t     = None
        self.mu      = None
        self.velocity = None

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp):
        super().solve(gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp)
        self.k_t, self.mu = k_t, mu
        V_vec   = VectorFunctionSpace(self.domain.Omega, "CG", 1)
        v_trial = TrialFunction(V_vec)
        v_test  = TestFunction(V_vec)
        a_proj  = inner(v_trial, v_test) * self.domain.dxOmega
        L_proj  = inner(Constant(- self.k_t/self.mu) * grad(self.uh3d), v_test) * self.domain.dxOmega
        velocity = Function(V_vec)
        solve(a_proj == L_proj, velocity, solver_parameters={"linear_solver": "mumps"})
        velocity.rename("3D Velocity (m/s)", "3D Velocity Distribution")
        self.velocity = velocity

    def compute_inflow_sink(self):
        n    = FacetNormal(self.domain.Omega)
        expr = conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.domain.dsOmegaSink)

    def compute_outflow_sink(self):
        n    = FacetNormal(self.domain.Omega)
        expr = conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.domain.dsOmegaSink)

    def compute_net_flow_sink(self):
        return self.compute_inflow_sink() + self.compute_outflow_sink()

    def compute_net_flow_sink_dolfin(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(self.velocity, n) * self.domain.dsOmegaSink)

    def compute_inflow_all(self):
        n    = FacetNormal(self.domain.Omega)
        expr = conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.domain.dsOmega)

    def compute_outflow_all(self):
        n    = FacetNormal(self.domain.Omega)
        expr = conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.domain.dsOmega)

    def compute_net_flow_all(self):
        return self.compute_inflow_all() + self.compute_outflow_all()

    def compute_net_flow_all_dolfin(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(self.velocity, n) * self.domain.dsOmega)

    def compute_net_flow_neumann_dolfin(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(self.velocity, n) * self.domain.dsOmegaNeumann)

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok = True)
        super().save_vtk(directory)
        File(os.path.join(directory, "velocity3d.pvd")) << self.velocity
