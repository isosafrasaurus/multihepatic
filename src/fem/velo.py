import os
import numpy as np
from dolfin import VectorFunctionSpace, Function, TrialFunction, TestFunction, Constant, inner, grad, FacetNormal, conditional, lt, gt, dot, assemble, solve, File
from .sink import Sink

class Velo(Sink):
    def __init__(self, G, Omega, Lambda_num_nodes_exp = 5, Lambda_inlet_nodes = None, Omega_sink_subdomain = None, order = 2):
        super().__init__(G, Omega, Lambda_num_nodes_exp, Lambda_inlet_nodes, Omega_sink_subdomain, order)
        self.k_t = None
        self.mu = None
        self.velocity = None

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp, directory = None):
        super().solve(gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp, directory = None)

        
        self.k_t = k_t
        self.mu = mu

        
        V_vec = VectorFunctionSpace(self.Omega, "CG", 1)
        v_trial = TrialFunction(V_vec)
        v_test = TestFunction(V_vec)
        a_proj = inner(v_trial, v_test) * self.dxOmega
        L_proj = inner(Constant(-self.k_t/self.mu) * grad(self.uh3d), v_test) * self.dxOmega
        velocity = Function(V_vec)
        solve(a_proj == L_proj, velocity, solver_parameters={"linear_solver": "mumps"})
        velocity.rename("3D Velocity (m/s)", "3D Velocity Distribution")
        self.velocity = velocity
        
        
        if directory is not None:
            os.makedirs(directory, exist_ok = True)
            File(os.path.join(directory, "velocity3d.pvd")) << self.velocity

    def compute_inflow_sink(self):
        n = FacetNormal(self.Omega)
        expr = conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.dsOmegaSink)

    def compute_outflow_sink(self):
        n = FacetNormal(self.Omega)
        expr = conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.dsOmegaSink)

    def compute_net_flow_sink(self):
        return self.compute_inflow_sink() + self.compute_outflow_sink()

    def compute_net_flow_sink_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaSink)

    def compute_inflow_all(self):
        n = FacetNormal(self.Omega)
        expr = conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.dsOmega)

    def compute_outflow_all(self):
        n = FacetNormal(self.Omega)
        expr = conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0)
        return assemble(expr * self.dsOmega)

    def compute_net_flow_all(self):
        return self.compute_inflow_all() + self.compute_outflow_all()

    def compute_net_flow_all_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmega)

    def compute_net_flow_neumann_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaNeumann)