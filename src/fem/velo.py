import os, tissue
import numpy as np
from dolfin import *
from .sink_kvc import Sink

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
        P_cvp: float
    ):     
        super().__init__(domain, gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, P_cvp)
        
        self.k_v = k_v
        self.k_t = k_t
        self.mu = mu

        V_vec = VectorFunctionSpace(domain.Omega, "CG", 1)
        v_trial = TrialFunction(V_vec)
        v_test  = TestFunction(V_vec)
        a_proj  = inner(v_trial, v_test) * dx
        L_proj  = inner(Constant(- self.k_t / self.mu) * grad(self.uh3d), v_test) * dx
        self.velocity = Function(V_vec)
        solve(a_proj == L_proj, self.velocity, solver_parameters = {"linear_solver": "mumps"})
        self.velocity.rename("3D Velocity (m/s)", "3D Velocity Distribution")

    def compute_inflow_sink(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.domain.dsOmegaSink)

    def compute_outflow_sink(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.domain.dsOmegaSink)

    def compute_net_flow_sink(self):
        return self.compute_inflow_sink() + self.compute_outflow_sink()

    def compute_net_flow_sink_dolfin(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(self.velocity, n) * self.domain.dsOmegaSink)

    def compute_inflow_all(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.domain.dsOmega)

    def compute_outflow_all(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.domain.dsOmega)

    def compute_net_flow_all(self):
        return self.compute_inflow_all() + self.compute_outflow_all()

    def compute_net_flow_all_dolfin(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(self.velocity, n) * self.domain.dsOmega)

    def compute_net_flow_neumann_dolfin(self):
        n = FacetNormal(self.domain.Omega)
        return assemble(dot(self.velocity, n) * self.domain.dsOmegaNeumann)

    def _compute_lambda_flux(self, ds_measure):
        D_area = np.pi * self.radius * self.radius
        flux_expr = (- self.k_v / self.mu) * self.domain.fenics_graph.dds(self.uh1d) * D_area
        return assemble(flux_expr * ds_measure)

    def _compute_lambda_flux_piecewise(self, ds_measure, direction="inflow"):
        D_area = np.pi * self.radius * self.radius
        flux_expr = (- self.k_v / self.mu) * self.domain.fenics_graph.dds(self.uh1d) * D_area
        if direction == "inflow":
            flux_expr = conditional(lt(flux_expr, 0), flux_expr, 0.0)
        elif direction == "outflow":
            flux_expr = conditional(gt(flux_expr, 0), flux_expr, 0.0)
        return assemble(flux_expr * ds_measure)

    def compute_lambda_inlet_inflow(self):
        return self._compute_lambda_flux_piecewise(self.domain.dsLambdaInlet, direction="inflow")

    def compute_lambda_inlet_outflow(self):
        return self._compute_lambda_flux_piecewise(self.domain.dsLambdaInlet, direction="outflow")

    def compute_lambda_inlet_net(self):
        return self.compute_lambda_inlet_inflow() + self.compute_lambda_inlet_outflow()

    def compute_lambda_out_inflow(self):
        return self._compute_lambda_flux_piecewise(self.domain.dsLambdaRobin, direction="inflow")

    def compute_lambda_out_outflow(self):
        return self._compute_lambda_flux_piecewise(self.domain.dsLambdaRobin, direction="outflow")

    def compute_lambda_out_net(self):
        return self.compute_lambda_out_inflow() + self.compute_lambda_out_outflow()

    def save_vtk(self, directory):
        os.makedirs(directory, exist_ok=True)
        super().save_vtk(directory)
        File(os.path.join(directory, "velocity3d.pvd")) << self.velocity
