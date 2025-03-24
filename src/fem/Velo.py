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
        V_vec = VectorFunctionSpace(self.Omega, "CG", 1)
        v_trial = TrialFunction(V_vec)
        v_test  = TestFunction(V_vec)
        a_proj  = inner(v_trial, v_test)*dx
        L_proj  = inner(Constant(- self.k_t / self.mu)*grad(self.uh3d), v_test)*dx
        self.velocity = Function(V_vec)
        solve(a_proj == L_proj, self.velocity, solver_parameters = {"linear_solver" : "mumps"})
        
        dim = self.Omega.geometric_dimension()
        n = FacetNormal(self.Omega)
        flux_neumann = assemble(dot(self.velocity, n) * self.dsOmegaNeumann)
        n_sum = np.array([assemble(n[i] * self.dsOmegaNeumann) for i in range(dim)])
        n_sum_norm_sq = np.dot(n_sum, n_sum)
        if n_sum_norm_sq > 1e-12:
            c_vec = flux_neumann / n_sum_norm_sq * n_sum
        else:
            c_vec = np.zeros(dim)
        mean_c = Constant(tuple(c_vec))
        mean_func = Function(self.velocity.function_space())
        mean_func.interpolate(mean_c)
        self.velocity.assign(self.velocity - mean_func)

    def compute_inflow_sink(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.dsOmegaSink)

    def compute_outflow_sink(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.dsOmegaSink)

    def compute_net_flow_sink(self):
        return self.compute_inflow_sink() + self.compute_outflow_sink()

    def compute_net_flow_sink_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaSink)

    def compute_net_flow_neumann_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaNeumann)

    def compute_inflow_all(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(lt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.dsOmega)

    def compute_outflow_all(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(gt(dot(self.velocity, n), 0), dot(self.velocity, n), 0.0) * self.dsOmega)

    def compute_net_flow_all(self):
        return self.compute_inflow_all() + self.compute_outflow_all()

    def compute_net_flow_all_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmega)

    def print_diagnostics(self):
        sink_inflow      = self.compute_inflow_sink()
        sink_outflow     = self.compute_outflow_sink()
        sink_net_sum     = self.compute_net_flow_sink()
        sink_net_dolfin  = self.compute_net_flow_sink_dolfin()
        all_inflow       = self.compute_inflow_all()
        all_outflow      = self.compute_outflow_all()
        all_net_sum      = self.compute_net_flow_all()
        all_net_dolfin   = self.compute_net_flow_all_dolfin()
        neumann_net_dolfin = self.compute_net_flow_neumann_dolfin()
        combined_net = sink_net_dolfin + neumann_net_dolfin

        print("Flow Diagnostics")
        print("--------------------------------------------------")
        print("Sink Boundary:")
        print(f"  Inflow               : {sink_inflow:.8g}")
        print(f"  Outflow              : {sink_outflow:.8g}")
        print(f"  Net Flow (sum)       : {sink_net_sum:.8g}")
        print(f"  Net Flow (Dolfin)    : {sink_net_dolfin:.8g}")
        print("  --> The 'Net Flow (sum)' should equal 'Net Flow (Dolfin)'.")
        print("--------------------------------------------------")
        print("Neumann Boundary:")
        print(f"  Net Flow (Dolfin)    : {neumann_net_dolfin:.8g}")
        print("--------------------------------------------------")
        print("Entire Domain Boundary:")
        print(f"  Inflow               : {all_inflow:.8g}")
        print(f"  Outflow              : {all_outflow:.8g}")
        print(f"  Net Flow (sum)       : {all_net_sum:.8g}")
        print(f"  Net Flow (Dolfin)    : {all_net_dolfin:.8g}")
        print("  --> The 'Net Flow (sum)' should equal 'Net Flow (Dolfin)'.")
        print("--------------------------------------------------")
        print("Sum of dsOmegaNeumann and dsOmegaSink (Dolfin):")
        print(f"  Neumann + Sink       : {combined_net:.8g}")
        print("  --> This should match the net flow over the entire domain boundary.")
        print("--------------------------------------------------")

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        super().save_vtk(directory)
        self.velocity.rename("3D Velocity (m/s)", "3D Velocity Distribution")
        velocity_file = File(os.path.join(directory, "velocity3d.pvd"))
        velocity_file << self.velocity

    def save_vtk_inflow(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        n = FacetNormal(self.Omega)
        dim = self.Omega.geometric_dimension()
        zero_vector = Constant(tuple(0.0 for _ in range(dim)))
        inflow_velocity = project(conditional(lt(dot(self.velocity, n), 0), self.velocity, zero_vector), self.velocity.function_space())
        inflow_velocity.rename("3D Inflow Velocity (m/s)", "3D Inflow Velocity Distribution")
        File(os.path.join(directory, "inflow3d.pvd")) << inflow_velocity

    def save_vtk_outflow(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        n = FacetNormal(self.Omega)
        dim = self.Omega.geometric_dimension()
        zero_vector = Constant(tuple(0.0 for _ in range(dim)))
        outflow_velocity = project(conditional(gt(dot(self.velocity, n), 0), self.velocity, zero_vector), self.velocity.function_space())
        outflow_velocity.rename("3D Outflow Velocity (m/s)", "3D Outflow Velocity Distribution")
        File(os.path.join(directory, "outflow3d.pvd")) << outflow_velocity