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
        solve(a_proj == L_proj, self.velocity, solver_parameters={"linear_solver": "mumps"})

    def compute_inflow_sink(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(lt(dot(self.velocity, n), 0),
                                    dot(self.velocity, n), 0.0) * self.dsOmegaSink)

    def compute_outflow_sink(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(gt(dot(self.velocity, n), 0),
                                    dot(self.velocity, n), 0.0) * self.dsOmegaSink)

    def compute_net_flow_sink(self):
        return self.compute_inflow_sink() + self.compute_outflow_sink()

    def compute_net_flow_sink_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaSink)

    def compute_inflow_all(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(lt(dot(self.velocity, n), 0),
                                    dot(self.velocity, n), 0.0) * self.dsOmega)

    def compute_outflow_all(self):
        n = FacetNormal(self.Omega)
        return assemble(conditional(gt(dot(self.velocity, n), 0),
                                    dot(self.velocity, n), 0.0) * self.dsOmega)

    def compute_net_flow_all(self):
        return self.compute_inflow_all() + self.compute_outflow_all()

    def compute_net_flow_all_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmega)

    def compute_net_flow_neumann_dolfin(self):
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n) * self.dsOmegaNeumann)

    def _compute_lambda_flux(self, ds_measure):
        D_area = np.pi * self.radius_map**2
        n_lambda = FacetNormal(self.Lambda)
        flux_expr = (- self.k_v / self.mu) * dot(grad(self.uh1d), n_lambda) * D_area
        return assemble(flux_expr * ds_measure)

    def _compute_lambda_flux_piecewise(self, ds_measure, direction="inflow"):
        D_area = np.pi * self.radius_map**2
        n_lambda = FacetNormal(self.Lambda)

        flux_expr = (- self.k_v / self.mu) * dot(grad(self.uh1d), n_lambda) * D_area
        if direction == "inflow":
            flux_expr = conditional(lt(flux_expr, 0), flux_expr, 0.0)
        elif direction == "outflow":
            flux_expr = conditional(gt(flux_expr, 0), flux_expr, 0.0)
        return assemble(flux_expr * ds_measure)

    def compute_lambda_inlet_inflow(self):
        return self._compute_lambda_flux_piecewise(self.dsLambdaInlet, direction="inflow")

    def compute_lambda_inlet_outflow(self):
        return self._compute_lambda_flux_piecewise(self.dsLambdaInlet, direction="outflow")

    def compute_lambda_inlet_net(self):
        return self.compute_lambda_inlet_inflow() + self.compute_lambda_inlet_outflow()

    def compute_lambda_out_inflow(self):
        return self._compute_lambda_flux_piecewise(self.dsLambdaRobin, direction="inflow")

    def compute_lambda_out_outflow(self):
        return self._compute_lambda_flux_piecewise(self.dsLambdaRobin, direction="outflow")

    def compute_lambda_out_net(self):
        return self.compute_lambda_out_inflow() + self.compute_lambda_out_outflow()

    def print_diagnostics(self, tol=1e-9):

        sink_inflow      = self.compute_inflow_sink()
        sink_outflow     = self.compute_outflow_sink()
        sink_net_sum     = self.compute_net_flow_sink()
        sink_net_dolfin  = self.compute_net_flow_sink_dolfin()

        neumann_inflow   = assemble(conditional(lt(dot(self.velocity, FacetNormal(self.Omega)), 0),
                                                  dot(self.velocity, FacetNormal(self.Omega)), 0.0) * self.dsOmegaNeumann)
        neumann_outflow  = assemble(conditional(gt(dot(self.velocity, FacetNormal(self.Omega)), 0),
                                                  dot(self.velocity, FacetNormal(self.Omega)), 0.0) * self.dsOmegaNeumann)
        neumann_net_sum  = neumann_inflow + neumann_outflow
        neumann_net_dolfin = self.compute_net_flow_neumann_dolfin()

        all_inflow       = self.compute_inflow_all()
        all_outflow      = self.compute_outflow_all()
        all_net_sum      = self.compute_net_flow_all()
        all_net_dolfin   = self.compute_net_flow_all_dolfin()

        combined_net     = sink_net_dolfin + neumann_net_dolfin

        lambda_inlet_inflow  = self.compute_lambda_inlet_inflow()
        lambda_inlet_outflow = self.compute_lambda_inlet_outflow()
        lambda_inlet_net     = self.compute_lambda_inlet_net()

        lambda_out_inflow    = self.compute_lambda_out_inflow()
        lambda_out_outflow   = self.compute_lambda_out_outflow()
        lambda_out_net       = self.compute_lambda_out_net()

        print("Flow Diagnostics")
        print("--------------------------------------------------")
        print("3D Sink Boundary:")
        print(f"  Inflow               : {sink_inflow:.8g}")
        print(f"  Outflow              : {sink_outflow:.8g}")
        print(f"  Net Flow (sum)       : {sink_net_sum:.8g}")
        print(f"  Net Flow (dolfin)    : {sink_net_dolfin:.8g}")
        if abs(sink_net_sum - sink_net_dolfin) <= tol:
            print("  CHECK PASSED: Sink Net Flow (sum) = Sink Net Flow (dolfin)")
        else:
            print("  CHECK FAILED: Sink Net Flow (sum) ≠ Sink Net Flow (dolfin)")
        print("--------------------------------------------------")
        print("3D Neumann Boundary:")
        print(f"  Inflow               : {neumann_inflow:.8g}")
        print(f"  Outflow              : {neumann_outflow:.8g}")
        print(f"  Net Flow (sum)       : {neumann_net_sum:.8g}")
        print(f"  Net Flow (dolfin)    : {neumann_net_dolfin:.8g}")
        if abs(neumann_net_sum - neumann_net_dolfin) <= tol:
            print("  CHECK PASSED: Neumann Net Flow (sum) = Neumann Net Flow (dolfin)")
        else:
            print("  CHECK FAILED: Neumann Net Flow (sum) ≠ Neumann Net Flow (dolfin)")
        print("--------------------------------------------------")
        print("Entire 3D Domain Boundary:")
        print(f"  Inflow               : {all_inflow:.8g}")
        print(f"  Outflow              : {all_outflow:.8g}")
        print(f"  Net Flow (sum)       : {all_net_sum:.8g}")
        print(f"  Net Flow (dolfin)    : {all_net_dolfin:.8g}")
        if abs(all_net_sum - all_net_dolfin) <= tol:
            print("  CHECK PASSED: Entire Domain Net Flow (sum) = Entire Domain Net Flow (dolfin)")
        else:
            print("  CHECK FAILED: Entire Domain Net Flow (sum) ≠ Entire Domain Net Flow (dolfin)")
        print("--------------------------------------------------")
        if abs(combined_net - all_net_dolfin) <= tol:
            print("CHECK PASSED: Sink + Neumann Net Flow (dolfin) equals Entire Domain Net Flow (dolfin)")
        else:
            print("CHECK FAILED: Sink + Neumann Net Flow (dolfin) does not equal Entire Domain Net Flow (dolfin)")
        print("--------------------------------------------------")
        print("1D Lambda Inlet (Dirichlet) Boundary:")
        print(f"  Inflow               : {lambda_inlet_inflow:.8g}")
        print(f"  Outflow              : {lambda_inlet_outflow:.8g}")
        print(f"  Net Flow             : {lambda_inlet_net:.8g}")
        print("--------------------------------------------------")
        print("1D Lambda Outlet (Robin) Boundary:")
        print(f"  Inflow               : {lambda_out_inflow:.8g}")
        print(f"  Outflow              : {lambda_out_outflow:.8g}")
        print(f"  Net Flow             : {lambda_out_net:.8g}")
        if abs(lambda_inlet_net - lambda_out_net) <= tol:
            print("CHECK PASSED: 1D Lambda Inlet Net Flow = 1D Lambda Outlet Net Flow")
        else:
            print("CHECK FAILED: 1D Lambda Inlet Net Flow ≠ 1D Lambda Outlet Net Flow")
        print("--------------------------------------------------")

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        super().save_vtk(directory)
        self.velocity.rename("3D Velocity (m/s)", "3D Velocity Distribution")
        velocity_file = File(os.path.join(directory, "velocity3d.pvd"))
        velocity_file << self.velocity