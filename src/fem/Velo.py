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
        self.domain = domain
        self.k_v = k_v
        self.k_t = k_t
        self.mu = mu

        # Call Sink to compute 3D and 1D pressures (uh3d, uh1d) and to set radius_map, ds measures, etc.
        super().__init__(domain, gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, p_cvp)
        self.fenics_graph = domain.fenics_graph

        # --- Compute 3D Velocity ---
        V_vec = VectorFunctionSpace(domain.Omega, "CG", 1)
        v_trial = TrialFunction(V_vec)
        v_test  = TestFunction(V_vec)
        a_proj  = inner(v_trial, v_test) * dx
        L_proj  = inner(Constant(-self.k_t / self.mu) * grad(self.uh3d), v_test) * dx
        self.velocity = Function(V_vec)
        solve(a_proj == L_proj, self.velocity, solver_parameters={"linear_solver": "mumps"})

        # --- Compute 3D Flux Diagnostics using domain.Omega and associated ds measures ---
        nOmega = FacetNormal(domain.Omega)
        
        # Sink (3D) boundary fluxes
        self.sink_inflow = assemble(conditional(lt(dot(self.velocity, nOmega), 0),
                                                 dot(self.velocity, nOmega), 0.0) * domain.dsOmegaSink)
        self.sink_outflow = assemble(conditional(gt(dot(self.velocity, nOmega), 0),
                                                  dot(self.velocity, nOmega), 0.0) * domain.dsOmegaSink)
        self.sink_net_flow = self.sink_inflow + self.sink_outflow
        self.sink_net_flow_dolfin = assemble(dot(self.velocity, nOmega) * domain.dsOmegaSink)

        # Neumann boundary fluxes
        self.neumann_inflow = assemble(conditional(lt(dot(self.velocity, nOmega), 0),
                                                     dot(self.velocity, nOmega), 0.0) * domain.dsOmegaNeumann)
        self.neumann_outflow = assemble(conditional(gt(dot(self.velocity, nOmega), 0),
                                                      dot(self.velocity, nOmega), 0.0) * domain.dsOmegaNeumann)
        self.neumann_net_flow = self.neumann_inflow + self.neumann_outflow
        self.neumann_net_flow_dolfin = assemble(dot(self.velocity, nOmega) * domain.dsOmegaNeumann)

        # Entire 3D domain boundary fluxes
        self.all_inflow = assemble(conditional(lt(dot(self.velocity, nOmega), 0),
                                                 dot(self.velocity, nOmega), 0.0) * domain.dsOmega)
        self.all_outflow = assemble(conditional(gt(dot(self.velocity, nOmega), 0),
                                                  dot(self.velocity, nOmega), 0.0) * domain.dsOmega)
        self.all_net_flow = self.all_inflow + self.all_outflow
        self.all_net_flow_dolfin = assemble(dot(self.velocity, nOmega) * domain.dsOmega)
        
        # Combined net: sink + neumann (using dolfin integration)
        self.combined_net_flow = self.sink_net_flow_dolfin + self.neumann_net_flow_dolfin

        # --- Compute 1D Lambda Fluxes using fenics_graph.dds and the cross-sectional area ---
        D_area = np.pi * domain.radius_map**2  # radius_map stored in domain
        
        def lambda_flux_piecewise(ds_measure, direction="inflow"):
            flux_expr = (-self.k_v / self.mu) * self.fenics_graph.dds(self.uh1d) * D_area
            if direction == "inflow":
                flux_expr = conditional(lt(flux_expr, 0), flux_expr, 0.0)
            elif direction == "outflow":
                flux_expr = conditional(gt(flux_expr, 0), flux_expr, 0.0)
            return assemble(flux_expr * ds_measure)
        
        self.lambda_inlet_inflow = lambda_flux_piecewise(domain.dsLambdaInlet, direction="inflow")
        self.lambda_inlet_outflow = lambda_flux_piecewise(domain.dsLambdaInlet, direction="outflow")
        self.lambda_inlet_net = self.lambda_inlet_inflow + self.lambda_inlet_outflow
        
        self.lambda_out_inflow = lambda_flux_piecewise(domain.dsLambdaRobin, direction="inflow")
        self.lambda_out_outflow = lambda_flux_piecewise(domain.dsLambdaRobin, direction="outflow")
        self.lambda_out_net = self.lambda_out_inflow + self.lambda_out_outflow

    def print_diagnostics(self, tol=1e-9):
        # Print 3D sink boundary diagnostics
        print("Flow Diagnostics")
        print("--------------------------------------------------")
        print("3D Sink Boundary:")
        print(f"  Inflow               : {self.sink_inflow:.8g}")
        print(f"  Outflow              : {self.sink_outflow:.8g}")
        print(f"  Net Flow (sum)       : {self.sink_net_flow:.8g}")
        print(f"  Net Flow (dolfin)    : {self.sink_net_flow_dolfin:.8g}")
        if abs(self.sink_net_flow - self.sink_net_flow_dolfin) <= tol:
            print("  CHECK PASSED: Sink Net Flow (sum) = Sink Net Flow (dolfin)")
        else:
            print("  CHECK FAILED: Sink Net Flow (sum) ≠ Sink Net Flow (dolfin)")
        print("--------------------------------------------------")
        # Print 3D Neumann boundary diagnostics
        print("3D Neumann Boundary:")
        print(f"  Inflow               : {self.neumann_inflow:.8g}")
        print(f"  Outflow              : {self.neumann_outflow:.8g}")
        print(f"  Net Flow (sum)       : {self.neumann_net_flow:.8g}")
        print(f"  Net Flow (dolfin)    : {self.neumann_net_flow_dolfin:.8g}")
        if abs(self.neumann_net_flow - self.neumann_net_flow_dolfin) <= tol:
            print("  CHECK PASSED: Neumann Net Flow (sum) = Neumann Net Flow (dolfin)")
        else:
            print("  CHECK FAILED: Neumann Net Flow (sum) ≠ Neumann Net Flow (dolfin)")
        print("--------------------------------------------------")
        # Print entire 3D domain boundary diagnostics
        print("Entire 3D Domain Boundary:")
        print(f"  Inflow               : {self.all_inflow:.8g}")
        print(f"  Outflow              : {self.all_outflow:.8g}")
        print(f"  Net Flow (sum)       : {self.all_net_flow:.8g}")
        print(f"  Net Flow (dolfin)    : {self.all_net_flow_dolfin:.8g}")
        if abs(self.all_net_flow - self.all_net_flow_dolfin) <= tol:
            print("  CHECK PASSED: Entire Domain Net Flow (sum) = Entire Domain Net Flow (dolfin)")
        else:
            print("  CHECK FAILED: Entire Domain Net Flow (sum) ≠ Entire Domain Net Flow (dolfin)")
        print("--------------------------------------------------")
        if abs(self.combined_net_flow - self.all_net_flow_dolfin) <= tol:
            print("CHECK PASSED: Sink + Neumann Net Flow (dolfin) equals Entire Domain Net Flow (dolfin)")
        else:
            print("CHECK FAILED: Sink + Neumann Net Flow (dolfin) does not equal Entire Domain Net Flow (dolfin)")
        print("--------------------------------------------------")
        # Print 1D Lambda diagnostics
        print("1D Lambda Inlet (Dirichlet) Boundary:")
        print(f"  Inflow               : {self.lambda_inlet_inflow:.8g}")
        print(f"  Outflow              : {self.lambda_inlet_outflow:.8g}")
        print(f"  Net Flow             : {self.lambda_inlet_net:.8g}")
        print("--------------------------------------------------")
        print("1D Lambda Outlet (Robin) Boundary:")
        print(f"  Inflow               : {self.lambda_out_inflow:.8g}")
        print(f"  Outflow              : {self.lambda_out_outflow:.8g}")
        print(f"  Net Flow             : {self.lambda_out_net:.8g}")
        if abs(self.lambda_inlet_net + self.lambda_out_net) <= tol:
            print("CHECK PASSED: 1D Lambda Inlet Net Flow = 1D Lambda Outlet Net Flow")
        else:
            print("CHECK FAILED: 1D Lambda Inlet Net Flow ≠ 1D Lambda Outlet Net Flow")
        print("--------------------------------------------------")
        if abs((self.lambda_inlet_net + self.lambda_out_net) - self.all_net_flow_dolfin) <= tol:
            print("CHECK PASSED: Global Mass Conservation")
        else:
            print("CHECK FAILED: Global Mass Conservation")
        print("--------------------------------------------------")

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        # Save 1D pressure field via TubeFile (from the graphnics package)
        from graphnics import TubeFile
        TubeFile(self.fenics_graph, os.path.join(directory, "pressure1d.pvd")) << self.uh1d
        # Save 3D pressure field
        File(os.path.join(directory, "pressure3d.pvd")) << self.uh3d
        # Save 3D velocity field
        self.velocity.rename("3D Velocity (m/s)", "3D Velocity Distribution")
        File(os.path.join(directory, "velocity3d.pvd")) << self.velocity
