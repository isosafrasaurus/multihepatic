import os
import numpy as np
from dolfin import (
    MeshFunction, Measure, FunctionSpace,
    TrialFunction, TestFunction, Constant, inner,
    grad, DirichletBC, File, FacetNormal,
    LUSolver, VectorFunctionSpace, project,
    assemble, dot
)

from .Sink import Sink

class Velo(Sink):
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
        p_cvp: float
    ):
        super().__init__(domain, gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, p_cvp)
        self.domain = domain

        self.velocity_expr = - (self.k_t / self.mu) * grad(self.uh3d)

    def compute_outflow_sink(self):
        """
        Net flux across boundary facets marked with ID=1 
        ('sink'). 
        """
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity_expr, n) * self.dsOmegaSink)

    def compute_outflow_all(self):
        """
        Net flux across *all* boundary facets of the 3D domain.
        """
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity_expr, n) * self.dsOmega)

    def compute_3d_flux_by_marker(self):
        """
        For each boundary marker in 'domain.boundary_Omega',
        compute:
          - net flux   = ∫(v·n) ds
          - inflow     = ∫(v·n) ds, restricted to (v·n < 0)
          - outflow    = ∫(v·n) ds, restricted to (v·n > 0)

        Returns a dict: 
          flux_dict_3d[marker_id] = {
              'net':  net_flux_value,
              'in':   inflow_value (will be negative or zero),
              'out':  outflow_value (will be positive or zero)
          }
        """
        boundary_ids = np.unique(self.domain.boundary_Omega.array())
        n = FacetNormal(self.Omega)
        flux_dict_3d = {}

        for b_id in boundary_ids:
            ds_mark = self.dsOmega(b_id)

            f = dot(self.velocity_expr, n)

            val_net = assemble(f * ds_mark)

            val_in = assemble(0.5 * (f - abs(f)) * ds_mark)

            val_out = assemble(0.5 * (f + abs(f)) * ds_mark)

            flux_dict_3d[b_id] = {
                'net': val_net,
                'in':  val_in,
                'out': val_out
            }

        return flux_dict_3d

    def compute_1d_velocity_expr(self):
        """
        Return a 1D velocity expression along Lambda, 
        e.g. v_1d = -(k_v / mu)*A * grad(uh1d).
        For a constant radius, A = π*(radius_map)^2. If the 
        radius_map is not constant, you might need a more 
        advanced expression to evaluate radius at each edge 
        of the 1D mesh. 
        """

        D_area = np.pi * (self.radius_map**2)
        return - (self.k_v / self.mu) * D_area * grad(self.uh1d)

    def compute_1d_flux_by_marker(self):
        """
        Same idea for the 1D boundary: compute net, inflow, 
        outflow flux for each boundary marker in 'domain.boundary_Lambda'.

        Returns flux_dict_1d[marker_id] = {
            'net':  ...,
            'in':   ...,
            'out':  ...
        }
        """
        boundary_ids_1d = np.unique(self.domain.boundary_Lambda.array())
        velocity_1d_expr = self.compute_1d_velocity_expr()
        n_1d = FacetNormal(self.Lambda)

        flux_dict_1d = {}
        for b_id in boundary_ids_1d:
            ds_mark_1d = self.dsLambda(b_id)

            f1d = dot(velocity_1d_expr, n_1d)
            val_net = assemble(f1d * ds_mark_1d)
            val_in = assemble(0.5 * (f1d - abs(f1d)) * ds_mark_1d)
            val_out = assemble(0.5 * (f1d + abs(f1d)) * ds_mark_1d)

            flux_dict_1d[b_id] = {
                'net': val_net,
                'in':  val_in,
                'out': val_out
            }

        return flux_dict_1d

    def report_flux_diagnostics(self):
        """
        Print a comprehensive flux budget for 3D boundary 
        and 1D boundary: net, in, out for each marker, plus 
        totals. 
        """

        print("\n===== 3D FLUX DIAGNOSTICS =====")
        flux_dict_3d = self.compute_3d_flux_by_marker()
        total_net_3d = 0.0
        total_in_3d = 0.0
        total_out_3d = 0.0

        print("Per-boundary ID fluxes (3D):")
        for b_id, vals in flux_dict_3d.items():
            net = vals['net']
            fin = vals['in']
            fout = vals['out']
            total_net_3d += net
            total_in_3d += fin
            total_out_3d += fout
            print(f"  ID={b_id}: net={net:.6e}, in={fin:.6e}, out={fout:.6e} (m^3/s)")

        print("Totals (3D):")
        print(f"  net   = {total_net_3d:.6e} m^3/s")
        print(f"  in    = {total_in_3d:.6e} m^3/s  (negative portion)")
        print(f"  out   = {total_out_3d:.6e} m^3/s  (positive portion)")

        print("\n===== 1D FLUX DIAGNOSTICS =====")
        flux_dict_1d = self.compute_1d_flux_by_marker()
        total_net_1d = 0.0
        total_in_1d = 0.0
        total_out_1d = 0.0

        print("Per-boundary ID fluxes (1D):")
        for b_id, vals in flux_dict_1d.items():
            net_1d = vals['net']
            fin_1d = vals['in']
            fout_1d = vals['out']
            total_net_1d += net_1d
            total_in_1d += fin_1d
            total_out_1d += fout_1d
            print(f"  ID={b_id}: net={net_1d:.6e}, in={fin_1d:.6e}, out={fout_1d:.6e} (m^3/s)")

        print("Totals (1D):")
        print(f"  net   = {total_net_1d:.6e} m^3/s")
        print(f"  in    = {total_in_1d:.6e} m^3/s  (negative portion)")
        print(f"  out   = {total_out_1d:.6e} m^3/s  (positive portion)")
        print("====================================\n")

    def save_vtk(self, directory: str):
        """
        Write pressure fields (3D, 1D) and velocity field (3D)
        in VTK/PVD format for visualization.
        """
        os.makedirs(directory, exist_ok=True)

        super().save_vtk(directory)

        V_vec = VectorFunctionSpace(self.Omega, "DG", 0)
        velocity_dg = project(self.velocity_expr, V_vec)
        velocity_dg.rename("3D Velocity (m/s)", "3D Velocity Distribution")
        velocity_file = File(os.path.join(directory, "velocity3d.pvd"))
        velocity_file << velocity_dg