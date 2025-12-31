from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import ufl
from fenicsx_ii import Average, LinearProblem, assemble_scalar
from mpi4py import MPI

from .core import MemoryManager, Parameters, PressureSolution, PressureVelocitySolution, SolverOptions
from .domains import Domain1D, Domain3D
from .forms import AssembledForms, build_assembled_forms


@dataclass(frozen=True, slots=True)
class AssemblyOptions:
    degree_3d: int = 1
    degree_1d: int = 1
    circle_quadrature_degree: int = 20


class PressureProblem:
    def __init__(
            self,
            tissue: Domain3D,
            network: Domain1D,
            *,
            params: Parameters = Parameters(),
            assembly: AssemblyOptions = AssemblyOptions(),
            solver: SolverOptions = SolverOptions(),
            radius_by_tag: dict[int, float] | Any | None = None,
            default_radius: float | None = None,
            cell_radius: fem.Function | None = None,
            vertex_radius: fem.Function | None = None,
    ) -> None:
        self._tissue = tissue
        self._network = network
        self._params = params
        self._assembly = assembly
        self._solver = solver

        self._radius_by_tag = radius_by_tag
        self._default_radius = default_radius
        self._cell_radius = cell_radius
        self._vertex_radius = vertex_radius

        self._assembled: AssembledForms | None = None
        self._linear_problem: Any | None = None

    def solve(self) -> PressureSolution:
        if self._assembled is None:
            self._assembled = build_assembled_forms(
                self._tissue,
                self._network,
                self._params,
                degree_3d=self._assembly.degree_3d,
                degree_1d=self._assembly.degree_1d,
                circle_quadrature_degree=self._assembly.circle_quadrature_degree,
                radius_by_tag=self._radius_by_tag,
                default_radius=self._default_radius,
                cell_radius=self._cell_radius,
                vertex_radius=self._vertex_radius,
            )

        forms = self._assembled

        self._linear_problem = LinearProblem(
            forms.a,
            forms.L,
            bcs=forms.bcs,
            petsc_options_prefix=self._solver.petsc_options_prefix,
            petsc_options=dict(self._solver.petsc_options),
        )

        tissue_pressure, network_pressure = self._linear_problem.solve()
        tissue_pressure.name = "p_tissue"
        network_pressure.name = "p_network"

        # Postprocessing total wall exchange and terminal exchange
        wall_exchange_form = (
                forms.gamma
                * forms.D_perimeter_cell
                * (network_pressure - Average(tissue_pressure, forms.circle_trial, forms.spaces.network_pressure))
                * forms.measures.dx_network
        )
        terminal_exchange_form = (
                (forms.gamma_a / forms.mu_network)
                * forms.D_area_vertex
                * (network_pressure - forms.P_cvp_network)
                * forms.measures.ds_network_outlet
        )

        total_wall_exchange = float(assemble_scalar(wall_exchange_form, op=MPI.SUM))
        total_terminal_exchange = float(assemble_scalar(terminal_exchange_form, op=MPI.SUM))

        return PressureSolution(
            tissue_pressure=tissue_pressure,
            network_pressure=network_pressure,
            cell_radius=forms.cell_radius,
            vertex_radius=forms.vertex_radius,
            total_wall_exchange=total_wall_exchange,
            total_terminal_exchange=total_terminal_exchange,
            metadata={
                "inlet_marker": self._network.inlet_marker,
                "outlet_marker": self._network.outlet_marker,
                "default_radius": forms.default_radius,
            },
        )

    def close(self) -> None:
        MemoryManager.close_if_possible(self._linear_problem)
        self._linear_problem = None
        self._assembled = None
        MemoryManager.collect()

    def __enter__(self) -> "PressureProblem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PressureVelocityProblem(PressureProblem):
    """
    Extends PressureProblem by computing a tissue velocity field:

        v = -(k_t / mu) * grad(p_tissue)
    """

    def __init__(
            self,
            tissue: Domain3D,
            network: Domain1D,
            *,
            params: Parameters = Parameters(),
            assembly: AssemblyOptions = AssemblyOptions(),
            solver: SolverOptions = SolverOptions(),
            radius_by_tag: dict[int, float] | Any | None = None,
            default_radius: float | None = None,
            cell_radius: fem.Function | None = None,
            vertex_radius: fem.Function | None = None,
            velocity_family: str = "DG",
            velocity_degree: int | None = None,
    ) -> None:
        super().__init__(
            tissue,
            network,
            params=params,
            assembly=assembly,
            solver=solver,
            radius_by_tag=radius_by_tag,
            default_radius=default_radius,
            cell_radius=cell_radius,
            vertex_radius=vertex_radius,
        )
        self._velocity_family = velocity_family
        self._velocity_degree = velocity_degree

    def solve(self) -> PressureVelocitySolution:
        pressure_solution = super().solve()

        mesh = self._tissue.mesh
        gdim = mesh.geometry.dim

        vis_degree = mesh.geometry.cmap.degree  # typically 1

        V_vis = fem.functionspace(mesh, ("Lagrange", vis_degree, (gdim,)))
        v_vis = fem.Function(V_vis)
        v_vis.name = "v_tissue"

        factor = float(self._params.k_t / self._params.mu)
        v_expr = -factor * ufl.grad(pressure_solution.tissue_pressure)

        # L2 projection to point-based vector field
        u = ufl.TrialFunction(V_vis)
        w = ufl.TestFunction(V_vis)
        a = ufl.inner(u, w) * ufl.dx
        L = ufl.inner(v_expr, w) * ufl.dx

        proj = fem_petsc.LinearProblem(
            a, L, u=v_vis,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
            petsc_options_prefix="velocity_projection",
        )
        proj.solve()
        v_vis.x.scatter_forward()

        return PressureVelocitySolution(
            tissue_pressure=pressure_solution.tissue_pressure,
            network_pressure=pressure_solution.network_pressure,
            cell_radius=pressure_solution.cell_radius,
            vertex_radius=pressure_solution.vertex_radius,
            total_wall_exchange=pressure_solution.total_wall_exchange,
            total_terminal_exchange=pressure_solution.total_terminal_exchange,
            metadata=pressure_solution.metadata,
            tissue_velocity=v_vis,
        )
