from __future__ import annotations

from typing import Any, Mapping

import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import ufl
from fenicsx_ii import Average, LinearProblem, assemble_scalar
from mpi4py import MPI

from .config import AssemblyOptions, Parameters, SolverOptions
from .domain import Domain1D, Domain3D
from .forms import PressureForms, build_pressure_forms
from .memory import MemoryManager
from .parallel import abort_on_exception, barrier, rank_print, setup_mpi_debug
from .solutions import PressureSolution, PressureVelocitySolution


class PressureProblem:
    def __init__(
            self,
            tissue: Domain3D,
            network: Domain1D,
            *,
            params: Parameters = Parameters(),
            assembly: AssemblyOptions = AssemblyOptions(),
            solver: SolverOptions = SolverOptions(),
            radius_by_tag: Mapping[int, float] | Any | None = None,
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

        self._forms: PressureForms | None = None
        self._linear_problem: Any | None = None

    @property
    def comm(self) -> MPI.Comm:
        return self._tissue.comm

    def solve(self) -> PressureSolution:
        comm = self.comm
        setup_mpi_debug(comm)
        rprint = rank_print(comm)

        barrier(comm, "PressureProblem.solve:start", rprint)

        try:
            if self._forms is None:
                self._forms = build_pressure_forms(
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

            forms = self._forms
            barrier(comm, "PressureProblem.solve:forms_built", rprint)

            self._linear_problem = LinearProblem(
                forms.a,
                forms.L,
                bcs=forms.bcs,
                petsc_options_prefix=self._solver.petsc_options_prefix,
                petsc_options=dict(self._solver.petsc_options),
            )

            barrier(comm, "PressureProblem.solve:linear_problem_built", rprint)

            p_tissue, p_network = self._linear_problem.solve()
            p_tissue.name = "p_tissue"
            p_network.name = "p_network"

            barrier(comm, "PressureProblem.solve:solved", rprint)

            # Postprocessing integrals
            wall_exchange_form = (
                    forms.gamma
                    * forms.D_perimeter_cell
                    * (p_network - Average(p_tissue, forms.circle_trial, forms.spaces.network_pressure))
                    * forms.measures.dx_network
            )
            terminal_exchange_form = (
                    (forms.gamma_a / forms.mu_network)
                    * forms.D_area_vertex
                    * (p_network - forms.P_cvp_network)
                    * forms.measures.ds_network_outlet
            )

            total_wall_exchange = float(assemble_scalar(wall_exchange_form, op=MPI.SUM))
            total_terminal_exchange = float(assemble_scalar(terminal_exchange_form, op=MPI.SUM))

            barrier(comm, "PressureProblem.solve:postprocess_done", rprint)

            return PressureSolution(
                tissue_pressure=p_tissue,
                network_pressure=p_network,
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

        except Exception as e:
            abort_on_exception(comm, rprint, e)
            raise  # unreachable (Abort), but satisfies type checkers

    def close(self) -> None:
        MemoryManager.close_if_possible(self._linear_problem)
        self._linear_problem = None
        self._forms = None
        MemoryManager.collect()

    def __enter__(self) -> "PressureProblem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PressureVelocityProblem(PressureProblem):
    """
    Extends PressureProblem by computing tissue velocity:
        v = -(k_t/mu) * grad(p_tissue)
    """

    def __init__(
            self,
            tissue: Domain3D,
            network: Domain1D,
            *,
            params: Parameters = Parameters(),
            assembly: AssemblyOptions = AssemblyOptions(),
            solver: SolverOptions = SolverOptions(),
            radius_by_tag: Mapping[int, float] | Any | None = None,
            default_radius: float | None = None,
            cell_radius: fem.Function | None = None,
            vertex_radius: fem.Function | None = None,
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

    def solve(self) -> PressureVelocitySolution:
        comm = self.comm
        setup_mpi_debug(comm)
        rprint = rank_print(comm)

        try:
            pressure_sol = super().solve()
            barrier(comm, "PressureVelocityProblem.solve:pressure_done", rprint)

            mesh = self._tissue.mesh
            gdim = mesh.geometry.dim
            vis_degree = mesh.geometry.cmap.degree  # typical 1

            V_vis = fem.functionspace(mesh, ("Lagrange", vis_degree, (gdim,)))
            v_vis = fem.Function(V_vis)
            v_vis.name = "v_tissue"

            factor = float(self._params.k_t / self._params.mu)
            v_expr = -factor * ufl.grad(pressure_sol.tissue_pressure)

            # L2 projection
            u = ufl.TrialFunction(V_vis)
            w = ufl.TestFunction(V_vis)
            a = ufl.inner(u, w) * ufl.dx
            L = ufl.inner(v_expr, w) * ufl.dx

            proj = fem_petsc.LinearProblem(
                a,
                L,
                u=v_vis,
                petsc_options={
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "ksp_error_if_not_converged": True,
                },
                petsc_options_prefix="velocity_projection",
            )
            proj.solve()
            v_vis.x.scatter_forward()

            barrier(comm, "PressureVelocityProblem.solve:velocity_done", rprint)

            return PressureVelocitySolution(
                tissue_pressure=pressure_sol.tissue_pressure,
                network_pressure=pressure_sol.network_pressure,
                cell_radius=pressure_sol.cell_radius,
                vertex_radius=pressure_sol.vertex_radius,
                total_wall_exchange=pressure_sol.total_wall_exchange,
                total_terminal_exchange=pressure_sol.total_terminal_exchange,
                metadata=pressure_sol.metadata,
                tissue_velocity=v_vis,
            )

        except Exception as e:
            abort_on_exception(comm, rprint, e)
            raise
