from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from fenicsx_ii import Average, LinearProblem, assemble_scalar
from mpi4py import MPI

from ..config import AssemblyOptions, Parameters, SolverOptions
from ..domain import Domain1D, Domain3D
from ..forms import PressureForms, build_pressure_forms
from ..memory import MemoryManager
from ..parallel import abort_on_exception, make_rank_logger, setup_mpi_debug
from ..solutions import PressureSolution


class PressureProblem:
    """
    Library solver: build forms + solve + postprocess exchange integrals.
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
            cell_radius: Any | None = None,
            vertex_radius: Any | None = None,
            log: Callable[[str], None] | None = None,  # ✅ ADDED
            barrier: Callable[[str], None] | None = None,  # ✅ ADDED
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

        self._log_cb = log
        self._barrier_cb = barrier

        self._forms: PressureForms | None = None
        self._linear_problem: Any | None = None

    @property
    def comm(self) -> MPI.Comm:
        return self._tissue.comm

    def _log(self, msg: str) -> None:
        if self._log_cb is not None:
            self._log_cb(msg)

    def _bar(self, tag: str) -> None:
        if self._barrier_cb is not None:
            self._barrier_cb(tag)

    def solve(self) -> PressureSolution:
        comm = self.comm
        setup_mpi_debug(comm)

        # Ensure we always have *some* logger for fatal exceptions.
        rprint = self._log_cb or make_rank_logger(comm)

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
                    log=self._log_cb,  # ✅ forwarded so prints stay
                    barrier=self._barrier_cb,  # ✅ forwarded so barrier tags stay
                )

            forms = self._forms

            self._log("ABOUT TO CONSTRUCT LinearProblem(...) (this may trigger JIT/assembly)")
            t0 = time.time()
            self._linear_problem = LinearProblem(
                forms.a,
                forms.L,
                bcs=forms.bcs,
                petsc_options_prefix=self._solver.petsc_options_prefix,
                petsc_options=dict(self._solver.petsc_options),
            )
            self._log(f"LinearProblem constructed in {time.time() - t0:.3f}s")
            self._bar("after LinearProblem ctor")

            self._log("ABOUT TO CALL problem.solve()")
            t0 = time.time()
            p_tissue, p_network = self._linear_problem.solve()
            self._log(f"problem.solve() returned in {time.time() - t0:.3f}s")
            self._bar("after problem.solve")

            # Postprocessing integrals (same as original)
            self._log("Assembling exchange integrals...")
            t0 = time.time()

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

            self._log(
                f"exchange assembled in {time.time() - t0:.3f}s; "
                f"Q_wall={total_wall_exchange} Q_term={total_terminal_exchange}"
            )
            self._bar("after assemble_scalar")

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
            raise

    def close(self) -> None:
        MemoryManager.close_if_possible(self._linear_problem)
        self._linear_problem = None
        self._forms = None
        MemoryManager.collect()

    def __enter__(self) -> "PressureProblem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
