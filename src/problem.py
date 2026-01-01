from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from fenicsx_ii import Average, LinearProblem, assemble_scalar
from mpi4py import MPI

from .domain import Domain1D, Domain3D
from .forms import Parameters, Forms, build_pressure_forms
from .system import abort_on_exception, make_rank_logger, setup_mpi_debug, close_if_possible, collect


@dataclass(frozen=True, slots=True)
class SolverOptions:
    petsc_options_prefix: str = "pressure"
    petsc_options: Mapping[str, Any] = field(
        default_factory=lambda: {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        }
    )


@dataclass(frozen=True, slots=True)
class AssemblyOptions:
    degree_3d: int = 1
    degree_1d: int = 1
    circle_quadrature_degree: int = 20


@dataclass(slots=True)
class PressureSolution:
    tissue_pressure: Any
    network_pressure: Any

    def release(self) -> None:
        self.tissue_pressure = None
        self.network_pressure = None
        collect()


@dataclass(slots=True)
class PressureVelocitySolution(PressureSolution):
    tissue_velocity: Any | None = None

    def release(self) -> None:
        self.tissue_velocity = None
        super().release()


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
            cell_radius: Any | None = None,
            vertex_radius: Any | None = None,
            log: Callable[[str], None] | None = None,
            barrier: Callable[[str], None] | None = None,
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

        self._forms: Forms | None = None
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
                    log=self._log_cb,
                    barrier=self._barrier_cb,
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

            # Postprocessing integrals
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
                network_pressure=p_network
            )

        except Exception as e:
            abort_on_exception(comm, rprint, e)
            raise

    def close(self) -> None:
        close_if_possible(self._linear_problem)
        self._linear_problem = None
        self._forms = None
        collect()

    def __enter__(self) -> "PressureProblem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
