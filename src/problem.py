from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import dolfinx.fem as fem
import numpy as np
import ufl
from dolfinx import default_scalar_type
from fenicsx_ii import Average, Circle, LinearProblem, assemble_scalar
from mpi4py import MPI

from .domain import Domain1D, Domain3D
from .radius import (
    build_boundary_vertex_radius_field,
    build_cell_radius_field,
    make_cell_radius_callable,
)
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


@dataclass(frozen=True, slots=True)
class Parameters:
    gamma: float = 3.6145827741262347e-05
    gamma_a: float = 8.225197366649115e-08
    gamma_R: float = 8.620057937882969e-08
    mu: float = 1.0e-3
    k_t: float = 1.0e-10
    P_in: float = 100.0 * 133.322
    P_cvp: float = 1.0 * 133.322


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
        self._default_radius = 1.0 if default_radius is None else float(default_radius)
        self._cell_radius_user = cell_radius
        self._vertex_radius_user = vertex_radius

        self._log_cb = log
        self._barrier_cb = barrier

        self._built = False

        self._V3: Any | None = None
        self._V1: Any | None = None
        self._W: Any | None = None

        self._dx_tissue: Any | None = None
        self._dx_network: Any | None = None
        self._ds_tissue: Any | None = None
        self._ds_network_outlet: Any | None = None

        self._cell_radius: Any | None = None
        self._vertex_radius: Any | None = None
        self._radius_per_cell: np.ndarray | None = None

        self._circle_trial: Any | None = None
        self._circle_test: Any | None = None

        self._a: Any | None = None
        self._L: Any | None = None
        self._bcs: list[Any] | None = None

        self._gamma: Any | None = None
        self._gamma_a: Any | None = None
        self._mu_network: Any | None = None
        self._P_cvp_network: Any | None = None
        self._D_perimeter_cell: Any | None = None
        self._D_area_vertex: Any | None = None

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

    def _build_if_needed(self) -> None:
        if self._built:
            return

        tissue = self._tissue
        network = self._network
        params = self._params

        degree_3d = self._assembly.degree_3d
        degree_1d = self._assembly.degree_1d
        circle_qdeg = self._assembly.circle_quadrature_degree
        default_radius = float(self._default_radius)

        self._log("Creating function spaces V3/V1 and mixed space W...")
        t0 = time.time()
        V3 = fem.functionspace(tissue.mesh, ("Lagrange", degree_3d))
        V1 = fem.functionspace(network.mesh, ("Lagrange", degree_1d))
        W = ufl.MixedFunctionSpace(V3, V1)
        self._log(f"Function spaces created in {time.time() - t0:.3f}s")
        self._bar("after spaces")

        self._V3, self._V1, self._W = V3, V1, W

        # Radii
        cell_radius = self._cell_radius_user
        vertex_radius = self._vertex_radius_user

        if cell_radius is None:
            if network.subdomains is None or self._radius_by_tag is None:
                raise ValueError("Need network.subdomains and radius_by_tag to build cell_radius automatically.")
            cell_radius, radius_per_cell = build_cell_radius_field(
                network.mesh,
                network.subdomains,
                self._radius_by_tag,
                default_radius=default_radius,
                untagged_tag=0,
            )
        else:
            cell_radius.x.scatter_forward()
            tdim = network.mesh.topology.dim
            cell_map = network.mesh.topology.index_map(tdim)
            num_cells = int(cell_map.size_local + cell_map.num_ghosts)
            radius_per_cell = np.empty((num_cells,), dtype=np.float64)
            for c in range(num_cells):
                dof = int(cell_radius.function_space.dofmap.cell_dofs(c)[0])
                radius_per_cell[c] = float(cell_radius.x.array[dof])

        if vertex_radius is None:
            vertex_radius = build_boundary_vertex_radius_field(
                V1,
                network.mesh,
                cell_radius=cell_radius,
                inlet_vertices=network.inlet_vertices,
                outlet_vertices=network.outlet_vertices,
                default_radius=default_radius,
            )

        self._cell_radius = cell_radius
        self._vertex_radius = vertex_radius
        self._radius_per_cell = radius_per_cell

        self._bar("before Circle")
        self._log("Creating Circle objects...")
        t0 = time.time()
        radius_fn = make_cell_radius_callable(network.mesh, radius_per_cell, default_radius=default_radius)
        circle_trial = Circle(network.mesh, radius=radius_fn, degree=circle_qdeg)
        circle_test = Circle(network.mesh, radius=radius_fn, degree=circle_qdeg)
        self._log(f"Circle objects created in {time.time() - t0:.3f}s")
        self._bar("after Circle")

        self._circle_trial = circle_trial
        self._circle_test = circle_test

        self._log("Building UFL trial/test functions and Average(...) ...")
        t0 = time.time()
        p_t, P = ufl.TrialFunctions(W)
        v_t, w = ufl.TestFunctions(W)
        p_avg = Average(p_t, circle_trial, V1)
        v_avg = Average(v_t, circle_test, V1)
        self._log(f"Average(...) created in {time.time() - t0:.3f}s")
        self._bar("after Average")

        dx_tissue = ufl.Measure("dx", domain=tissue.mesh)
        dx_network = ufl.Measure("dx", domain=network.mesh)
        ds_tissue = ufl.Measure("ds", domain=tissue.mesh)
        ds_network = ufl.Measure("ds", domain=network.mesh, subdomain_data=network.boundaries)
        ds_network_outlet = ds_network(network.outlet_marker)

        self._dx_tissue = dx_tissue
        self._dx_network = dx_network
        self._ds_tissue = ds_tissue
        self._ds_network_outlet = ds_network_outlet

        self._log("Creating Constants...")
        k_t = fem.Constant(tissue.mesh, default_scalar_type(params.k_t))
        mu_t = fem.Constant(tissue.mesh, default_scalar_type(params.mu))
        gamma_R = fem.Constant(tissue.mesh, default_scalar_type(params.gamma_R))
        P_cvp_t = fem.Constant(tissue.mesh, default_scalar_type(params.P_cvp))

        mu_n = fem.Constant(network.mesh, default_scalar_type(params.mu))
        gamma = fem.Constant(network.mesh, default_scalar_type(params.gamma))
        gamma_a = fem.Constant(network.mesh, default_scalar_type(params.gamma_a))
        P_cvp_n = fem.Constant(network.mesh, default_scalar_type(params.P_cvp))
        self._log("Constants created.")
        self._bar("after Constants")

        D_area_cell = ufl.pi * cell_radius ** 2
        D_perimeter_cell = 2.0 * ufl.pi * cell_radius
        k_v_cell = (cell_radius ** 2) / 8.0
        D_area_vertex = ufl.pi * vertex_radius ** 2

        self._log("Building bilinear/linear forms a, L ...")
        t0 = time.time()

        a00 = (k_t / mu_t) * ufl.inner(ufl.grad(p_t), ufl.grad(v_t)) * dx_tissue
        a00 += gamma_R * p_t * v_t * ds_tissue
        a00 += gamma * p_avg * v_avg * D_perimeter_cell * dx_network

        a01 = -gamma * P * v_avg * D_perimeter_cell * dx_network
        a01 += -(gamma_a / mu_n) * P * v_avg * D_area_vertex * ds_network_outlet

        a10 = -gamma * p_avg * w * D_perimeter_cell * dx_network

        a11 = (k_v_cell / mu_n) * D_area_cell * ufl.inner(ufl.grad(P), ufl.grad(w)) * dx_network
        a11 += gamma * P * w * D_perimeter_cell * dx_network
        a11 += (gamma_a / mu_n) * P * w * D_area_vertex * ds_network_outlet

        a = a00 + a01 + a10 + a11

        L0 = gamma_R * P_cvp_t * v_t * ds_tissue
        L0 += (gamma_a / mu_n) * P_cvp_n * v_avg * D_area_vertex * ds_network_outlet
        L1 = (gamma_a / mu_n) * P_cvp_n * w * D_area_vertex * ds_network_outlet
        L = L0 + L1

        self._log(f"Forms built in {time.time() - t0:.3f}s")
        self._bar("after forms")

        self._log("Locating inlet dofs...")
        inlet_dofs = fem.locate_dofs_topological(V1, 0, network.inlet_vertices)
        inlet_val = fem.Function(V1)
        inlet_val.x.array[:] = default_scalar_type(params.P_in)
        bc_inlet = fem.dirichletbc(inlet_val, inlet_dofs)
        bcs = [bc_inlet]
        self._bar("after BCs")

        self._a, self._L, self._bcs = a, L, bcs

        self._gamma = gamma
        self._gamma_a = gamma_a
        self._mu_network = mu_n
        self._P_cvp_network = P_cvp_n
        self._D_perimeter_cell = D_perimeter_cell
        self._D_area_vertex = D_area_vertex

        self._built = True

    def solve(self) -> PressureSolution:
        comm = self.comm
        setup_mpi_debug(comm)
        rprint = self._log_cb or make_rank_logger(comm)

        try:
            self._build_if_needed()

            assert self._a is not None and self._L is not None and self._bcs is not None
            assert self._V1 is not None
            assert self._dx_network is not None and self._ds_network_outlet is not None
            assert self._circle_trial is not None
            assert self._gamma is not None and self._gamma_a is not None
            assert self._mu_network is not None and self._P_cvp_network is not None
            assert self._D_perimeter_cell is not None and self._D_area_vertex is not None

            self._log("ABOUT TO CONSTRUCT LinearProblem(...) (this may trigger JIT/assembly)")
            t0 = time.time()
            self._linear_problem = LinearProblem(
                self._a,
                self._L,
                bcs=self._bcs,
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
                    self._gamma
                    * self._D_perimeter_cell
                    * (p_network - Average(p_tissue, self._circle_trial, self._V1))
                    * self._dx_network
            )
            terminal_exchange_form = (
                    (self._gamma_a / self._mu_network)
                    * self._D_area_vertex
                    * (p_network - self._P_cvp_network)
                    * self._ds_network_outlet
            )

            total_wall_exchange = float(assemble_scalar(wall_exchange_form, op=MPI.SUM))
            total_terminal_exchange = float(assemble_scalar(terminal_exchange_form, op=MPI.SUM))

            self._log(
                f"exchange assembled in {time.time() - t0:.3f}s; "
                f"Q_wall={total_wall_exchange} Q_term={total_terminal_exchange}"
            )
            self._bar("after assemble_scalar")

            return PressureSolution(tissue_pressure=p_tissue, network_pressure=p_network)

        except Exception as e:
            abort_on_exception(comm, rprint, e)
            raise

    def close(self) -> None:
        close_if_possible(self._linear_problem)
        self._linear_problem = None

        # Drop cached build artifacts
        self._built = False
        self._V3 = self._V1 = self._W = None
        self._dx_tissue = self._dx_network = self._ds_tissue = self._ds_network_outlet = None
        self._cell_radius = self._vertex_radius = None
        self._radius_per_cell = None
        self._circle_trial = self._circle_test = None
        self._a = self._L = None
        self._bcs = None
        self._gamma = self._gamma_a = None
        self._mu_network = self._P_cvp_network = None
        self._D_perimeter_cell = self._D_area_vertex = None

        collect()

    def __enter__(self) -> "PressureProblem":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
