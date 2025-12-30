from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Problem(Protocol):
    def solve(self) -> Any:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class Parameters:
    gamma: float = 3.6145827741262347e-05
    gamma_a: float = 8.225197366649115e-08
    gamma_R: float = 8.620057937882969e-08
    mu: float = 1.0e-3
    k_t: float = 1.0e-10
    P_in: float = 100.0 * 133.322
    P_cvp: float = 1.0 * 133.322


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


class MemoryManager:
    @staticmethod
    def close_if_possible(obj: Any) -> None:
        if obj is None:
            return
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def destroy_if_possible(obj: Any) -> None:
        if obj is None:
            return
        destroy = getattr(obj, "destroy", None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass

    @staticmethod
    def collect() -> None:
        gc.collect()
        try:
            from petsc4py import PETSc  # type: ignore

            PETSc.garbage_cleanup()
        except Exception:
            pass


@dataclass(slots=True)
class PressureSolution:
    tissue_pressure: Any
    network_pressure: Any
    cell_radius: Any | None = None
    vertex_radius: Any | None = None
    total_wall_exchange: float | None = None
    total_terminal_exchange: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def release(self) -> None:
        self.tissue_pressure = None
        self.network_pressure = None
        self.cell_radius = None
        self.vertex_radius = None
        self.metadata.clear()
        MemoryManager.collect()


@dataclass(slots=True)
class PressureVelocitySolution(PressureSolution):
    tissue_velocity: Any | None = None

    def release(self) -> None:
        self.tissue_velocity = None
        super().release()
