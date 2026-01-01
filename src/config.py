from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


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


@dataclass(frozen=True, slots=True)
class AssemblyOptions:
    degree_3d: int = 1
    degree_1d: int = 1
    circle_quadrature_degree: int = 20
