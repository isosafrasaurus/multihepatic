from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from dolfin import Constant


class Problem(Protocol):
    def solve(self, params: "Parameters") -> "Solution": ...
    def close(self) -> None: ...


class PostProcessor(Protocol):
    def compute(self, **kwargs) -> Any: ...
    def close(self) -> None: ...


@dataclass
class Parameters:
    gamma: float
    gamma_a: float
    gamma_R: float
    mu: float
    k_t: float
    P_in: float
    P_cvp: float


@dataclass
class ParamConstants:
    gamma: Constant = field(default_factory=lambda: Constant(0.0))
    gamma_a: Constant = field(default_factory=lambda: Constant(0.0))
    gamma_R: Constant = field(default_factory=lambda: Constant(0.0))
    mu: Constant = field(default_factory=lambda: Constant(0.0))
    k_t: Constant = field(default_factory=lambda: Constant(0.0))
    P_in: Constant = field(default_factory=lambda: Constant(0.0))
    P_cvp: Constant = field(default_factory=lambda: Constant(0.0))

    def assign_from(self, p: Parameters) -> None:
        self.gamma.assign(float(p.gamma))
        self.gamma_a.assign(float(p.gamma_a))
        self.gamma_R.assign(float(p.gamma_R))
        self.mu.assign(float(p.mu))
        self.k_t.assign(float(p.k_t))
        self.P_in.assign(float(p.P_in))
        self.P_cvp.assign(float(p.P_cvp))


@dataclass
class Solution:
    p3d: Any
    p1d: Any
    v3d: Optional[Any] = None

    def close(self) -> None:
        self.p3d = None
        self.p1d = None
        self.v3d = None


def release_solution(sol: Any) -> None:
    
    if sol is None:
        return
    if hasattr(sol, "close") and callable(sol.close):
        sol.close()
    else:
        for k in ("p3d", "p1d", "v3d", "velocity"):
            if hasattr(sol, k):
                setattr(sol, k, None)
    gc.collect()


__all__ = [
    "Problem",
    "PostProcessor",
    "Parameters",
    "ParamConstants",
    "Solution",
    "release_solution",
]
