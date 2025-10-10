# src/simulation.py
from __future__ import annotations
import gc
from typing import Optional, List, Type
from .domain import Domain1D, Domain3D
from .composition import build_assembled_forms, Parameters
from .contracts import Problem, Solution
from .problem import PressureProblem

class Simulation:
    """
    Orchestrates the lifecycle: Domains -> AssembledForms -> Problem -> Solution.
    Simulation owns the Problem and closes it on exit.
    """
    def __init__(
        self,
        Lambda: Domain1D,
        Omega: Domain3D,
        *,
        problem_cls: Type[Problem] = PressureProblem,
        inlet_nodes: Optional[List[int]] = None,
        Omega_sink_subdomain = None,
        order: int = 2,
        linear_solver: str = "mumps",
    ):
        forms = build_assembled_forms(
            Lambda.G,
            Omega.Omega,
            inlet_nodes=inlet_nodes or Lambda.inlet_nodes,
            Omega_sink_subdomain=Omega_sink_subdomain,
            order=order,
        )
        # Pass Omega optionally; PressureProblem ignores it; PressureVelocityProblem uses it
        self.problem: Problem = problem_cls(forms=forms, Omega=Omega.Omega, linear_solver=linear_solver)
        self._closed = False

    def run(self, params: Parameters) -> Solution:
        if self._closed:
            raise RuntimeError("Simulation is closed")
        return self.problem.solve(params)

    # Optional alias for backward compatibility
    def solve(self, params: Parameters) -> Solution:
        return self.run(params)

    def close(self) -> None:
        if not self._closed and self.problem:
            self.problem.close()
            self.problem = None  # type: ignore
            self._closed = True
        gc.collect()

    def __enter__(self) -> "Simulation":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

