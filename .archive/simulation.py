import gc
from typing import List, Optional, Type

from .core import Parameters, Problem, Solution
from .domain import Domain1D, Domain3D
from .forms import build_assembled_forms
from .problem import PressureProblem


class Simulation:
    def __init__(
            self,
            Lambda: Domain1D,
            Omega: Domain3D,
            *,
            problem_cls: Type[Problem] = PressureProblem,
            inlet_nodes: Optional[List[int]] = None,
            Omega_sink_subdomain=None,
            order: int = 2,
            linear_solver: str = "mumps",
    ) -> None:
        forms = build_assembled_forms(
            Lambda.graph,
            Omega.mesh,
            inlet_nodes=inlet_nodes or Lambda.inlet_nodes,
            Omega_sink_subdomain=Omega_sink_subdomain,
            order=order,
        )
        self.problem: Problem = problem_cls(forms=forms, Omega=Omega.mesh, linear_solver=linear_solver)
        self._closed = False

    def run(self, params: Parameters) -> Solution:
        if self._closed:
            raise RuntimeError("Simulation is closed")
        return self.problem.solve(params)

    def solve(self, params: Parameters) -> Solution:
        return self.run(params)

    def close(self) -> None:
        if not self._closed and self.problem:
            self.problem.close()
            self.problem = None
            self._closed = True
        gc.collect()

    def __enter__(self) -> "Simulation":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["Simulation"]
