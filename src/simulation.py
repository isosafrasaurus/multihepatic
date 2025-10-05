
import gc
from typing import Optional, Sequence, Type, Dict, Any
from .domain import Domain1D, Domain3D
from .composition import build_coupled_pressure_forms
from .problem import PressureProblem, PressureVelocityProblem

class Simulation:
    
    def __init__(self, Lambda: Domain1D, Omega: Domain3D, *,
                 problem_cls: Type[PressureProblem] = PressureProblem,
                 inlet_nodes: Optional[Sequence[int]] = None,
                 Omega_sink_subdomain=None, order: int = 2):
        forms = build_coupled_pressure_forms(
            Lambda.G, Omega.Omega, inlet_nodes=inlet_nodes or Lambda.inlet_nodes,
            Omega_sink_subdomain=Omega_sink_subdomain, order=order
        )
        self.problem = problem_cls(G=Lambda.G, Omega=Omega.Omega, forms=forms)

    def solve(self, **params) -> Dict[str, Any]:
        return self.problem.solve(**params)

    def dispose(self):
        if self.problem:
            self.problem.dispose()
            self.problem = None
        gc.collect()
