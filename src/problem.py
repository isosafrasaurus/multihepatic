# src/problem.py
import gc
from typing import Dict, Any, Optional
from fem.solver import solve_block
from fem.projection import project_velocity
from .composition import assign_params

class PressureProblem:
    """Build once using forms + constants provided by composition, solve many times by reassigning constants."""
    def __init__(self, *, G, Omega, forms: Dict[str, Any]):
        self.G = G
        self.Omega = Omega
        self.forms = forms
        self._last = None

    def solve(self, **params) -> Dict[str, Any]:
        assign_params(self.forms["consts"], params)
        uh3d, uh1d = solve_block(self.forms["W"], self.forms["a_blocks"], self.forms["L_blocks"],
                                 inlet_bc=self.forms["inlet_bc"], linear_solver="mumps")
        self._last = dict(p3d=uh3d, p1d=uh1d)
        return self._last

    def dispose(self):
        self._last = None
        self.forms = None
        gc.collect()

class PressureVelocityProblem(PressureProblem):
    def solve(self, **params) -> Dict[str, Any]:
        res = super().solve(**params)
        v = project_velocity(self.Omega, res["p3d"], k_t=params["k_t"], mu=params["mu"],
                             dx=self.forms["measures"]["dxOmega"])
        res["velocity"] = v
        return res

