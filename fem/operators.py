from typing import Any
from dolfin import VectorFunctionSpace, TrialFunction, TestFunction, Function, Constant, inner, grad, solve

class ProjectionOperator:
    def __init__(self, Omega: Any) -> None:
        self._Omega = Omega
        self._closed = False

    def project_velocity(self, uh3d: Any, *, k_t: float, mu: float, dx) -> Any:
        if self._closed:
            raise RuntimeError("ProjectionOperator is closed")

        V = VectorFunctionSpace(self._Omega, "CG", 1)
        vT, vS = TrialFunction(V), TestFunction(V)
        a = inner(vT, vS) * dx
        L = inner(Constant(-k_t/mu) * grad(uh3d), vS) * dx
        v = Function(V)
        solve(a == L, v, solver_parameters={"linear_solver": "mumps"})
        v.rename("v3d", "3D Velocity")
        return v

    def close(self) -> None:
        self._Omega = None
        self._closed = True