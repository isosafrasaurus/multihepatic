from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Callable, Optional, Sequence, Tuple

from dolfin import (
    Constant,
    Function,
    LUSolver,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    grad,
    inner,
    solve,
)
from xii import apply_bc, ii_assemble, ii_convert, ii_Function


class ResourcePool:
    def __init__(self) -> None:
        self._stack = ExitStack()
        self._closed = False

    def push(self, finalizer: Callable[[], None]) -> None:
        if self._closed:
            raise RuntimeError("ResourcePool is closed")
        self._stack.callback(finalizer)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stack.close()

    def __enter__(self) -> "ResourcePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PetscDestroy:
    def __init__(self, A: Optional[Any] = None, b: Optional[Any] = None) -> None:
        self.A = A
        self.b = b

    def close(self) -> None:
        try:
            if self.A is not None:
                from dolfin import as_backend_type
                as_backend_type(self.A).mat().destroy()
        except Exception:
            pass
        try:
            if self.b is not None:
                from dolfin import as_backend_type
                as_backend_type(self.b).vec().destroy()
        except Exception:
            pass
        self.A = None
        self.b = None


class FenicsHandle:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def close(self) -> None:
        self.obj = None


class BlockLinearSolver:
    def __init__(self, linear_solver: str = "mumps") -> None:
        self._linear_solver = linear_solver
        self._closed = False

    def solve_block(
        self,
        W: Sequence,
        a_blocks,
        L_blocks,
        *,
        inlet_bc=None,
    ) -> Tuple[object, object]:
        if self._closed:
            raise RuntimeError("BlockLinearSolver is closed")

        with ResourcePool() as pool:
            A, b = map(ii_assemble, (a_blocks, L_blocks))
            if inlet_bc is not None:
                A, b = apply_bc(A, b, [[], [inlet_bc]])

            A, b = map(ii_convert, (A, b))
            pool.push(PetscDestroy(A, b).close)

            wh = ii_Function(W)
            solver = LUSolver(A, self._linear_solver)
            try:
                solver.solve(wh.vector(), b)
            finally:
                del solver

            uh3d, uh1d = wh
            uh3d.rename("p3d", "3D Pressure")
            uh1d.rename("p1d", "1D Pressure")
            return uh3d, uh1d

    def close(self) -> None:
        self._closed = True


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
        L = inner(Constant(-k_t / mu) * grad(uh3d), vS) * dx
        v = Function(V)
        solve(a == L, v, solver_parameters={"linear_solver": "mumps"})
        v.rename("v3d", "3D Velocity")
        return v

    def close(self) -> None:
        self._Omega = None
        self._closed = True


__all__ = [
    "ResourcePool",
    "PetscDestroy",
    "FenicsHandle",
    "BlockLinearSolver",
    "ProjectionOperator",
]

