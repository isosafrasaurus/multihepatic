# fem/solver.py
from __future__ import annotations
from typing import Sequence, Tuple, Optional
from dolfin import LUSolver
from xii import ii_assemble, apply_bc, ii_convert, ii_Function
from src.resources import ResourcePool, PetscDestroy

class BlockLinearSolver:
    """
    Deep module. Owns PETSc A,b and LUSolver handle during `solve_block`.
    Guarantees PETSc resources are destroyed on success or failure.
    """
    def __init__(self, linear_solver: str = "mumps") -> None:
        self._linear_solver = linear_solver
        self._closed = False

    def solve_block(
        self,
        W: Sequence,
        a_blocks,
        L_blocks,
        *,
        inlet_bc=None
    ) -> Tuple[object, object]:
        """
        Inputs are FEniCS/xii objects only. Returns (uh3d, uh1d).
        """
        if self._closed:
            raise RuntimeError("BlockLinearSolver is closed")

        with ResourcePool() as pool:
            # Assemble & apply BCs
            A, b = map(ii_assemble, (a_blocks, L_blocks))
            if inlet_bc is not None:
                A, b = apply_bc(A, b, [[], [inlet_bc]])

            # Convert to PETSc backend and register for destruction
            A, b = map(ii_convert, (A, b))
            pool.push(PetscDestroy(A, b).close)

            # Solve
            wh = ii_Function(W)
            solver = LUSolver(A, self._linear_solver)
            try:
                solver.solve(wh.vector(), b)
            finally:
                # Drop references quickly; pool will destroy PETSc handles
                del solver

            uh3d, uh1d = wh  # block split
            uh3d.rename("p3d", "3D Pressure")
            uh1d.rename("p1d", "1D Pressure")
            # we return functions; A,b are destroyed by pool on exit
            return uh3d, uh1d

    def close(self) -> None:
        self._closed = True

