from typing import Sequence, Tuple
from dolfin import LUSolver
from xii import ii_assemble, apply_bc, ii_convert, ii_Function
from .resources import ResourcePool, PetscDestroy

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