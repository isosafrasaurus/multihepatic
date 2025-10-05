# fem/solver.py
from typing import Sequence, Tuple, Optional
import gc
from dolfin import LUSolver, as_backend_type
from xii import ii_assemble, apply_bc, ii_convert, ii_Function

def _destroy_petsc(A, b) -> None:
    try:
        as_backend_type(A).mat().destroy()
    except Exception:
        pass
    try:
        as_backend_type(b).vec().destroy()
    except Exception:
        pass

def solve_block(W: Sequence, a_blocks, L_blocks, inlet_bc=None,
                linear_solver: str = "mumps") -> Tuple[object, object]:
    """
    Inputs are FEniCS/xii objects only. Returns (uh3d, uh1d).
    """
    A, b = map(ii_assemble, (a_blocks, L_blocks))
    if inlet_bc is not None:
        A, b = apply_bc(A, b, [[], [inlet_bc]])

    A, b = map(ii_convert, (A, b))
    wh = ii_Function(W)

    solver = LUSolver(A, linear_solver)
    solver.solve(wh.vector(), b)
    del solver

    _destroy_petsc(A, b)
    A = b = None
    gc.collect()

    uh3d, uh1d = wh
    uh3d.rename("p3d", "3D Pressure")
    uh1d.rename("p1d", "1D Pressure")
    return uh3d, uh1d
