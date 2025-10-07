
import gc
from typing import Any

def release_solution(sol: Any) -> None:
    if sol is None:
        return
    if hasattr(sol, "free") and callable(sol.free):
        sol.free()
        return
    for k in ("p3d", "p1d", "v3d", "velocity"):
        if hasattr(sol, k):
            setattr(sol, k, None)
    gc.collect()

