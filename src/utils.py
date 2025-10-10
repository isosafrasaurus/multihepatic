
from __future__ import annotations
import gc
from typing import Any

def release_solution(sol: Any) -> None:
    
    if sol is None:
        return
    if hasattr(sol, "close") and callable(sol.close):
        sol.close()
    else:
        for k in ("p3d", "p1d", "v3d", "velocity"):
            if hasattr(sol, k):
                setattr(sol, k, None)
    gc.collect()

