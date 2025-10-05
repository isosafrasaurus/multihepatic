# src/utils.py
import gc
from typing import Mapping

def release_result(res: Mapping[str, object]) -> None:
    if not res: return
    for k in ("p3d", "p1d", "velocity"):
        if k in res:
            res[k] = None  # type: ignore
    gc.collect()

