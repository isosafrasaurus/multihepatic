from __future__ import annotations

import gc
from typing import Any


class MemoryManager:
    @staticmethod
    def close_if_possible(obj: Any) -> None:
        if obj is None:
            return
        close = getattr(obj, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def destroy_if_possible(obj: Any) -> None:
        if obj is None:
            return
        destroy = getattr(obj, "destroy", None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass

    @staticmethod
    def collect() -> None:
        gc.collect()
        try:
            from petsc4py import PETSc  # type: ignore

            PETSc.garbage_cleanup()
        except Exception:
            pass
