from contextlib import ExitStack
from typing import Any, Callable, Optional


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


__all__ = ["ResourcePool", "PetscDestroy", "FenicsHandle"]
