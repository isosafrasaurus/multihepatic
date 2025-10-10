# src/contracts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional, Any

class Problem(Protocol):
    def solve(self, params: "Parameters") -> "Solution": ...
    def close(self) -> None: ...

class PostProcessor(Protocol):
    def compute(self, **kwargs) -> Any: ...
    def close(self) -> None: ...

@dataclass
class Solution:
    """
    Generic container; does not own PETSc resources—callers must .close()
    to drop references for GC.
    """
    p3d: Any
    p1d: Any
    v3d: Optional[Any] = None

    def close(self) -> None:
        # Drop references; FEniCS/PETSc memory is ref-counted from Python side
        self.p3d = None
        self.p1d = None
        self.v3d = None

