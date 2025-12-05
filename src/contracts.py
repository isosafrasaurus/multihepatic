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
    p3d: Any
    p1d: Any
    v3d: Optional[Any] = None

    def close(self) -> None:
        self.p3d = None
        self.p1d = None
        self.v3d = None
