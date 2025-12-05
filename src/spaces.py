from dataclasses import dataclass
from typing import List

from dolfin import FunctionSpace


@dataclass(frozen=True)
class Spaces:
    V3: FunctionSpace
    V1: FunctionSpace

    @property
    def W(self) -> List[FunctionSpace]:
        return [self.V3, self.V1]


__all__ = ["Spaces"]
