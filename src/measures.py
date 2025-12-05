from dataclasses import dataclass
from typing import Any

from dolfin import Measure


@dataclass(frozen=True)
class Measures:
    dxOmega: Measure
    dxLambda: Measure
    dsOmega: Measure
    dsOmegaSink: Any


__all__ = ["Measures"]
