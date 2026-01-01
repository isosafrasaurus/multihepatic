from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import MemoryManager


@dataclass(slots=True)
class PressureSolution:
    tissue_pressure: Any
    network_pressure: Any
    cell_radius: Any | None = None
    vertex_radius: Any | None = None
    total_wall_exchange: float | None = None
    total_terminal_exchange: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def release(self) -> None:
        self.tissue_pressure = None
        self.network_pressure = None
        self.cell_radius = None
        self.vertex_radius = None
        self.metadata.clear()
        MemoryManager.collect()


@dataclass(slots=True)
class PressureVelocitySolution(PressureSolution):
    tissue_velocity: Any | None = None

    def release(self) -> None:
        self.tissue_velocity = None
        super().release()
