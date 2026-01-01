from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dolfinx.mesh as dmesh
import numpy as np
from mpi4py import MPI
from networks_fenicsx import NetworkMesh


@dataclass(slots=True)
class Domain3D:
    mesh: dmesh.Mesh
    name: str = "Omega"

    def __post_init__(self) -> None:
        tdim = self.mesh.topology.dim
        if tdim >= 1:
            self.mesh.topology.create_connectivity(tdim - 1, tdim)

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm

    @classmethod
    def from_box(
            cls,
            comm: MPI.Comm,
            min_corner: np.ndarray,
            max_corner: np.ndarray,
            target_h: float,
            cell_type: dmesh.CellType = dmesh.CellType.tetrahedron,
            name: str = "Omega",
    ) -> "Domain3D":
        extent = max_corner - min_corner
        n = [max(2, int(np.ceil(extent[i] / target_h))) for i in range(3)]
        mesh = dmesh.create_box(comm, [min_corner.tolist(), max_corner.tolist()], n, cell_type=cell_type)
        return cls(mesh=mesh, name=name)


@dataclass(slots=True)
class Domain1D:
    mesh: dmesh.Mesh
    boundaries: dmesh.MeshTags
    inlet_marker: int
    outlet_marker: int
    subdomains: dmesh.MeshTags | None = None
    name: str = "Lambda"

    def __post_init__(self) -> None:
        self.mesh.topology.create_connectivity(0, 1)
        self.mesh.topology.create_connectivity(1, 0)

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm

    def boundary_vertices(self, marker: int) -> np.ndarray:
        values = self.boundaries.values
        indices = self.boundaries.indices
        return indices[values == marker].astype(np.int32, copy=False)

    @property
    def inlet_vertices(self) -> np.ndarray:
        return self.boundary_vertices(self.inlet_marker)

    @property
    def outlet_vertices(self) -> np.ndarray:
        return self.boundary_vertices(self.outlet_marker)

    @classmethod
    def from_networkx_graph(
            cls,
            graph: Any,
            points_per_edge: int,
            comm: MPI.Comm,
            graph_rank: int = 0,
            inlet_marker: int | None = None,
            outlet_marker: int | None = None,
            color_strategy: Any | None = None,  # ✅ ADDED
            name: str = "Lambda",
    ) -> "Domain1D":
        network = NetworkMesh(
            graph,
            N=points_per_edge,
            comm=comm,
            graph_rank=graph_rank,
            color_strategy=color_strategy,  # ✅ FORWARDED
        )

        inlet = int(network.out_marker) if inlet_marker is None else int(inlet_marker)
        outlet = int(network.in_marker) if outlet_marker is None else int(outlet_marker)

        return cls(
            mesh=network.mesh,
            boundaries=network.boundaries,
            subdomains=getattr(network, "subdomains", None),
            inlet_marker=inlet,
            outlet_marker=outlet,
            name=name,
        )
