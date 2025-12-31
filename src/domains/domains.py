from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dolfinx.mesh as dmesh
import numpy as np
from mpi4py import MPI


def bbox_from_points(points: np.ndarray, pad: float) -> tuple[np.ndarray, np.ndarray]:
    min_corner = points.min(axis=0) - pad
    max_corner = points.max(axis=0) + pad
    return min_corner, max_corner


def create_box_mesh(
        comm: MPI.Comm,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
        target_h: float,
        *,
        cell_type: dmesh.CellType = dmesh.CellType.tetrahedron,
) -> dmesh.Mesh:
    extent = max_corner - min_corner
    num_cells = [max(2, int(np.ceil(extent[i] / target_h))) for i in range(3)]
    mesh = dmesh.create_box(
        comm,
        [min_corner.tolist(), max_corner.tolist()],
        num_cells,
        cell_type=cell_type,
    )
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    return mesh


@dataclass(slots=True)
class Domain3D:
    mesh: dmesh.Mesh
    name: str = "Omega"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tdim = self.mesh.topology.dim
        if tdim >= 1:
            self.mesh.topology.create_connectivity(tdim - 1, tdim)

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm


@dataclass(slots=True)
class Domain1D:
    mesh: dmesh.Mesh
    boundaries: dmesh.MeshTags
    inlet_marker: int
    outlet_marker: int
    subdomains: dmesh.MeshTags | None = None
    name: str = "Lambda"
    metadata: dict[str, Any] = field(default_factory=dict)

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
    def from_network_mesh(
            cls,
            network_mesh: Any,
            *,
            inlet_marker: int | None = None,
            outlet_marker: int | None = None,
            name: str = "Lambda",
            metadata: dict[str, Any] | None = None,
    ) -> "Domain1D":
        """
        Build Domain1D from a networks_fenicsx.NetworkMesh-like object.

        This method matches the original script's default choice:
        inlet_marker = network.out_marker, outlet_marker = network.in_marker.
        """
        inlet = int(network_mesh.out_marker) if inlet_marker is None else int(inlet_marker)
        outlet = int(network_mesh.in_marker) if outlet_marker is None else int(outlet_marker)

        return cls(
            mesh=network_mesh.mesh,
            subdomains=getattr(network_mesh, "subdomains", None),
            boundaries=network_mesh.boundaries,
            inlet_marker=inlet,
            outlet_marker=outlet,
            name=name,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_networkx_graph(
            cls,
            graph: Any,
            *,
            points_per_edge: int,
            comm: MPI.Comm,
            graph_rank: int = 0,
            inlet_marker: int | None = None,
            outlet_marker: int | None = None,
            name: str = "Lambda",
            metadata: dict[str, Any] | None = None,
    ) -> "Domain1D":
        from networks_fenicsx import NetworkMesh  # lazy import

        network = NetworkMesh(
            graph,
            N=points_per_edge,
            comm=comm,
            graph_rank=graph_rank,
        )
        return cls.from_network_mesh(
            network,
            inlet_marker=inlet_marker,
            outlet_marker=outlet_marker,
            name=name,
            metadata=metadata,
        )
