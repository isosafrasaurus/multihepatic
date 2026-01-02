from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import dolfinx.mesh as dmesh
import numpy as np
from mpi4py import MPI
from networks_fenicsx import NetworkMesh


def _axis_to_int(axis: int | str) -> int:
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
        return axis
    a = axis.lower().strip()
    if a == "x":
        return 0
    if a == "y":
        return 1
    if a == "z":
        return 2
    raise ValueError(f"axis must be 0/1/2 or 'x'/'y'/'z'; got {axis!r}")


def _merge_meshtags(
    mesh: dmesh.Mesh,
    dim: int,
    old: dmesh.MeshTags,
    new_indices: np.ndarray,
    new_values: np.ndarray,
    *,
    override: bool,
) -> dmesh.MeshTags:
    """Merge tags on a given entity dim. If override=True, new_values win on overlaps."""
    oi = np.asarray(old.indices, dtype=np.int32)
    ov = np.asarray(old.values, dtype=np.int32)
    ni = np.asarray(new_indices, dtype=np.int32).ravel()
    nv = np.asarray(new_values, dtype=np.int32).ravel()

    if ni.size == 0:
        return old

    # Concatenate so that "last occurrence wins" per index group.
    if override:
        idx_all = np.concatenate([oi, ni])
        val_all = np.concatenate([ov, nv])
    else:
        idx_all = np.concatenate([ni, oi])
        val_all = np.concatenate([nv, ov])

    # Stable sort by entity index
    order = np.argsort(idx_all, kind="mergesort")
    idx_s = idx_all[order]
    val_s = val_all[order]

    # Take the last value per unique index (so "later in concat" wins)
    uniq_idx, first, counts = np.unique(idx_s, return_index=True, return_counts=True)
    last_pos = first + counts - 1
    uniq_val = val_s[last_pos]

    return dmesh.meshtags(mesh, dim, uniq_idx, uniq_val)


@dataclass(slots=True)
class Domain3D:
    mesh: dmesh.Mesh

    # Facet (tdim-1) tags describing boundary subdomains for the tissue mesh.
    # If set, PressureProblem will apply Robin terms only on ds(outlet_marker).
    boundaries: dmesh.MeshTags | None = None
    outlet_marker: int | None = None

    def __post_init__(self) -> None:
        tdim = self.mesh.topology.dim
        if tdim >= 1:
            self.mesh.topology.create_connectivity(tdim - 1, tdim)

    @property
    def comm(self) -> MPI.Comm:
        return self.mesh.comm

    def axis_bounds(self, axis: int | str) -> tuple[float, float]:
        """Global (MPI) min/max coordinate along axis."""
        a = _axis_to_int(axis)
        x = self.mesh.geometry.x  # shape: (num_points_local, gdim)
        if x.size == 0:
            local_min = float("inf")
            local_max = float("-inf")
        else:
            local_min = float(np.min(x[:, a]))
            local_max = float(np.max(x[:, a]))
        gmin = self.comm.allreduce(local_min, op=MPI.MIN)
        gmax = self.comm.allreduce(local_max, op=MPI.MAX)
        return gmin, gmax

    def add_boundary_facets(
        self,
        facets: np.ndarray,
        *,
        marker: int,
        override: bool = True,
    ) -> None:
        """
        Tag the given boundary facets with 'marker' in self.boundaries.

        If self.boundaries already exists, merge tags. By default, new tags override
        existing tags for the same facet indices.
        """
        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        facets = np.asarray(facets, dtype=np.int32).ravel()
        if facets.size == 0:
            raise ValueError("No facets were provided to add_boundary_facets().")

        # Ensure unique/sorted indices for stable behavior
        facets = np.unique(facets)
        values = np.full((facets.size,), int(marker), dtype=np.int32)

        if self.boundaries is None:
            self.boundaries = dmesh.meshtags(self.mesh, fdim, facets, values)
        else:
            if self.boundaries.dim != fdim:
                raise ValueError(
                    f"Domain3D.boundaries has dim={self.boundaries.dim}, expected {fdim} for facet tags."
                )
            self.boundaries = _merge_meshtags(
                self.mesh, fdim, self.boundaries, facets, values, override=override
            )

        # Record which tag corresponds to the Robin/outflow part of the boundary
        self.outlet_marker = int(marker)

    def mark_outlet_axis_plane(
        self,
        axis: int | str,
        *,
        value: float | None = None,
        side: str | None = None,
        tol: float | None = None,
        marker: int = 1,
        override: bool = True,
    ) -> np.ndarray:
        """
        Convenience: define the sink/outlet boundary as an axis-aligned plane.

        Examples:
          - x = xmax:  mark_outlet_axis_plane("x", side="max")
          - y = 0.0:   mark_outlet_axis_plane("y", value=0.0)
          - z = zmin:  mark_outlet_axis_plane(2, side="min")

        This tags the located boundary facets with 'marker' and stores them in
        Domain3D.boundaries; Domain3D.outlet_marker is set to 'marker'.
        """
        a = _axis_to_int(axis)
        if value is None:
            if side is None:
                raise ValueError("Provide either value=... or side='min'/'max'.")
            s = side.lower().strip()
            amin, amax = self.axis_bounds(a)
            if s == "min":
                value = amin
            elif s == "max":
                value = amax
            else:
                raise ValueError("side must be 'min' or 'max'.")

        if tol is None:
            amin, amax = self.axis_bounds(a)
            tol = 1e-8 * max(1.0, abs(amax - amin))

        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        def plane_marker(x: np.ndarray) -> np.ndarray:
            # x has shape (gdim, num_points)
            return np.isclose(x[a], float(value), atol=float(tol))

        facets = dmesh.locate_entities_boundary(self.mesh, fdim, plane_marker)

        if facets.size == 0:
            raise ValueError(
                f"mark_outlet_axis_plane found no boundary facets for axis={axis!r}, "
                f"value={value}, tol={tol}. (Try increasing tol.)"
            )

        self.add_boundary_facets(facets, marker=marker, override=override)
        return facets

    @classmethod
    def from_box(
        cls,
        comm: MPI.Comm,
        min_corner: np.ndarray,
        max_corner: np.ndarray,
        target_h: float,
        cell_type: dmesh.CellType = dmesh.CellType.tetrahedron,
    ) -> Domain3D:
        extent = max_corner - min_corner
        n = [max(2, int(np.ceil(extent[i] / target_h))) for i in range(3)]
        mesh = dmesh.create_box(comm, [min_corner.tolist(), max_corner.tolist()], n, cell_type=cell_type)
        return cls(mesh=mesh)


@dataclass(slots=True)
class Domain1D:
    mesh: dmesh.Mesh
    boundaries: dmesh.MeshTags
    inlet_marker: int
    outlet_marker: int
    subdomains: dmesh.MeshTags | None = None

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
    def from_network(
        cls,
        graph: Any,
        points_per_edge: int,
        comm: MPI.Comm,
        graph_rank: int = 0,
        inlet_marker: int | None = None,
        outlet_marker: int | None = None,
        color_strategy: Any | None = None,
    ) -> "Domain1D":
        network = NetworkMesh(
            graph,
            N=points_per_edge,
            comm=comm,
            graph_rank=graph_rank,
            color_strategy=color_strategy,
        )

        inlet = int(network.out_marker) if inlet_marker is None else int(inlet_marker)
        outlet = int(network.in_marker) if outlet_marker is None else int(outlet_marker)

        return cls(
            mesh=network.mesh,
            boundaries=network.boundaries,
            subdomains=getattr(network, "subdomains", None),
            inlet_marker=inlet,
            outlet_marker=outlet,
        )
