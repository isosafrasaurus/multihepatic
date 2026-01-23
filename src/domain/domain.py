from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

import dolfinx.mesh as dmesh
import numpy as np
from mpi4py import MPI
from networks_fenicsx import NetworkMesh

# XDMF/HDF5 mesh I/O helpers (see domain/mesh.py)
from .mesh import read_mesh_xdmf, read_meshtags_xdmf, load_boundary_facets_from_xdmf


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
        """Tag the given boundary facets with 'marker' in self.boundaries.

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

        # Give facet tags a stable name for XDMF round-tripping/ParaView
        try:
            if self.boundaries is not None and not getattr(self.boundaries, "name", ""):
                self.boundaries.name = "boundaries"
        except Exception:
            pass
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
        """Convenience: define the sink/outlet boundary as an axis-aligned plane.

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

        # NOTE: np.isclose default rtol=1e-5 can lead to major over-selection if
        # coordinates have a large absolute offset. We always set rtol=0 so only
        # atol matters, and choose a robust default atol if not provided.
        tol_was_auto = tol is None
        if tol is None:
            amin, amax = self.axis_bounds(a)
            extent = abs(amax - amin)
            tol = max(
                1e-8 * max(1.0, extent),
                1e-12 * max(1.0, abs(float(value))),
                )

        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        # Build connectivities used by locate_entities_boundary and later measures
        # Connectivity isn't strictly required for locate_entities_boundary, but
        # creating it improves robustness across versions and later postprocessing.
        self.mesh.topology.create_connectivity(fdim, 0)
        self.mesh.topology.create_connectivity(0, fdim)

        def _locate(with_tol: float) -> np.ndarray:
            def plane_marker(x: np.ndarray) -> np.ndarray:
                # x has shape (gdim, num_points)
                return np.isclose(x[a], float(value), atol=float(with_tol), rtol=0.0)

            return dmesh.locate_entities_boundary(self.mesh, fdim, plane_marker)

        facets = _locate(float(tol))

        # If tol was auto-picked, and nothing was found LOCALLY, relax a few times.
        # This helps when the boundary plane isn't represented exactly in floating point.
        if facets.size == 0 and tol_was_auto:
            for factor in (10.0, 100.0, 1000.0, 10000.0):
                facets = _locate(float(tol) * factor)
                if facets.size:
                    tol = float(tol) * factor
                    break

        # In parallel, it's normal that some ranks own zero facets on this plane.
        # Only error if NO rank found any facets.
        n_global = self.comm.allreduce(int(facets.size), op=MPI.SUM)
        if n_global == 0:
            raise ValueError(
                f"mark_outlet_axis_plane found no boundary facets on any rank for axis={axis!r}, "
                f"value={value}, tol={tol}. (Try increasing tol.)"
            )

        # Ensure boundaries MeshTags exists on ALL ranks (can be empty locally),
        # so downstream code consistently uses ds(subdomain_data=...) everywhere.
        if self.boundaries is None:
            empty = np.zeros((0,), dtype=np.int32)
            self.boundaries = dmesh.meshtags(self.mesh, fdim, empty, empty)
            try:
                self.boundaries.name = "boundaries"
            except Exception:
                pass

        # Add local facets if we have any on this rank
        if facets.size:
            self.add_boundary_facets(facets, marker=marker, override=override)
        else:
            # Still record the outlet marker even on ranks with no local facets
            self.outlet_marker = int(marker)
        return facets

    @classmethod
    def from_xdmf(
            cls,
            comm: MPI.Comm,
            path: str | Path,
            *,
            mesh_name: str = "Grid",
            ghost_mode: dmesh.GhostMode = dmesh.GhostMode.shared_facet,
            boundaries_name: str | None = None,
            boundaries_path: str | Path | None = None,
            outlet_marker: int | None = None,
    ) -> "Domain3D":
        """Construct a Domain3D by reading a mesh from an XDMF (HDF5-backed) file.

        Parameters
        ----------
        comm:
            MPI communicator to read/partition the mesh on.
        path:
            Path to the .xdmf mesh file (or an .h5/.hdf5 file if the matching .xdmf exists).
        mesh_name:
            XDMF Grid name (commonly "Grid").
        ghost_mode:
            Ghosting mode for parallel runs.
        boundaries_name:
            Optional MeshTags name (facet tags) to also read from the file.
            If provided, Domain3D.boundaries is set and you may set outlet_marker.
        boundaries_path:
            Optional separate XDMF path to read boundaries tags from (defaults to `path`).
            Optional separate XDMF path to read boundaries tags from (defaults to `path`).
        outlet_marker:
            If provided and boundaries_name is provided, sets Domain3D.outlet_marker.
        """
        mesh = read_mesh_xdmf(comm, path, mesh_name=mesh_name, ghost_mode=ghost_mode)
        dom = cls(mesh=mesh)

        if boundaries_name is not None:
            bpath = path if boundaries_path is None else boundaries_path
            fdim = mesh.topology.dim - 1
            dom.boundaries = read_meshtags_xdmf(mesh, bpath, name=boundaries_name, dim=fdim)
            # Ensure a stable tag name for round-tripping
            try:
                if dom.boundaries is not None and not getattr(dom.boundaries, "name", ""):
                    dom.boundaries.name = str(boundaries_name)
            except Exception:
                pass
            if outlet_marker is not None:
                dom.outlet_marker = int(outlet_marker)

        return dom

    # Backwards/UX alias
    @classmethod
    def from_meshfile(
            cls,
            comm: MPI.Comm,
            path: str | Path,
            **kwargs: Any,
    ) -> "Domain3D":
        """Alias for from_xdmf(...)."""
        return cls.from_xdmf(comm, path, **kwargs)

    def set_boundaries_from_xdmf(
            self,
            path: str | Path,
            *,
            name: str,
            outlet_marker: int | None = None,
            replace: bool = True,
            override: bool = True,
    ) -> None:
        """Load facet MeshTags from an XDMF file into this Domain3D.

        This is a mesh-based alternative to mark_outlet_axis_plane(...). It assumes that
        the MeshTags in `path` correspond to this exact mesh (same entity numbering).

        Parameters
        ----------
        path:
            XDMF containing facet MeshTags for this mesh.
        name:
            MeshTags name in the XDMF file.
        outlet_marker:
            If provided, set Domain3D.outlet_marker to this value.
        replace:
            If True, replace existing boundaries tags; otherwise merge with existing tags.
        override:
            If merging, whether the newly read tags override existing ones on overlaps.
        """
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        tags = read_meshtags_xdmf(mesh, path, name=name, dim=fdim)
        try:
            tags.name = str(name)
        except Exception:
            pass

        if replace or self.boundaries is None:
            self.boundaries = tags
        else:
            # Merge facet tags; new tags can override existing ones if override=True
            self.boundaries = _merge_meshtags(
                mesh,
                fdim,
                self.boundaries,
                np.asarray(tags.indices, dtype=np.int32),
                np.asarray(tags.values, dtype=np.int32),
                override=override,
            )

        if outlet_marker is not None:
            self.outlet_marker = int(outlet_marker)

    def mark_outlet_from_xdmf(
            self,
            path: str | Path,
            *,
            tags_name: str,
            marker: int,
            replace_boundaries: bool = True,
            override: bool = True,
    ) -> np.ndarray:
        """Mark the outlet/sink boundary using facet tags loaded from XDMF.

        This is a mesh-based alternative to mark_outlet_axis_plane(...).

        Two modes:
          - replace_boundaries=True (default): read *all* facet tags from XDMF into
            self.boundaries and set self.outlet_marker=marker.
          - replace_boundaries=False: only read the facets with the given marker and
            merge them into existing boundaries via add_boundary_facets(...).

        Returns
        -------
        np.ndarray
            The facet indices tagged as the outlet marker on this rank.
        """
        marker_i = int(marker)

        if replace_boundaries:
            self.set_boundaries_from_xdmf(path, name=tags_name, outlet_marker=marker_i, replace=True)
            assert self.boundaries is not None
            facets = np.asarray(self.boundaries.indices, dtype=np.int32)[
                np.asarray(self.boundaries.values, dtype=np.int32) == marker_i
                ]
        else:
            facets = load_boundary_facets_from_xdmf(self.mesh, path, tags_name=tags_name, marker=marker_i)
            if facets.size == 0:
                raise ValueError(
                    f"No facets found with marker={marker_i} in MeshTags {tags_name!r} from {path!r}."
                )
            self.add_boundary_facets(facets, marker=marker_i, override=override)

        if facets.size == 0:
            raise ValueError(
                f"MeshTags {tags_name!r} from {path!r} contains no facets with marker={marker_i}."
            )

        return facets

    @classmethod
    def from_box(
            cls,
            comm: MPI.Comm,
            min_corner: np.ndarray,
            max_corner: np.ndarray,
            target_h: float,
            cell_type: dmesh.CellType = dmesh.CellType.tetrahedron,
    ) -> "Domain3D":
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