from __future__ import annotations

from typing import Mapping

import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np


def _radius_lookup_array(
        radius_by_tag: Mapping[int, float] | np.ndarray,
        *,
        max_tag: int,
        default_radius: float,
) -> np.ndarray:
    lookup = np.full((max_tag + 1,), float(default_radius), dtype=np.float64)
    if isinstance(radius_by_tag, np.ndarray):
        n = min(radius_by_tag.size, max_tag + 1)
        lookup[:n] = radius_by_tag[:n].astype(np.float64, copy=False)
    else:
        for tag, radius in radius_by_tag.items():
            if tag < 0:
                raise ValueError(f"Radius tags must be nonnegative integers, got tag={tag}")
            if tag <= max_tag:
                lookup[int(tag)] = float(radius)
    return lookup


def build_cell_radius_field(
        mesh_1d: dmesh.Mesh,
        subdomains: dmesh.MeshTags,
        radius_by_tag: Mapping[int, float] | np.ndarray,
        *,
        default_radius: float,
        untagged_tag: int | None = None,
        name: str = "radius_cell",
) -> tuple[fem.Function, np.ndarray]:
    DG0 = fem.functionspace(mesh_1d, ("DG", 0))
    r_cell = fem.Function(DG0)
    r_cell.name = name

    tdim = mesh_1d.topology.dim
    cell_map = mesh_1d.topology.index_map(tdim)
    num_cells_local = int(cell_map.size_local)
    num_cells = num_cells_local + int(cell_map.num_ghosts)

    max_tag_mesh = int(np.max(subdomains.values)) if subdomains.values.size else 0
    max_tag_map = int(radius_by_tag.size - 1) if isinstance(radius_by_tag, np.ndarray) else (
        int(max(radius_by_tag.keys())) if radius_by_tag else 0
    )
    max_tag = max(max_tag_mesh, max_tag_map, 0)
    if untagged_tag is not None:
        max_tag = max(max_tag, int(untagged_tag))

    if untagged_tag is None:
        # Use an out-of-range tag index mapped to default_radius via lookup fill.
        cell_tags = np.full((num_cells,), -1, dtype=np.int32)
    else:
        cell_tags = np.full((num_cells,), int(untagged_tag), dtype=np.int32)

    cell_tags[subdomains.indices] = subdomains.values

    lookup = _radius_lookup_array(radius_by_tag, max_tag=max_tag, default_radius=default_radius)

    if untagged_tag is None:
        radius_per_cell = np.full((num_cells,), float(default_radius), dtype=np.float64)
        mask = cell_tags >= 0
        radius_per_cell[mask] = lookup[cell_tags[mask]]
    else:
        radius_per_cell = lookup[cell_tags]

    r_cell.x.array[:num_cells_local] = radius_per_cell[:num_cells_local].astype(r_cell.x.array.dtype, copy=False)
    r_cell.x.scatter_forward()

    return r_cell, radius_per_cell


def build_boundary_vertex_radius_field(
        V1: fem.FunctionSpace,
        mesh_1d: dmesh.Mesh,
        *,
        cell_radius: fem.Function,
        inlet_vertices: np.ndarray,
        outlet_vertices: np.ndarray,
        default_radius: float,
        scatter: bool = True,
        name: str = "radius_vertex",
) -> fem.Function:
    """
    Vertex radius field on the network pressure space:
      - default everywhere = default_radius
      - inlet/outlet vertices set from adjacent cell DG0 radius

    IMPORTANT (MPI): locate_dofs_topological is called exactly once per marker per rank.
    """
    r_vertex = fem.Function(V1)
    r_vertex.name = name
    r_vertex.x.array[:] = float(default_radius)

    mesh_1d.topology.create_connectivity(0, 1)
    v2c = mesh_1d.topology.connectivity(0, 1)
    DG0 = cell_radius.function_space

    def radius_at_vertex(v: int) -> float:
        cells = v2c.links(v)
        if len(cells) == 0:
            return float(default_radius)
        c0 = int(cells[0])
        dof = int(DG0.dofmap.cell_dofs(c0)[0])
        return float(cell_radius.x.array[dof])

    inlet_vertices = inlet_vertices.astype(np.int32, copy=False)
    outlet_vertices = outlet_vertices.astype(np.int32, copy=False)

    inlet_dofs = fem.locate_dofs_topological(V1, 0, inlet_vertices)
    outlet_dofs = fem.locate_dofs_topological(V1, 0, outlet_vertices)

    if inlet_vertices.size:
        if len(inlet_dofs) != len(inlet_vertices):
            raise RuntimeError(
                f"Expected 1 dof per vertex (scalar CG on 1D). inlet_dofs={len(inlet_dofs)} "
                f"inlet_vertices={len(inlet_vertices)}"
            )
        r_vertex.x.array[inlet_dofs] = np.array(
            [radius_at_vertex(int(v)) for v in inlet_vertices],
            dtype=r_vertex.x.array.dtype,
        )

    if outlet_vertices.size:
        if len(outlet_dofs) != len(outlet_vertices):
            raise RuntimeError(
                f"Expected 1 dof per vertex (scalar CG on 1D). outlet_dofs={len(outlet_dofs)} "
                f"outlet_vertices={len(outlet_vertices)}"
            )
        r_vertex.x.array[outlet_dofs] = np.array(
            [radius_at_vertex(int(v)) for v in outlet_vertices],
            dtype=r_vertex.x.array.dtype,
        )

    if scatter:
        r_vertex.x.scatter_forward()
    return r_vertex


def make_cell_radius_callable(
        mesh_1d: dmesh.Mesh,
        radius_per_cell: np.ndarray,
        *,
        default_radius: float,
):
    from dolfinx import geometry

    tdim = mesh_1d.topology.dim
    tree = geometry.bb_tree(mesh_1d, tdim)

    def radius_fn(xT: np.ndarray) -> np.ndarray:
        points = np.asarray(xT.T, dtype=np.float64)
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        candidates = geometry.compute_collisions_points(tree, points)
        colliding = geometry.compute_colliding_cells(mesh_1d, candidates, points)

        out = np.full((points.shape[0],), float(default_radius), dtype=np.float64)
        for i in range(points.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                out[i] = float(radius_per_cell[int(links[0])])
        return out

    return radius_fn
