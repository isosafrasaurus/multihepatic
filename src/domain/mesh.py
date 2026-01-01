from __future__ import annotations

from pathlib import Path

import dolfinx.mesh as dmesh
import numpy as np
import ufl
from mpi4py import MPI

from .domain import Domain1D, Domain3D


class VTKImportError(RuntimeError):
    pass


def _require_meshio():
    try:
        import meshio  # type: ignore
    except Exception as e:
        raise VTKImportError(
            "Reading .vtk requires `meshio`. Install with: pip install meshio"
        ) from e
    return meshio


def read_domain3d_vtk(comm: MPI.Comm, path: Path, *, name: str = "Omega") -> Domain3D:
    if comm.size != 1:
        raise VTKImportError(
            "read_domain3d_vtk currently supports only comm.size == 1.\n"
            "For parallel workflows, convert VTK->XDMF and read with dolfinx.io.XDMFFile."
        )

    meshio = _require_meshio()
    msh = meshio.read(str(path))

    # Prefer tetra cells
    cells = None
    for c in msh.cells:
        if c.type in ("tetra", "tetra10"):
            cells = c.data
            break
    if cells is None:
        raise VTKImportError(f"No tetra cells found in {path}")

    x = np.asarray(msh.points, dtype=np.float64)
    if x.shape[1] < 3:
        x3 = np.zeros((x.shape[0], 3), dtype=np.float64)
        x3[:, : x.shape[1]] = x
        x = x3

    cell = ufl.Cell("tetrahedron", geometric_dimension=3)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))

    mesh = dmesh.create_mesh(comm, np.asarray(cells, dtype=np.int64), x, domain)
    return Domain3D(mesh=mesh, name=name)


def read_domain1d_vtk(
        comm: MPI.Comm,
        path: Path,
        *,
        inlet_marker: int,
        outlet_marker: int,
        boundary_marker_name: str = "boundaries",
        subdomain_marker_name: str = "subdomains",
        name: str = "Lambda",
) -> Domain1D:
    if comm.size != 1:
        raise VTKImportError(
            "read_domain1d_vtk currently supports only comm.size == 1.\n"
            "For MPI runs, convert VTK->XDMF or build the network via Domain1D.from_networkx_graph."
        )

    meshio = _require_meshio()
    msh = meshio.read(str(path))

    # Prefer line cells
    cells = None
    for c in msh.cells:
        if c.type in ("line", "line3"):
            cells = c.data
            break
    if cells is None:
        raise VTKImportError(f"No line cells found in {path}")

    x = np.asarray(msh.points, dtype=np.float64)
    if x.shape[1] < 3:
        x3 = np.zeros((x.shape[0], 3), dtype=np.float64)
        x3[:, : x.shape[1]] = x
        x = x3

    # boundary markers
    if boundary_marker_name not in msh.point_data:
        raise VTKImportError(
            f"VTK missing point_data['{boundary_marker_name}'] required for Domain1D.boundaries."
        )
    bmark = np.asarray(msh.point_data[boundary_marker_name], dtype=np.int32).reshape(-1)
    if bmark.shape[0] != x.shape[0]:
        raise VTKImportError("Boundary marker array length does not match number of points.")

    cell = ufl.Cell("interval", geometric_dimension=3)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = dmesh.create_mesh(comm, np.asarray(cells, dtype=np.int64), x, domain)

    # Build MeshTags for vertices: include all vertices
    v_idx = np.arange(x.shape[0], dtype=np.int32)
    boundaries = dmesh.meshtags(mesh, 0, v_idx, bmark)

    # Optional cell subdomains
    subdomains = None
    if msh.cell_data and subdomain_marker_name in msh.cell_data:
        # meshio stores cell_data as dict[name] -> list per cell block
        # We pick the block matching "line" cells we used.
        cell_blocks = msh.cell_data[subdomain_marker_name]
        # Find the matching index of the chosen line block
        line_block_i = None
        for i, c in enumerate(msh.cells):
            if c.type in ("line", "line3"):
                line_block_i = i
                break
        if line_block_i is not None and line_block_i < len(cell_blocks):
            smark = np.asarray(cell_blocks[line_block_i], dtype=np.int32).reshape(-1)
            c_idx = np.arange(smark.shape[0], dtype=np.int32)
            subdomains = dmesh.meshtags(mesh, 1, c_idx, smark)

    return Domain1D(
        mesh=mesh,
        boundaries=boundaries,
        subdomains=subdomains,
        inlet_marker=int(inlet_marker),
        outlet_marker=int(outlet_marker),
        name=name,
    )
