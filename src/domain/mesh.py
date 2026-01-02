from __future__ import annotations

from pathlib import Path

import dolfinx.mesh as dmesh
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI


def _normalize_xdmf_path(path: str | Path) -> Path:
    """Normalize a mesh path for XDMF reading.

    Users sometimes pass the underlying .h5/.hdf5 file. DOLFINx typically needs the
    .xdmf *metadata* file; if a matching .xdmf exists next to the HDF5 file, we use it.
    """
    p = Path(path).expanduser().resolve()
    suf = p.suffix.lower()
    if suf in {".h5", ".hdf5", ".hdf"}:
        xdmf = p.with_suffix(".xdmf")
        if xdmf.exists():
            return xdmf
        raise ValueError(
            f"Got HDF5 file {p.name!r}. Please pass the corresponding .xdmf file "
            f"(expected {xdmf.name!r} next to it)."
        )
    return p


def read_mesh_xdmf(
    comm: MPI.Comm,
    path: str | Path,
    *,
    mesh_name: str = "Grid",
    ghost_mode: dmesh.GhostMode = dmesh.GhostMode.shared_facet,
) -> dmesh.Mesh:
    """Read a DOLFINx mesh from an XDMF.

    Parameters
    comm:
        MPI communicator to read/partition the mesh on.
    path:
        Path to an .xdmf file (or an .h5/.hdf5 file if the matching .xdmf exists).
    mesh_name:
        XDMF Grid name. DOLFINx commonly uses "Grid" by default.
    ghost_mode:
        Ghosting mode for parallel runs.

    Returns
    dolfinx.mesh.Mesh
    """
    p = _normalize_xdmf_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with XDMFFile(comm, str(p), "r") as xdmf:
        # API differs slightly across dolfinx versions; be defensive.
        try:
            mesh = xdmf.read_mesh(name=mesh_name, ghost_mode=ghost_mode)
        except TypeError:
            try:
                mesh = xdmf.read_mesh(name=mesh_name)
            except TypeError:
                mesh = xdmf.read_mesh()

    # Ensure basic connectivities for downstream operations
    tdim = mesh.topology.dim
    if tdim >= 1:
        mesh.topology.create_connectivity(tdim - 1, tdim)
    return mesh


def read_meshtags_xdmf(
    mesh: dmesh.Mesh,
    path: str | Path,
    *,
    name: str,
    dim: int | None = None,
) -> dmesh.MeshTags:
    """Read MeshTags from an XDMF file.

    To reliably read tags, you should have written them with a stable name, e.g.:

        tags.name = "boundaries"
        xdmf.write_meshtags(tags, mesh.geometry)

    Then read them back with name="boundaries".
    """
    p = _normalize_xdmf_path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with XDMFFile(mesh.comm, str(p), "r") as xdmf:
        # API differs slightly across dolfinx versions; be defensive.
        try:
            tags = xdmf.read_meshtags(mesh, name=name)
        except TypeError:
            try:
                tags = xdmf.read_meshtags(mesh, name)
            except TypeError:
                # Some versions may require geometry as an argument
                tags = xdmf.read_meshtags(mesh, name, mesh.geometry)  # type: ignore[misc]

    # Preserve the requested name for round-tripping.
    try:
        tags.name = str(name)
    except Exception:
        pass

    if dim is not None and int(tags.dim) != int(dim):
        raise ValueError(f"MeshTags {name!r} has dim={tags.dim}, expected dim={dim}.")
    return tags


def entities_with_marker(tags: dmesh.MeshTags, marker: int) -> np.ndarray:
    """Return tag.indices where tag.values == marker."""
    marker_i = int(marker)
    mask = np.asarray(tags.values, dtype=np.int32) == marker_i
    return np.asarray(tags.indices, dtype=np.int32)[mask]


def load_boundary_facets_from_xdmf(
    mesh: dmesh.Mesh,
    path: str | Path,
    *,
    tags_name: str,
    marker: int,
) -> np.ndarray:
    """Convenience: read facet MeshTags and return facet indices for a given marker."""
    fdim = mesh.topology.dim - 1
    tags = read_meshtags_xdmf(mesh, path, name=tags_name, dim=fdim)
    facets = entities_with_marker(tags, marker)
    return facets
