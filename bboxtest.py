#!/dolfinx-env/bin/python3
"""
bboxtest.py  (NO CLI, SERIAL ONLY)

Hard-coded inputs:
  - NIfTI: /workspace/newData.nii.gz
  - VTK:   /workspace/dataNew.vtk  (fallback: dataNew.vtk)

Key properties:
  - Serial only: uses MPI.COMM_SELF (minimal MPI needed by dolfinx).
  - No networks_fenicsx, no fenicsx_ii.
  - Builds 1D network as ONE dolfinx mesh via dolfinx.mesh.create_mesh.
  - mark_outlet_from_nifti is implemented inline (no tissue.method import/call).

Outputs:
  out_nifti_bbox_sink/
    tissue_box.xdmf
    network_1d.xdmf
    tissue_bc_labels.xdmf
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import sys
import numpy as np

from mpi4py import MPI

import dolfinx.mesh as dmesh
from dolfinx.io import XDMFFile
from dolfinx import fem, default_scalar_type

import basix.ufl

def ensure_entity_cell_connectivity(mesh: dmesh.Mesh, entity_dim: int) -> None:
    """
    Ensure the mesh has entity->cell connectivity for the given entity_dim.

    XDMFFile.write_meshtags requires this for tags on entities with dim < tdim.
    """
    tdim = mesh.topology.dim
    if int(entity_dim) == int(tdim):
        return
    # This is the one XDMF needs: entity_dim -> tdim
    mesh.topology.create_connectivity(int(entity_dim), int(tdim))
    # Often useful elsewhere too (safe no-op if already present)
    try:
        mesh.topology.create_connectivity(int(tdim), int(entity_dim))
    except Exception:
        pass



# -----------------------------------------------------------------------------
# Hard-coded file paths (NO CLI)
# -----------------------------------------------------------------------------
NIFTI_PATH = Path("/workspace/newData.nii.gz")
VTK_PATH_PRIMARY = Path("/workspace/dataNew.vtk")
VTK_PATH_FALLBACK = Path("dataNew.vtk")

SINK_LABEL = 3
SINK_MARKER = 3

OUTDIR = Path("out_nifti_bbox_sink")

TARGET_H_FACTOR = 2.0
VOXEL_TARGET_H = 2.0


def r0(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# NIfTI helpers
# =============================================================================
def load_nifti(path: Path, *, volume: int = 0) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "This script requires nibabel to read NIfTI.\n"
            "Install with: pip install nibabel\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    if not path.exists():
        raise FileNotFoundError(str(path))

    img = nib.load(str(path))
    affine = np.asarray(img.affine, dtype=np.float64)

    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        if not (0 <= int(volume) < int(data.shape[3])):
            raise ValueError(f"Requested volume={volume} but NIfTI has shape {data.shape}.")
        data3 = np.asanyarray(data[..., int(volume)])
    else:
        data3 = np.asanyarray(data)

    if data3.ndim != 3:
        raise ValueError(f"Expected 3D (or 4D) NIfTI. Got data ndim={data3.ndim}, shape={data3.shape}.")

    shape = (int(data3.shape[0]), int(data3.shape[1]), int(data3.shape[2]))
    return np.asarray(data3), affine, shape


def nifti_voxel_spacing(affine: np.ndarray) -> np.ndarray:
    A = np.asarray(affine[:3, :3], dtype=np.float64)
    return np.linalg.norm(A, axis=0)


def nifti_bbox_corners_world(affine: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """
    8 corners of voxel-domain bounding box in world coords, using half-voxel padding:
      [-0.5, n-0.5] in each axis.
    """
    nx_, ny_, nz_ = shape
    i0, i1 = -0.5, nx_ - 0.5
    j0, j1 = -0.5, ny_ - 0.5
    k0, k1 = -0.5, nz_ - 0.5

    corners_ijk = np.array(
        [
            [i0, j0, k0],
            [i0, j0, k1],
            [i0, j1, k0],
            [i0, j1, k1],
            [i1, j0, k0],
            [i1, j0, k1],
            [i1, j1, k0],
            [i1, j1, k1],
        ],
        dtype=np.float64,
    )
    ones = np.ones((8, 1), dtype=np.float64)
    ijk_h = np.hstack([corners_ijk, ones])
    xyz = (np.asarray(affine, dtype=np.float64) @ ijk_h.T).T[:, :3]
    return xyz


def world_to_voxel(inv_affine: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_world, dtype=np.float64).reshape((-1, 3))
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xh = np.hstack([xyz, ones])
    ijk = (np.asarray(inv_affine, dtype=np.float64) @ xh.T).T[:, :3]
    return ijk


def inside_fraction(ijk: np.ndarray, shape: tuple[int, int, int], tol: float = 0.5) -> float:
    ijk = np.asarray(ijk, dtype=np.float64)
    lo = -float(tol)
    hi = (np.asarray(shape, dtype=np.float64) - 1.0) + float(tol)
    ok = np.all((ijk >= lo) & (ijk <= hi), axis=1)
    return float(np.mean(ok)) if ok.size else 0.0


# =============================================================================
# VTK POLYDATA v5.1 ASCII reader (your exact format)
# =============================================================================
def vtk_ascii_tokens(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            for tok in s.split():
                yield tok


def read_vtk_polydata_v51_points_lines_radius(path: Path) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """
    Parses:
      - POINTS N float
      - LINES nlines nconn
        OFFSETS vtktypeint64
        <offsets...>   # until CONNECTIVITY
        CONNECTIVITY vtktypeint64
        <nconn ints>
      - POINT_DATA N
        FIELD FieldData 1
        Radius 1 N double
        <N floats>
    """
    it = iter(vtk_ascii_tokens(path))

    def next_tok() -> str:
        try:
            return next(it)
        except StopIteration:
            raise ValueError(f"Unexpected EOF while parsing {path}")

    # Seek DATASET POLYDATA
    tok = next_tok()
    while tok.upper() != "DATASET":
        tok = next_tok()
    if next_tok().upper() != "POLYDATA":
        raise ValueError("Expected DATASET POLYDATA")

    # Seek POINTS
    tok = next_tok()
    while tok.upper() != "POINTS":
        tok = next_tok()
    npoints = int(next_tok())
    _ = next_tok()  # dtype

    pts = np.fromiter((float(next_tok()) for _ in range(3 * npoints)), dtype=np.float64, count=3 * npoints)
    pts = pts.reshape((npoints, 3))

    # Seek LINES
    tok = next_tok()
    while tok.upper() != "LINES":
        tok = next_tok()
    nlines = int(next_tok())
    nconn = int(next_tok())

    if next_tok().upper() != "OFFSETS":
        raise ValueError("Expected OFFSETS after LINES")
    _ = next_tok()  # offsets dtype

    offsets_list: list[int] = []
    tok = next_tok()
    while tok.upper() != "CONNECTIVITY":
        offsets_list.append(int(tok))
        tok = next_tok()

    _ = next_tok()  # connectivity dtype
    conn = np.fromiter((int(next_tok()) for _ in range(nconn)), dtype=np.int64, count=nconn)

    offsets = np.asarray(offsets_list, dtype=np.int64)
    if offsets.size not in (nlines, nlines + 1):
        raise ValueError(f"Unexpected OFFSETS length {offsets.size}; expected {nlines} or {nlines+1}.")

    def line_range(i: int) -> tuple[int, int]:
        if offsets.size == nlines:
            s = int(offsets[i])
            e = int(offsets[i + 1]) if i < nlines - 1 else int(nconn)
            return s, e
        else:
            return int(offsets[i]), int(offsets[i + 1])

    segments: list[tuple[int, int]] = []
    for i in range(nlines):
        s, e = line_range(i)
        if e - s < 2:
            continue
        idx = conn[s:e]
        for j in range(int(idx.size) - 1):
            a = int(idx[j])
            b = int(idx[j + 1])
            if a != b:
                segments.append((a, b))

    if not segments:
        raise ValueError("VTK parsed but produced 0 segments")

    # Seek POINT_DATA
    tok = next_tok()
    while tok.upper() != "POINT_DATA":
        tok = next_tok()
    n_pd = int(next_tok())
    if n_pd != npoints:
        raise ValueError(f"POINT_DATA {n_pd} != POINTS {npoints}")

    # FIELD FieldData 1
    if next_tok().upper() != "FIELD":
        raise ValueError("Expected FIELD after POINT_DATA")
    _ = next_tok()  # FieldData
    n_fields = int(next_tok())
    if n_fields != 1:
        raise ValueError(f"Expected FIELD ... 1, got {n_fields}")

    field_name = next_tok()
    if field_name != "Radius":
        raise ValueError(f"Expected field 'Radius', got {field_name!r}")
    ncomp = int(next_tok())
    ntuples = int(next_tok())
    _ = next_tok()  # dtype
    if ncomp != 1 or ntuples != npoints:
        raise ValueError(f"Expected Radius 1 {npoints} ..., got Radius {ncomp} {ntuples} ...")

    radius = np.fromiter((float(next_tok()) for _ in range(npoints)), dtype=np.float64, count=npoints)
    return pts, segments, radius


# =============================================================================
# Coordinate mode detection + conversions
# =============================================================================
def pick_vtk_coordinate_mode(points_vtk: np.ndarray, affine: np.ndarray, shape: tuple[int, int, int]) -> str:
    pts = np.asarray(points_vtk, dtype=np.float64).reshape((-1, 3))
    inv_aff = np.linalg.inv(np.asarray(affine, dtype=np.float64))

    frac_ras = inside_fraction(world_to_voxel(inv_aff, pts), shape)
    frac_lps = inside_fraction(world_to_voxel(inv_aff, pts * np.array([-1.0, -1.0, 1.0])), shape)
    frac_ijk = inside_fraction(pts, shape)

    scores = {"ras": frac_ras, "lps": frac_lps, "ijk": frac_ijk}
    return max(scores.keys(), key=lambda k: scores[k])


def world_to_vtk_coords(mode: str, affine: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_world, dtype=np.float64).reshape((-1, 3))
    if mode == "ras":
        return xyz
    if mode == "lps":
        return xyz * np.array([-1.0, -1.0, 1.0], dtype=np.float64)
    if mode == "ijk":
        inv_aff = np.linalg.inv(np.asarray(affine, dtype=np.float64))
        return world_to_voxel(inv_aff, xyz)
    raise ValueError(f"Unknown mode {mode!r}")


def mesh_to_world_affine(mode: str, affine: np.ndarray) -> np.ndarray | None:
    """
    Returns 4x4 mapping mesh coords (VTK coords) -> NIfTI world (RAS).
      - ras: None (identity)
      - lps: flip x,y
      - ijk: use NIfTI affine
    """
    if mode == "ras":
        return None
    if mode == "lps":
        A = np.eye(4, dtype=np.float64)
        A[0, 0] = -1.0
        A[1, 1] = -1.0
        return A
    if mode == "ijk":
        return np.asarray(affine, dtype=np.float64)
    raise ValueError(f"Unknown mode {mode!r}")


def apply_affine_4x4(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    P = np.asarray(pts, dtype=np.float64).reshape((-1, 3))
    ones = np.ones((P.shape[0], 1), dtype=np.float64)
    Ph = np.hstack([P, ones])
    Q = (A @ Ph.T).T
    return Q[:, :3]


# =============================================================================
# Inline "mark_outlet_from_nifti" (no import from Domain3D)
# =============================================================================
def mark_outlet_from_nifti_inline(
    mesh: dmesh.Mesh,
    nifti_path: Path,
    *,
    label_value: int,
    marker_value: int,
    mesh_to_world: np.ndarray | None = None,
    volume: int = 0,
) -> tuple[dmesh.MeshTags, np.ndarray]:
    """
    Mark exterior boundary facets whose midpoint samples to NIfTI label_value.
    Returns (facet_tags, marked_facet_indices).

    facet_tags has dim = tdim-1 and contains ONLY the marked facets with value marker_value.
    """
    data3, affine, shape = load_nifti(nifti_path, volume=volume)
    inv_aff = np.linalg.inv(np.asarray(affine, dtype=np.float64))

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    if tdim != 3 or gdim != 3:
        raise ValueError(f"Requires 3D mesh. Got tdim={tdim}, gdim={gdim}.")

    fdim = tdim - 1

    # Exterior facets (robust across versions)
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
    try:
        facets = dmesh.exterior_facet_indices(mesh.topology)
    except Exception:
        def all_boundary(x: np.ndarray) -> np.ndarray:
            return np.ones((x.shape[1],), dtype=np.bool_)
        facets = dmesh.locate_entities_boundary(mesh, fdim, all_boundary)

    facets = np.asarray(facets, dtype=np.int32).ravel()
    if facets.size == 0:
        raise RuntimeError("No exterior boundary facets found.")

    mids = dmesh.compute_midpoints(mesh, fdim, facets)  # (N,3) in mesh coords

    # Map mesh coords -> NIfTI world (RAS) if needed
    world_pts = mids
    if mesh_to_world is not None and mids.size:
        world_pts = apply_affine_4x4(mesh_to_world, mids)

    # World -> voxel coordinates (float)
    ones = np.ones((world_pts.shape[0], 1), dtype=np.float64)
    wh = np.hstack([world_pts, ones])
    ijk_f = (inv_aff @ wh.T).T[:, :3]

    # Nearest-neighbor (avoid bankers rounding quirks)
    ijk = np.floor(ijk_f + 0.5).astype(np.int64, copy=False)

    nx_, ny_, nz_ = shape
    inb = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < nx_) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < ny_) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < nz_)
    )

    selected = np.zeros((facets.size,), dtype=bool)
    if np.any(inb):
        ii = ijk[inb, 0]
        jj = ijk[inb, 1]
        kk = ijk[inb, 2]
        vals = np.asarray(data3[ii, jj, kk])

        if np.issubdtype(vals.dtype, np.floating):
            match = np.isclose(vals.astype(np.float64), float(label_value))
        else:
            match = vals.astype(np.int64, copy=False) == int(label_value)

        selected[np.flatnonzero(inb)[match]] = True

    marked_facets = facets[selected].astype(np.int32, copy=False)
    if marked_facets.size == 0:
        raise ValueError(
            f"Found 0 boundary facets whose midpoint maps to NIfTI label {int(label_value)}.\n"
            "Common causes: mesh/world mismatch, units mismatch, or label is not on/near the *box boundary*."
        )

    marked_facets = np.unique(marked_facets)
    values = np.full((marked_facets.size,), int(marker_value), dtype=np.int32)
    tags = dmesh.meshtags(mesh, fdim, marked_facets, values)
    try:
        tags.name = "boundaries"
    except Exception:
        pass

    return tags, marked_facets


# =============================================================================
# Build tissue box mesh
# =============================================================================
def create_tissue_box(comm: MPI.Comm, min_corner: np.ndarray, max_corner: np.ndarray, target_h: float) -> dmesh.Mesh:
    min_corner = np.asarray(min_corner, dtype=np.float64).reshape((3,))
    max_corner = np.asarray(max_corner, dtype=np.float64).reshape((3,))
    extent = max_corner - min_corner
    n = [max(2, int(np.ceil(float(extent[i]) / float(target_h)))) for i in range(3)]
    mesh = dmesh.create_box(comm, [min_corner.tolist(), max_corner.tolist()], n, cell_type=dmesh.CellType.tetrahedron)
    return mesh


# =============================================================================
# Build a single dolfinx 1D mesh directly from VTK segments
# =============================================================================
def build_network_mesh_direct(
    comm: MPI.Comm,
    points: np.ndarray,
    segments: list[tuple[int, int]],
    radius_per_point: np.ndarray,
    *,
    default_radius: float = 1.0,
    inlet_marker: int = 1,
    outlet_marker: int = 2,
) -> tuple[dmesh.Mesh, dmesh.MeshTags, dmesh.MeshTags, np.ndarray]:
    """
    Returns:
      mesh_1d, boundaries_vertex_tags(dim=0), subdomains_cell_tags(dim=1), radius_by_tag ndarray
    """
    points = np.asarray(points, dtype=np.float64)
    radius_per_point = np.asarray(radius_per_point, dtype=np.float64).reshape((-1,))
    npts = int(points.shape[0])

    # Edge radius = mean(endpoint radii (directed segments))
    seg_r = np.array(
        [0.5 * (float(radius_per_point[a]) + float(radius_per_point[b])) for (a, b) in segments],
        dtype=np.float64,
    )

    # Keep only vertices used by at least one segment
    used: set[int] = set()
    for (a, b) in segments:
        used.add(int(a))
        used.add(int(b))
    used_sorted = np.array(sorted(used), dtype=np.int64)

    old2new = np.full((npts,), -1, dtype=np.int64)
    old2new[used_sorted] = np.arange(used_sorted.size, dtype=np.int64)

    x = points[used_sorted, :].astype(np.float64, copy=False)
    r_v = radius_per_point[used_sorted].astype(np.float64, copy=False)

    # Build undirected unique edges (mesh cannot represent parallel identical edges)
    edge_bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (a0, b0), rr in zip(segments, seg_r):
        a = int(old2new[int(a0)])
        b = int(old2new[int(b0)])
        if a < 0 or b < 0 or a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        edge_bucket[(u, v)].append(float(rr))

    if not edge_bucket:
        raise ValueError("After remapping/deduplication, network has 0 edges.")

    cells = np.array(list(edge_bucket.keys()), dtype=np.int64)
    edge_radii = np.array([float(np.mean(edge_bucket[k])) for k in edge_bucket.keys()], dtype=np.float64)

    # Leaf detection (degree==1) on the undirected mesh graph
    deg = np.zeros((x.shape[0],), dtype=np.int32)
    for (u, v) in cells:
        deg[int(u)] += 1
        deg[int(v)] += 1
    leaves = np.flatnonzero(deg == 1).astype(np.int32)

    if leaves.size == 0:
        inlet_v = int(np.argmin(x[:, 0]))
        outlet_vs = np.array([int(np.argmax(x[:, 0]))], dtype=np.int32)
    else:
        inlet_v = int(leaves[int(np.argmax(r_v[leaves]))])
        outlet_vs = leaves[leaves != inlet_v]
        if outlet_vs.size == 0:
            d = np.linalg.norm(x - x[inlet_v], axis=1)
            outlet_vs = np.array([int(np.argmax(d))], dtype=np.int32)

    inlet_vertices = np.array([inlet_v], dtype=np.int32)
    outlet_vertices = np.asarray(outlet_vs, dtype=np.int32)

    # Coordinate element for an interval embedded in R^3:
    # basix.ufl.element(..., shape=(3,)) replaces ufl.VectorElement in newer stacks.
    coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))

    # dolfinx.mesh.create_mesh signature: create_mesh(comm, cells, e, x)
    mesh_1d = dmesh.create_mesh(comm, cells, coord_el, x)
    # Needed for writing vertex MeshTags to XDMF (dim=0 tags need 0->cell connectivity)
    mesh_1d.topology.create_connectivity(0, 1)
    mesh_1d.topology.create_connectivity(1, 0)

    # Subdomains: unique tag per cell: 1..ncells
    tdim = mesh_1d.topology.dim
    ncells = int(cells.shape[0])
    cell_indices = np.arange(ncells, dtype=np.int32)
    cell_tags = (np.arange(ncells, dtype=np.int32) + 1)
    subdomains = dmesh.meshtags(mesh_1d, tdim, cell_indices, cell_tags)
    try:
        subdomains.name = "subdomains"
    except Exception:
        pass

    # radius_by_tag lookup array where index==tag
    radius_by_tag = np.full((ncells + 1,), float(default_radius), dtype=np.float64)
    radius_by_tag[1:] = edge_radii

    # Boundaries (vertex tags): inlet + all outlets
    pairs: list[tuple[int, int]] = [(int(v), int(inlet_marker)) for v in inlet_vertices.tolist()]
    pairs += [(int(v), int(outlet_marker)) for v in outlet_vertices.tolist()]
    pairs = sorted(set(pairs), key=lambda p: p[0])

    b_idx = np.array([p[0] for p in pairs], dtype=np.int32)
    b_val = np.array([p[1] for p in pairs], dtype=np.int32)
    boundaries = dmesh.meshtags(mesh_1d, 0, b_idx, b_val)
    try:
        boundaries.name = "boundaries"
    except Exception:
        pass

    return mesh_1d, boundaries, subdomains, radius_by_tag


# =============================================================================
# Debug helper output (same idea as your earlier helper)
# =============================================================================
def write_tissue_bc_label_mesh(
    outdir: Path,
    mesh: dmesh.Mesh,
    *,
    boundaries: dmesh.MeshTags | None,
    outlet_marker: int | None,
    time: float = 0.0,
    basename: str = "tissue_bc_labels",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    tdim = mesh.topology.dim
    fdim = tdim - 1

    DG0 = fem.functionspace(mesh, ("DG", 0))
    bc_type = fem.Function(DG0)
    bc_type.name = "bc_type"
    facet_marker = fem.Function(DG0)
    facet_marker.name = "facet_marker"

    bc_type.x.array[:] = default_scalar_type(0.0)
    facet_marker.x.array[:] = default_scalar_type(0.0)

    if boundaries is not None:
        mesh.topology.create_connectivity(fdim, tdim)
        f2c = mesh.topology.connectivity(fdim, tdim)

        for facet, tag in zip(boundaries.indices, boundaries.values):
            tag_i = int(tag)
            for cell in f2c.links(int(facet)):
                dof = int(DG0.dofmap.cell_dofs(int(cell))[0])
                facet_marker.x.array[dof] = default_scalar_type(tag_i)
                if outlet_marker is not None and tag_i == int(outlet_marker):
                    bc_type.x.array[dof] = default_scalar_type(1.0)

    bc_type.x.scatter_forward()
    facet_marker.x.scatter_forward()

    with XDMFFile(mesh.comm, str(outdir / f"{basename}.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(bc_type, time)
        xdmf.write_function(facet_marker, time)
        if boundaries is not None:
            xdmf.write_meshtags(boundaries, mesh.geometry)  # type: ignore[arg-type]


def main() -> None:
    # Minimal MPI required by dolfinx, but force strict serial.
    if MPI.COMM_WORLD.size != 1:
        raise SystemExit("This script is serial-only. Run WITHOUT mpiexec/srun.")
    comm = MPI.COMM_SELF

    vtk_path = VTK_PATH_PRIMARY if VTK_PATH_PRIMARY.exists() else VTK_PATH_FALLBACK
    if not vtk_path.exists():
        raise FileNotFoundError(f"VTK file not found:\n  {VTK_PATH_PRIMARY}\n  {VTK_PATH_FALLBACK}")

    r0(f"[i] NIfTI: {NIFTI_PATH}")
    r0(f"[i] VTK:   {vtk_path}")

    # --- Load NIfTI
    nifti_data, affine, shape = load_nifti(NIFTI_PATH, volume=0)
    spacing = nifti_voxel_spacing(affine)
    world_corners = nifti_bbox_corners_world(affine, shape)

    # --- Read VTK
    pts_vtk, segments, radius_pt = read_vtk_polydata_v51_points_lines_radius(vtk_path)
    r0(f"[i] Parsed VTK: points={pts_vtk.shape[0]}, segments={len(segments)}")
    r0(f"[i] Radius range (per point): {float(radius_pt.min()):.6g} .. {float(radius_pt.max()):.6g}")

    # --- Detect coordinate mode
    inv_aff = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    frac_ras = inside_fraction(world_to_voxel(inv_aff, pts_vtk), shape)
    frac_lps = inside_fraction(world_to_voxel(inv_aff, pts_vtk * np.array([-1.0, -1.0, 1.0])), shape)
    frac_ijk = inside_fraction(pts_vtk, shape)

    mode = pick_vtk_coordinate_mode(pts_vtk, affine, shape)
    r0(f"[i] Coordinate mode auto-detected: {mode!r}")
    r0(f"[i] Inside fractions: ras={frac_ras:.3f}, lps={frac_lps:.3f}, ijk={frac_ijk:.3f}")

    # --- Build tissue bounding box in VTK coords
    corners_vtk = world_to_vtk_coords(mode, affine, world_corners)
    min_corner = corners_vtk.min(axis=0)
    max_corner = corners_vtk.max(axis=0)

    if mode == "ijk":
        target_h = float(VOXEL_TARGET_H)
    else:
        target_h = float(TARGET_H_FACTOR) * float(np.min(spacing))

    r0(f"[i] Tissue box min_corner (VTK coords): {min_corner}")
    r0(f"[i] Tissue box max_corner (VTK coords): {max_corner}")
    r0(f"[i] Tissue target_h: {target_h:g}")

    tissue_mesh = create_tissue_box(comm, min_corner, max_corner, target_h)

    # --- Build 1D network mesh directly (no networks_fenicsx)
    network_mesh, network_boundaries, network_subdomains, radius_by_tag = build_network_mesh_direct(
        comm,
        pts_vtk,
        segments,
        radius_pt,
        default_radius=1.0,
        inlet_marker=1,
        outlet_marker=2,
    )
    r0(f"[i] Network: vertices={network_mesh.geometry.x.shape[0]}, cells={network_mesh.topology.index_map(1).size_local}")

    # --- Mark sink facets from NIfTI label (inline implementation)
    r0(f"[i] Tagging sink boundary facets from NIfTI label {SINK_LABEL} ...")
    m2w = mesh_to_world_affine(mode, affine)
    tissue_boundaries, marked_facets = mark_outlet_from_nifti_inline(
        tissue_mesh,
        NIFTI_PATH,
        label_value=int(SINK_LABEL),
        marker_value=int(SINK_MARKER),
        mesh_to_world=m2w,
        volume=0,
    )
    r0(f"[i] Done. Marked facets: {int(marked_facets.size)}; outlet_marker={SINK_MARKER}")

    # --- Write outputs
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with XDMFFile(comm, str(OUTDIR / "tissue_box.xdmf"), "w") as xdmf:
        xdmf.write_mesh(tissue_mesh)
        xdmf.write_meshtags(tissue_boundaries, tissue_mesh.geometry)  # type: ignore[arg-type]

    # Make meshtags writable: ensure entity->cell connectivity exists
    ensure_entity_cell_connectivity(network_mesh, network_boundaries.dim)  # dim=0 -> cells
    ensure_entity_cell_connectivity(network_mesh, network_subdomains.dim)  # dim=1 (cells) is usually fine, but harmless

    with XDMFFile(comm, str(OUTDIR / "network_1d.xdmf"), "w") as xdmf:
        xdmf.write_mesh(network_mesh)
        xdmf.write_meshtags(network_boundaries, network_mesh.geometry)  # type: ignore[arg-type]
        xdmf.write_meshtags(network_subdomains, network_mesh.geometry)  # type: ignore[arg-type]

    write_tissue_bc_label_mesh(
        OUTDIR,
        tissue_mesh,
        boundaries=tissue_boundaries,
        outlet_marker=int(SINK_MARKER),
        time=0.0,
        basename="tissue_bc_labels",
    )

    pts_min = pts_vtk.min(axis=0)
    pts_max = pts_vtk.max(axis=0)
    r0(f"[i] VTK network bounds: min={pts_min}, max={pts_max}")
    r0(f"[i] Tissue box bounds:  min={min_corner}, max={max_corner}")
    r0(f"[OK] Wrote outputs to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
