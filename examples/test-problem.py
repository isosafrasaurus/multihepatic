#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from mpi4py import MPI

import dolfinx.mesh as dmesh
import dolfinx.fem as fem

# Make sure repo root is importable (adjust if your layout differs)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import (
    Domain1D,
    Domain3D,
    PressureProblem,
    AssemblyOptions,
    SolverOptions,
    Parameters,
    OutputOptions,
    write_solution,
)
from src.system import make_rank_logger, setup_mpi_debug, collect


def _find_first_existing(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(str(p) for p in paths))


def find_nifti_path(repo_root: Path, *, stem: str = "newData") -> Path:
    """
    Find newData.nii / newData.nii.gz in a few common locations.
    """
    candidates = [
        repo_root / f"{stem}.nii",
        repo_root / f"{stem}.nii.gz",
        repo_root / "data-cropped" / f"{stem}.nii",
        repo_root / "data-cropped" / f"{stem}.nii.gz",
        Path(__file__).with_name(f"{stem}.nii"),
        Path(__file__).with_name(f"{stem}.nii.gz"),
        ]
    return _find_first_existing(candidates)


def find_vtk_path(repo_root: Path, *, stem: str = "dataNew") -> Path:
    """
    Find dataNew.vtk in a few common locations (matches your current script behavior).
    """
    candidates = [
        repo_root / "data-cropped" / f"{stem}.vtk",
        repo_root / f"{stem}.vtk",
        Path(__file__).with_name(f"{stem}.vtk"),
        ]
    return _find_first_existing(candidates)


def nifti_world_bounds_from_nonzero(
        nifti_path: Path,
        *,
        volume: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute axis-aligned world-coordinate bounds of the *nonzero* voxels in a NIfTI.

    Returns:
      min_corner_world: (3,) float64
      max_corner_world: (3,) float64
      voxel_spacing_world: (3,) float64  (norms of affine columns)

    Notes on coordinate handling:
      - NIfTI affine maps voxel indices (i,j,k) to world coordinates of voxel centers.
      - We compute min/max over the *centers* of all nonzero voxels, then expand by
        a conservative half-voxel pad in each world axis to cover voxel extents.
      - The padding is robust even if the affine includes rotations/shear:
            pad_world_axis = 0.5 * sum(|A_axis,voxel_dim|)
        where A = affine[:3,:3].
    """
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise ImportError(
            "This script requires nibabel for NIfTI reading. Install it with: pip install nibabel"
        ) from e

    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj)

    if data.ndim == 4:
        if not (0 <= int(volume) < int(data.shape[3])):
            raise ValueError(f"Requested volume={volume} but NIfTI has shape {data.shape}.")
        data3 = np.asanyarray(data[..., int(volume)])
    else:
        data3 = np.asanyarray(data)

    if data3.ndim != 3:
        raise ValueError(f"Expected a 3D (or 4D) NIfTI. Got ndim={data3.ndim}, shape={data3.shape}.")

    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"NIfTI affine must be (4,4). Got {affine.shape}.")

    mask = (data3 != 0)
    if not np.any(mask):
        raise ValueError(f"{nifti_path} has no nonzero voxels; cannot compute bounds.")

    I, J, K = np.nonzero(mask)
    I = I.astype(np.float64, copy=False)
    J = J.astype(np.float64, copy=False)
    K = K.astype(np.float64, copy=False)

    A = affine[:3, :3]  # world per voxel
    b = affine[:3, 3]   # world origin

    ijk = np.vstack([I, J, K])  # (3,N)
    xyz = (A @ ijk) + b[:, None]  # (3,N) world coords of voxel centers

    min_center = np.min(xyz, axis=1)
    max_center = np.max(xyz, axis=1)

    # Conservative half-voxel padding in each world axis (handles rotations/shear)
    pad_world = 0.5 * np.sum(np.abs(A), axis=1)

    min_corner = (min_center - pad_world).astype(np.float64, copy=False)
    max_corner = (max_center + pad_world).astype(np.float64, copy=False)

    # Approx voxel spacing (column norms). Not perfect under shear, but a good scale.
    spacing = np.sqrt(np.sum(A * A, axis=0)).astype(np.float64, copy=False)

    return min_corner, max_corner, spacing


def read_vtk_points_and_point_radius_legacy_ascii(
        path: Path, *, radius_name: str = "Radius"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal ASCII legacy VTK POLYDATA reader for:
      - POINTS n <float|double>
      - POINT_DATA n
      - FIELD ... containing: Radius 1 n <type>

    Returns:
      points: (n,3) float64
      radius: (n,)  float64
    """
    lines = path.read_text().splitlines()
    nlines = len(lines)

    # ---- POINTS
    points = None
    npts = None
    for i in range(nlines):
        s = lines[i].strip()
        if s.startswith("POINTS"):
            parts = s.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POINTS header: {s!r}")
            npts = int(parts[1])
            need = 3 * npts
            toks: list[str] = []
            j = i + 1
            while j < nlines and len(toks) < need:
                ss = lines[j].strip()
                if ss:
                    toks.extend(ss.split())
                j += 1
            if len(toks) < need:
                raise ValueError("POINTS section ended early.")
            points = np.asarray(toks[:need], dtype=np.float64).reshape((npts, 3))
            break
    if points is None or npts is None:
        raise ValueError(f"No POINTS section found in {path}")

    # ---- Radius in FIELD (point data)
    radius = None
    for i in range(nlines):
        s = lines[i].strip()
        if s.startswith(radius_name + " "):
            # e.g. "Radius 1 3174 double"
            parts = s.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed {radius_name} header: {s!r}")
            ncomp = int(parts[1])
            ntup = int(parts[2])
            need = ncomp * ntup

            toks: list[str] = []
            j = i + 1
            while j < nlines and len(toks) < need:
                ss = lines[j].strip()
                if ss:
                    # Stop if we hit a new section header
                    if ss.startswith(("CELL_DATA", "POINT_DATA", "LINES", "POLYGONS", "VERTICES", "TRIANGLE_STRIPS")):
                        break
                    toks.extend(ss.split())
                j += 1

            if len(toks) < need:
                raise ValueError(f"{radius_name} array ended early: need {need}, got {len(toks)}")

            arr = np.asarray(toks[:need], dtype=np.float64)
            if ncomp != 1:
                arr = arr.reshape((ntup, ncomp))[:, 0]
            radius = arr.reshape((ntup,))
            break

    if radius is None:
        raise ValueError(f"Did not find FIELD array named {radius_name!r} in {path}")

    if radius.shape[0] != points.shape[0]:
        raise ValueError(f"{radius_name} length {radius.shape[0]} != num points {points.shape[0]}")

    return points, radius


def build_coord_radius_lookup(
        points: np.ndarray, radius: np.ndarray, *, decimals: int = 8
) -> dict[tuple[float, float, float], float]:
    """
    Build a coordinate -> radius lookup using rounded coordinates.
    If multiple VTK points share the same rounded coordinate, use the mean radius.
    """
    ptsq = np.round(points.astype(np.float64, copy=False), decimals=decimals)
    sums: dict[tuple[float, float, float], float] = {}
    counts: dict[tuple[float, float, float], int] = {}

    for p, r in zip(ptsq, radius):
        k = (float(p[0]), float(p[1]), float(p[2]))
        sums[k] = sums.get(k, 0.0) + float(r)
        counts[k] = counts.get(k, 0) + 1

    return {k: sums[k] / counts[k] for k in sums}


def override_inlet_and_sinks_by_vtk_point_id(
        comm: MPI.Comm,
        network: Domain1D,
        vtk_points: np.ndarray,
        *,
        inlet_point_id: int,
        tol: float = 1e-8,
) -> None:
    """
    Tag boundary vertices on the *network mesh*:
      - inlet = vertex matching VTK POINTS[inlet_point_id]
      - all other endpoints = sink/outlet
    """
    mesh = network.mesh
    if mesh.topology.dim != 1:
        raise RuntimeError("Expected a 1D network mesh.")

    inlet_xyz = vtk_points[inlet_point_id].astype(np.float64, copy=False)

    mesh.topology.create_connectivity(0, 1)
    v2c = mesh.topology.connectivity(0, 1)

    imap = mesh.topology.index_map(0)
    n_owned = int(imap.size_local)
    x = mesh.geometry.x

    # Find closest owned vertex on each rank
    if n_owned == 0:
        local_best_d2 = float("inf")
        local_best_v = -1
    else:
        xo = x[:n_owned]
        d2 = np.sum((xo - inlet_xyz[None, :]) ** 2, axis=1)
        local_best_v = int(np.argmin(d2))
        local_best_d2 = float(d2[local_best_v])

    # Pick a single winning rank (root chooses min d2, tie-break by rank)
    d2_list = comm.gather(local_best_d2, root=0)
    v_list = comm.gather(local_best_v, root=0)
    if comm.rank == 0:
        winner_rank = int(np.argmin(np.asarray(d2_list, dtype=np.float64)))
        winner_d2 = float(d2_list[winner_rank])
        winner_v = int(v_list[winner_rank])
    else:
        winner_rank = 0
        winner_d2 = 0.0
        winner_v = -1

    winner_rank = comm.bcast(winner_rank, root=0)
    winner_d2 = comm.bcast(winner_d2, root=0)
    winner_v = comm.bcast(winner_v, root=0)

    if not np.isfinite(winner_d2) or winner_d2 > float(tol) ** 2:
        raise RuntimeError(
            f"Could not locate inlet mesh vertex for VTK point_id={inlet_point_id}. "
            f"Best squared distance={winner_d2:.3e}, tol^2={(tol**2):.3e}."
        )

    # Endpoints: owned vertices with exactly one incident cell
    endpoints = np.asarray([v for v in range(n_owned) if len(v2c.links(v)) == 1], dtype=np.int32)

    # On the winning rank, ensure the inlet is indeed an endpoint
    if comm.rank == winner_rank:
        if winner_v < 0 or not np.any(endpoints == winner_v):
            raise AssertionError(
                f"VTK point_id={inlet_point_id} matched mesh vertex {winner_v}, but it is not an endpoint."
            )

    inlet_marker = int(network.inlet_marker)
    outlet_marker = int(network.outlet_marker)

    idx = endpoints
    val = np.full((idx.size,), outlet_marker, dtype=np.int32)
    if comm.rank == winner_rank and idx.size:
        val[idx == winner_v] = inlet_marker

    tags = dmesh.meshtags(mesh, 0, idx, val)
    try:
        tags.name = "boundaries"
    except Exception:
        pass
    network.boundaries = tags


def build_cell_radius_from_vtk_point_radius(
        comm: MPI.Comm,
        network: Domain1D,
        vtk_points: np.ndarray,
        vtk_point_radius: np.ndarray,
        *,
        decimals: int = 8,
        nearest_tol: float = 1e-8,
) -> fem.Function:
    """
    Build DG0 cell radius for the network mesh by averaging endpoint VTK point radii:
      r_cell(cell) = 0.5 * (r(v0) + r(v1))

    Assumes network mesh vertices coincide with VTK points (best with points_per_edge=1).
    """
    mesh = network.mesh
    if mesh.topology.dim != 1:
        raise RuntimeError("Expected a 1D network mesh.")

    coord2r = build_coord_radius_lookup(vtk_points, vtk_point_radius, decimals=decimals)

    pts = vtk_points.astype(np.float64, copy=False)
    rad = vtk_point_radius.astype(np.float64, copy=False)

    def key_of(p: np.ndarray) -> tuple[float, float, float]:
        q = np.round(p, decimals=decimals)
        return (float(q[0]), float(q[1]), float(q[2]))

    def radius_at_point(p: np.ndarray) -> float:
        k = key_of(p)
        if k in coord2r:
            return float(coord2r[k])
        # fallback: nearest VTK point
        d2 = np.sum((pts - p[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        if float(d2[j]) > float(nearest_tol) ** 2:
            raise RuntimeError(f"Failed to map mesh point {p} to any VTK point (min d^2={float(d2[j]):.3e}).")
        return float(rad[j])

    tdim = 1
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)

    DG0 = fem.functionspace(mesh, ("DG", 0))
    r_cell = fem.Function(DG0)
    r_cell.name = "radius_cell"

    cell_map = mesh.topology.index_map(tdim)
    n_cells_local = int(cell_map.size_local)

    x = mesh.geometry.x

    for c in range(n_cells_local):
        dof = int(DG0.dofmap.cell_dofs(c)[0])
        vs = c2v.links(c)
        if len(vs) != 2:
            raise RuntimeError(f"Expected interval cell with 2 vertices, got {len(vs)}.")
        r0 = radius_at_point(x[int(vs[0])])
        r1 = radius_at_point(x[int(vs[1])])
        r_cell.x.array[dof] = 0.5 * (r0 + r1)

    r_cell.x.scatter_forward()
    return r_cell


def _make_timestamped_outdir(comm: MPI.Comm, base_dir: Path) -> Path:
    """
    Create results/YYYYmmdd_HHMMSS (or similar) only on rank 0, then broadcast.
    """
    outdir_str = None
    if comm.rank == 0:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = base_dir / f"results_{stamp}"
        outdir.mkdir(parents=True, exist_ok=False)
        outdir_str = str(outdir)
    outdir_str = comm.bcast(outdir_str, root=0)
    return Path(outdir_str)


def main():
    comm = MPI.COMM_WORLD
    setup_mpi_debug(comm)
    rprint = make_rank_logger(comm)

    # --- inputs ---
    vtk_path = find_vtk_path(REPO_ROOT, stem="dataNew")
    nifti_path = find_nifti_path(REPO_ROOT, stem="newData")

    if comm.rank == 0:
        rprint(f"Reading VTK:   {vtk_path}")
        rprint(f"Reading NIfTI: {nifti_path}")

    # --- NIfTI bounds (rank 0 -> broadcast) ---
    min_corner = None
    max_corner = None
    spacing = None
    if comm.rank == 0:
        min_corner, max_corner, spacing = nifti_world_bounds_from_nonzero(nifti_path, volume=0)
        rprint(f"NIfTI bounds (world AABB):")
        rprint(f"  min_corner = {min_corner.tolist()}")
        rprint(f"  max_corner = {max_corner.tolist()}")
        rprint(f"  voxel_spacing ~= {spacing.tolist()}")

    min_corner = comm.bcast(min_corner, root=0)
    max_corner = comm.bcast(max_corner, root=0)
    spacing = comm.bcast(spacing, root=0)

    min_corner = np.asarray(min_corner, dtype=np.float64)
    max_corner = np.asarray(max_corner, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)

    extent = np.maximum(max_corner - min_corner, 0.0)
    extent_max = float(np.max(extent)) if extent.size else 1.0
    min_spacing = float(np.min(spacing)) if spacing.size else max(1e-6, extent_max / 100.0)

    # Allow override from env var for reproducibility / speed control
    target_h_env = os.environ.get("TISSUE_TARGET_H", "").strip()
    if target_h_env:
        target_h = float(target_h_env)
    else:
        # Heuristic: not smaller than ~2 voxels, and not too coarse relative to the box
        target_h = max(2.0 * min_spacing, extent_max / 40.0)

    if comm.rank == 0:
        rprint(f"Tissue mesh target_h = {target_h:g}  (set TISSUE_TARGET_H to override)")

    # --- network input (VTK -> Domain1D) ---
    vtk_points = None
    vtk_rad = None
    if comm.rank == 0:
        vtk_points, vtk_rad = read_vtk_points_and_point_radius_legacy_ascii(vtk_path, radius_name="Radius")
    vtk_points = comm.bcast(vtk_points, root=0)
    vtk_rad = comm.bcast(vtk_rad, root=0)
    vtk_points = np.asarray(vtk_points, dtype=np.float64)
    vtk_rad = np.asarray(vtk_rad, dtype=np.float64)

    network = Domain1D.from_vtk_polydata(
        comm=comm,
        path=vtk_path,
        points_per_edge=1,
        graph_rank=0,
        color_strategy="largest_first",
        radius_name="Radius",
        default_radius=1.0,
        reverse_edges=False,
        strict_if_grouped=False,
    )

    # Set inlet; all other endpoints are sinks/outlets
    override_inlet_and_sinks_by_vtk_point_id(comm, network, vtk_points, inlet_point_id=89, tol=1e-8)

    # Build explicit DG0 cell radius from VTK point radii
    r_cell = build_cell_radius_from_vtk_point_radius(comm, network, vtk_points, vtk_rad, decimals=8, nearest_tol=1e-8)

    # --- tissue mesh: box from NIfTI bounds ---
    tissue = Domain3D.from_box(
        comm=comm,
        min_corner=min_corner,
        max_corner=max_corner,
        target_h=float(target_h),
    )

    # --- sink selection: NIfTI marker 2 voxels ---
    #
    # This uses the new Domain3D feature:
    #   tissue.mark_outlet_from_nifti(...)
    #
    # Because the tissue mesh was created in the NIfTI world coordinate system
    # (using the affine-derived bounds), we do not provide mesh_to_world.
    if comm.rank == 0:
        rprint("Marking sink/outlet boundary facets from NIfTI label=2 ...")
    marked_facets = tissue.mark_outlet_from_nifti(nifti_path, marker=2, override=True)

    # In parallel it's normal that some ranks have 0 marked facets; print global count
    n_marked_global = comm.allreduce(int(marked_facets.size), op=MPI.SUM)
    if comm.rank == 0:
        rprint(f"Sink/outlet facets marked (global): {n_marked_global}")

    assembly = AssemblyOptions(
        degree_3d=1,
        degree_1d=1,
        circle_quadrature_degree=6,
        degree_velocity=1,
    )
    solver = SolverOptions(
        petsc_options_prefix="pressure_run",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
    )

    # ---- timestamped output directory ----
    outdir = _make_timestamped_outdir(comm, REPO_ROOT / "results")
    if comm.rank == 0:
        rprint(f"Writing outputs to: {outdir}")

    if comm.rank == 0:
        rprint("Solving coupled PressureProblem...")

    with PressureProblem(
            tissue=tissue,
            network=network,
            params=Parameters(),
            assembly=assembly,
            solver=solver,
            cell_radius=r_cell,
            radius_by_tag=None,
            default_radius=1.0,
    ) as problem:
        sol = problem.solve()

        # Diagnostics
        sol.tissue_pressure.x.scatter_forward()
        sol.network_pressure.x.scatter_forward()

        p3 = np.asarray(sol.tissue_pressure.x.array, dtype=np.float64)
        p1 = np.asarray(sol.network_pressure.x.array, dtype=np.float64)

        local_min3 = float(np.min(p3)) if p3.size else float("inf")
        local_max3 = float(np.max(p3)) if p3.size else float("-inf")
        local_min1 = float(np.min(p1)) if p1.size else float("inf")
        local_max1 = float(np.max(p1)) if p1.size else float("-inf")

        gmin3 = comm.allreduce(local_min3, op=MPI.MIN)
        gmax3 = comm.allreduce(local_max3, op=MPI.MAX)
        gmin1 = comm.allreduce(local_min1, op=MPI.MIN)
        gmax1 = comm.allreduce(local_max1, op=MPI.MAX)

        if comm.rank == 0:
            rprint(f"Tissue pressure range:  [{gmin3:.6e}, {gmax3:.6e}]")
            rprint(f"Network pressure range: [{gmin1:.6e}, {gmax1:.6e}]")

        # ---- write results into timestamp folder ----
        write_solution(
            outdir=outdir,
            tissue=tissue,
            network=network,
            solution=sol,
            options=OutputOptions(format="vtk", time=0.0, write_meshtags=True),
        )

        sol.release()

    collect()
    if comm.rank == 0:
        rprint("Done.")


if __name__ == "__main__":
    main()
