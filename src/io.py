from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dolfinx.io as dolfinx_io
import numpy as np
from mpi4py import MPI

from .core import PressureSolution, PressureVelocitySolution
from .domains import Domain1D, Domain3D


@dataclass(frozen=True, slots=True)
class OutputOptions:
    format: str = "xdmf"  # "xdmf" | "vtk" | "vtx"
    time: float = 0.0
    write_meshtags: bool = True

    write_network_tube: bool = False
    network_tube_basename: str = "network_tube"  # produces network_tube.pvd/.pvtp/.vtp


def _fmt_floats(a: np.ndarray) -> str:
    # Compact ASCII float formatting for XML
    return " ".join(f"{x:.17g}" for x in a.reshape(-1))


def _fmt_ints(a: np.ndarray) -> str:
    return " ".join(str(int(x)) for x in a.reshape(-1))


def _write_vtp_ascii(
        path: Path,
        points_xyz: np.ndarray,  # (N, 3) float64
        connectivity: np.ndarray,  # (sum(npts_cell),) int32
        offsets: np.ndarray,  # (num_cells_local,) int32
        point_data: dict[str, np.ndarray],  # name -> (N,) or (N, ncomp)
) -> None:
    N = int(points_xyz.shape[0])
    M = int(offsets.shape[0])

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <PolyData>\n")
        f.write(f'    <Piece NumberOfPoints="{N}" NumberOfLines="{M}">\n')

        # Points
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write("          " + _fmt_floats(points_xyz) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        # Lines
        f.write("      <Lines>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + _fmt_ints(connectivity.astype(np.int32, copy=False)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + _fmt_ints(offsets.astype(np.int32, copy=False)) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Lines>\n")

        # PointData
        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            if arr.ndim == 1:
                ncomp = 1
                flat = arr
            else:
                ncomp = int(arr.shape[1])
                flat = arr.reshape(-1)
            f.write(
                f'        <DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            f.write("          " + _fmt_floats(flat.astype(np.float64, copy=False)) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")

        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def _write_pvtp(
        path: Path,
        piece_filenames: list[str],
        point_arrays: list[tuple[str, int]],  # (name, ncomp)
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PPolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PPolyData GhostLevel="0">\n')

        # Points declaration
        f.write("    <PPoints>\n")
        f.write('      <PDataArray type="Float64" NumberOfComponents="3"/>\n')
        f.write("    </PPoints>\n")

        # PointData declaration
        f.write("    <PPointData>\n")
        for name, ncomp in point_arrays:
            f.write(
                f'      <PDataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{int(ncomp)}"/>\n'
            )
        f.write("    </PPointData>\n")

        # Lines declaration
        f.write("    <PLines>\n")
        f.write('      <PDataArray type="Int32" Name="connectivity"/>\n')
        f.write('      <PDataArray type="Int32" Name="offsets"/>\n')
        f.write("    </PLines>\n")

        for src in piece_filenames:
            f.write(f'    <Piece Source="{src}"/>\n')

        f.write("  </PPolyData>\n")
        f.write("</VTKFile>\n")


def _write_pvd_single_entry(path: Path, timestep: float, file_rel: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write("  <Collection>\n")
        f.write(f'    <DataSet timestep="{timestep:.17g}" part="0" file="{file_rel}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


def _local_polyline_piece_from_mesh(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tdim = mesh.topology.dim
    if tdim != 1:
        raise ValueError(f"Tube output expects a 1D mesh, got tdim={tdim}")

    # Owned cells only
    cell_map = mesh.topology.index_map(tdim)
    num_cells_local = int(cell_map.size_local)

    # Geometry dofmap gives the point indices for each cell in mesh.geometry.x
    # Docs note: cell_geometry can be built via geometry.x[geometry.dofmap.cell_dofs(i)]. :contentReference[oaicite:1]{index=1}
    geom_dofmap = np.asarray(mesh.geometry.dofmap[:num_cells_local], dtype=np.int32)
    if num_cells_local == 0:
        # Empty piece (still valid)
        points_xyz = np.zeros((0, 3), dtype=np.float64)
        connectivity = np.zeros((0,), dtype=np.int32)
        offsets = np.zeros((0,), dtype=np.int32)
        used_gdofs = np.zeros((0,), dtype=np.int32)
        rep_cell = np.zeros((0,), dtype=np.int32)
        return points_xyz, connectivity, offsets, used_gdofs, rep_cell

    npts_cell = int(geom_dofmap.shape[1])

    used_gdofs = np.unique(geom_dofmap.reshape(-1))
    # Map old geometry indices -> new contiguous point indices for this piece
    old_to_new = -np.ones((mesh.geometry.x.shape[0],), dtype=np.int32)
    old_to_new[used_gdofs] = np.arange(used_gdofs.size, dtype=np.int32)

    connectivity = old_to_new[geom_dofmap.reshape(-1)].astype(np.int32, copy=False)
    offsets = (np.arange(1, num_cells_local + 1, dtype=np.int32) * npts_cell).astype(np.int32, copy=False)

    coords = np.asarray(mesh.geometry.x[used_gdofs], dtype=np.float64)
    points_xyz = np.zeros((coords.shape[0], 3), dtype=np.float64)
    points_xyz[:, : coords.shape[1]] = coords  # pad z=0 if needed

    # Representative cell index for each used point
    rep_cell_all = -np.ones((mesh.geometry.x.shape[0],), dtype=np.int32)
    for c in range(num_cells_local):
        for gdof in geom_dofmap[c]:
            if rep_cell_all[gdof] < 0:
                rep_cell_all[gdof] = c
    rep_cell_for_used = rep_cell_all[used_gdofs].astype(np.int32, copy=False)

    return points_xyz, connectivity, offsets, used_gdofs, rep_cell_for_used


def _eval_function_on_piece_points(func, points_xyz: np.ndarray, rep_cells: np.ndarray) -> np.ndarray:
    """
    Evaluate a dolfinx.fem.Function on the piece points.

    DOLFINx expects x shaped (num_points, 3) and cells shaped (num_points,).
    Returns array shaped (num_points, value_size). :contentReference[oaicite:2]{index=2}
    """
    if points_xyz.shape[0] == 0:
        # value_size unknown without basix, just return empty scalar
        return np.zeros((0, 1), dtype=np.float64)
    vals = func.eval(points_xyz, rep_cells)  # type: ignore[attr-defined]
    vals = np.asarray(vals)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)
    return vals


def _point_radius_from_cell_radius(mesh, cell_radius_func, used_gdofs: np.ndarray) -> np.ndarray:
    """
    Build pointwise 'radius' by averaging DG0 cell radii onto the geometry points
    referenced by owned cells on this rank.
    """
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    num_cells_local = int(cell_map.size_local)

    if num_cells_local == 0 or used_gdofs.size == 0:
        return np.zeros((used_gdofs.size,), dtype=np.float64)

    geom_dofmap = np.asarray(mesh.geometry.dofmap[:num_cells_local], dtype=np.int32)

    sums = np.zeros((mesh.geometry.x.shape[0],), dtype=np.float64)
    cnts = np.zeros((mesh.geometry.x.shape[0],), dtype=np.int32)

    dm = cell_radius_func.function_space.dofmap
    for c in range(num_cells_local):
        dof = int(dm.cell_dofs(c)[0])
        r = float(cell_radius_func.x.array[dof])
        for gdof in geom_dofmap[c]:
            sums[gdof] += r
            cnts[gdof] += 1

    out = np.zeros((mesh.geometry.x.shape[0],), dtype=np.float64)
    mask = cnts > 0
    out[mask] = sums[mask] / cnts[mask]
    return out[used_gdofs]


def write_network_tube_vtk(
        outdir: Path,
        network,
        *,
        fields: list[Any],
        cell_radius: Any | None,
        time: float,
        basename: str = "network_tube",
) -> None:
    comm: MPI.Comm = network.mesh.comm
    rank = comm.rank
    size = comm.size

    step = 0  # matches your current single-step workflow
    pvd_name = f"{basename}.pvd"
    pvtp_name = f"{basename}{step:06d}.pvtp"
    piece_name = f"{basename}{step:06d}_rank{rank:04d}.vtp"

    points_xyz, connectivity, offsets, used_gdofs, rep_cells = _local_polyline_piece_from_mesh(network.mesh)

    # Build point-data arrays
    point_data: dict[str, np.ndarray] = {}

    # Evaluate each requested field at the piece points
    point_arrays_meta: list[tuple[str, int]] = []
    for f in fields:
        vals = _eval_function_on_piece_points(f, points_xyz, rep_cells)
        # vals: (N, value_size)
        name = getattr(f, "name", "field")
        if vals.shape[1] == 1:
            point_data[name] = vals[:, 0].astype(np.float64, copy=False)
            point_arrays_meta.append((name, 1))
        else:
            point_data[name] = vals.astype(np.float64, copy=False)
            point_arrays_meta.append((name, int(vals.shape[1])))

    # Add tube radius as point scalar
    if cell_radius is not None:
        r = _point_radius_from_cell_radius(network.mesh, cell_radius, used_gdofs)
        point_data["radius"] = r.astype(np.float64, copy=False)
        point_arrays_meta.append(("radius", 1))

    # Write this rank's piece
    _write_vtp_ascii(outdir / piece_name, points_xyz, connectivity, offsets, point_data)

    # Rank 0 writes the .pvtp and .pvd
    comm.Barrier()
    if rank == 0:
        piece_files = [f"{basename}{step:06d}_rank{r:04d}.vtp" for r in range(size)]
        _write_pvtp(outdir / pvtp_name, piece_files, point_arrays_meta)
        _write_pvd_single_entry(outdir / pvd_name, time, pvtp_name)


def write_solution(
        outdir: Path,
        tissue: Domain3D,
        network: Domain1D,
        solution: PressureSolution | PressureVelocitySolution,
        *,
        options: OutputOptions = OutputOptions(),
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fmt = options.format.lower()

    network_fields: list[Any] = [solution.network_pressure]
    if solution.cell_radius is not None:
        network_fields.append(solution.cell_radius)
    if solution.vertex_radius is not None:
        network_fields.append(solution.vertex_radius)

    tissue_pressure_fields: list[Any] = [solution.tissue_pressure]

    tissue_velocity_field: Any | None = None
    if isinstance(solution, PressureVelocitySolution):
        tissue_velocity_field = solution.tissue_velocity

    if fmt == "vtx":
        # Pressure in its own BP file
        with dolfinx_io.VTXWriter(
                tissue.mesh.comm, outdir / "tissue_pressure.bp", tissue_pressure_fields
        ) as vtx:
            vtx.write(options.time)

        # Velocity in its own BP file
        if tissue_velocity_field is not None:
            with dolfinx_io.VTXWriter(
                    tissue.mesh.comm, outdir / "tissue_velocity.bp", [tissue_velocity_field]
            ) as vtx:
                vtx.write(options.time)

        # Network BP file
        with dolfinx_io.VTXWriter(network.mesh.comm, outdir / "network.bp", network_fields) as vtx:
            vtx.write(options.time)

    elif fmt == "vtk":
        tissue_fields = [solution.tissue_pressure]
        if isinstance(solution, PressureVelocitySolution) and solution.tissue_velocity is not None:
            tissue_fields.append(solution.tissue_velocity)

        with dolfinx_io.VTKFile(tissue.mesh.comm, outdir / "tissue.pvd", "w") as vtk:
            vtk.write_function(tissue_fields, options.time)

        with dolfinx_io.VTKFile(network.mesh.comm, outdir / "network.pvd", "w") as vtk:
            vtk.write_function(network_fields, options.time)

        if options.write_network_tube:
            tube_fields = [solution.network_pressure]
            write_network_tube_vtk(
                outdir,
                network,
                fields=tube_fields,
                cell_radius=solution.cell_radius,
                time=options.time,
                basename=options.network_tube_basename,
            )


    elif fmt == "xdmf":
        with dolfinx_io.XDMFFile(tissue.mesh.comm, outdir / "tissue_pressure.xdmf", "w") as xdmf:
            xdmf.write_mesh(tissue.mesh)
            for f in tissue_pressure_fields:
                xdmf.write_function(f, options.time)

        if tissue_velocity_field is not None:
            with dolfinx_io.XDMFFile(tissue.mesh.comm, outdir / "tissue_velocity.xdmf", "w") as xdmf:
                xdmf.write_mesh(tissue.mesh)
                xdmf.write_function(tissue_velocity_field, options.time)

        # Network in its own XDMF
        with dolfinx_io.XDMFFile(network.mesh.comm, outdir / "network.xdmf", "w") as xdmf:
            xdmf.write_mesh(network.mesh)
            for f in network_fields:
                xdmf.write_function(f, options.time)

            if options.write_meshtags:
                try:
                    xdmf.write_meshtags(network.boundaries, network.mesh.geometry)
                except Exception:
                    pass
                try:
                    if network.subdomains is not None:
                        xdmf.write_meshtags(network.subdomains, network.mesh.geometry)
                except Exception:
                    pass
    else:
        raise ValueError(f"Unknown format={options.format!r}. Use 'xdmf', 'vtk', or 'vtx'.")


def read_domain3d_xdmf(
        comm: MPI.Comm,
        path: Path,
        *,
        mesh_name: str = "Grid",
) -> Domain3D:
    with dolfinx_io.XDMFFile(comm, str(path), "r") as xdmf:
        mesh = xdmf.read_mesh(name=mesh_name)
    return Domain3D(mesh=mesh)


def read_domain1d_xdmf(
        comm: MPI.Comm,
        path: Path,
        *,
        inlet_marker: int,
        outlet_marker: int,
        mesh_name: str = "Grid",
        boundaries_name: str = "boundaries",
        subdomains_name: str = "subdomains",
) -> Domain1D:
    with dolfinx_io.XDMFFile(comm, str(path), "r") as xdmf:
        mesh = xdmf.read_mesh(name=mesh_name)
        boundaries = xdmf.read_meshtags(mesh, name=boundaries_name)
        try:
            subdomains = xdmf.read_meshtags(mesh, name=subdomains_name)
        except Exception:
            subdomains = None

    return Domain1D(
        mesh=mesh,
        boundaries=boundaries,
        subdomains=subdomains,
        inlet_marker=inlet_marker,
        outlet_marker=outlet_marker,
    )
