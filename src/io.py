from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dolfinx.io as dolfinx_io
from mpi4py import MPI

from .core import PressureSolution, PressureVelocitySolution
from .domains import Domain1D, Domain3D


@dataclass(frozen=True, slots=True)
class OutputOptions:
    format: str = "xdmf"  # "xdmf" | "vtk" | "vtx"
    time: float = 0.0
    write_meshtags: bool = True


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

    # Collect network-side fields
    network_fields: list[Any] = [solution.network_pressure]
    if solution.cell_radius is not None:
        network_fields.append(solution.cell_radius)
    if solution.vertex_radius is not None:
        network_fields.append(solution.vertex_radius)

    # Collect tissue-side fields
    tissue_fields: list[Any] = [solution.tissue_pressure]
    if isinstance(solution, PressureVelocitySolution) and solution.tissue_velocity is not None:
        tissue_fields.append(solution.tissue_velocity)

    if fmt == "vtx":
        with dolfinx_io.VTXWriter(tissue.mesh.comm, outdir / "tissue.bp", tissue_fields) as vtx:
            vtx.write(options.time)
        with dolfinx_io.VTXWriter(network.mesh.comm, outdir / "network.bp", network_fields) as vtx:
            vtx.write(options.time)

    elif fmt == "vtk":
        with dolfinx_io.VTKFile(tissue.mesh.comm, outdir / "tissue.pvd", "w") as vtk:
            vtk.write_mesh(tissue.mesh, options.time)
            for f in tissue_fields:
                vtk.write_function(f, options.time)

        with dolfinx_io.VTKFile(network.mesh.comm, outdir / "network.pvd", "w") as vtk:
            vtk.write_mesh(network.mesh, options.time)
            for f in network_fields:
                vtk.write_function(f, options.time)

    elif fmt == "xdmf":
        with dolfinx_io.XDMFFile(tissue.mesh.comm, outdir / "tissue.xdmf", "w") as xdmf:
            xdmf.write_mesh(tissue.mesh)
            for f in tissue_fields:
                xdmf.write_function(f, options.time)

        with dolfinx_io.XDMFFile(network.mesh.comm, outdir / "network.xdmf", "w") as xdmf:
            xdmf.write_mesh(network.mesh)
            for f in network_fields:
                xdmf.write_function(f, options.time)

            if options.write_meshtags:
                # Write tags best-effort
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
