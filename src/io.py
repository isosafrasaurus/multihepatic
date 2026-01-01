from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dolfinx.io as dolfinx_io

from .domain import Domain1D, Domain3D
from .solutions import PressureSolution, PressureVelocitySolution


@dataclass(frozen=True, slots=True)
class OutputNames:
    tissue_pressure: str = "p_t"  # matches your script
    tissue_velocity: str = "v_tissue"  # if present
    network: str = "network"  # matches your script
    network_vtx: str = "P"  # matches your script (P.bp)


@dataclass(frozen=True, slots=True)
class OutputOptions:
    format: str = "xdmf"  # "xdmf" | "vtk" | "vtx"
    time: float = 0.0
    write_meshtags: bool = True
    names: OutputNames = OutputNames()


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
    names = options.names

    tissue_pressure = solution.tissue_pressure
    network_fields: list[Any] = [solution.network_pressure]
    if solution.cell_radius is not None:
        network_fields.append(solution.cell_radius)
    if solution.vertex_radius is not None:
        network_fields.append(solution.vertex_radius)

    tissue_velocity: Any | None = None
    if isinstance(solution, PressureVelocitySolution):
        tissue_velocity = solution.tissue_velocity

    if fmt == "vtx":
        with dolfinx_io.VTXWriter(tissue.mesh.comm, outdir / f"{names.tissue_pressure}.bp", [tissue_pressure]) as vtx:
            vtx.write(options.time)

        if tissue_velocity is not None:
            with dolfinx_io.VTXWriter(tissue.mesh.comm, outdir / f"{names.tissue_velocity}.bp",
                                      [tissue_velocity]) as vtx:
                vtx.write(options.time)

        with dolfinx_io.VTXWriter(network.mesh.comm, outdir / f"{names.network_vtx}.bp", network_fields) as vtx:
            vtx.write(options.time)

    elif fmt == "vtk":
        with dolfinx_io.VTKFile(tissue.mesh.comm, outdir / f"{names.tissue_pressure}.pvd", "w") as vtk:
            vtk.write_mesh(tissue.mesh, options.time)
            vtk.write_function(tissue_pressure, options.time)

        if tissue_velocity is not None:
            with dolfinx_io.VTKFile(tissue.mesh.comm, outdir / f"{names.tissue_velocity}.pvd", "w") as vtk:
                vtk.write_mesh(tissue.mesh, options.time)
                vtk.write_function(tissue_velocity, options.time)

        with dolfinx_io.VTKFile(network.mesh.comm, outdir / f"{names.network}.pvd", "w") as vtk:
            vtk.write_mesh(network.mesh, options.time)
            for f in network_fields:
                vtk.write_function(f, options.time)

    elif fmt == "xdmf":
        with dolfinx_io.XDMFFile(tissue.mesh.comm, outdir / f"{names.tissue_pressure}.xdmf", "w") as xdmf:
            xdmf.write_mesh(tissue.mesh)
            xdmf.write_function(tissue_pressure, options.time)

        if tissue_velocity is not None:
            with dolfinx_io.XDMFFile(tissue.mesh.comm, outdir / f"{names.tissue_velocity}.xdmf", "w") as xdmf:
                xdmf.write_mesh(tissue.mesh)
                xdmf.write_function(tissue_velocity, options.time)

        with dolfinx_io.XDMFFile(network.mesh.comm, outdir / f"{names.network}.xdmf", "w") as xdmf:
            xdmf.write_mesh(network.mesh)
            for f in network_fields:
                xdmf.write_function(f, options.time)

            if options.write_meshtags:
                try:
                    xdmf.write_meshtags(network.subdomains, network.mesh.geometry)  # type: ignore[arg-type]
                except Exception:
                    pass
                try:
                    xdmf.write_meshtags(network.boundaries, network.mesh.geometry)
                except Exception:
                    pass

    else:
        raise ValueError(f"Unknown format={options.format!r}. Use 'xdmf', 'vtk', or 'vtx'.")
