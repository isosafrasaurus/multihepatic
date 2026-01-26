# === io.py ===
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dolfinx.io as dolfinx_io
import numpy as np
from dolfinx import default_scalar_type, fem

from .domain import Domain1D, Domain3D
from .problem import PressureSolution, PressureVelocitySolution


@dataclass(frozen=True, slots=True)
class OutputNames:
    tissue_pressure: str = "p_t"
    tissue_velocity: str = "v_tissue"
    network: str = "network"
    network_vtx: str = "P"


@dataclass(frozen=True, slots=True)
class OutputOptions:
    format: str = "xdmf"  # "xdmf" | "vtk" | "vtx"
    time: float = 0.0
    write_meshtags: bool = True
    names: OutputNames = OutputNames()


def _vtk_write_functions(vtk: dolfinx_io.VTKFile, funcs: list[Any], t: float) -> None:
    """
    Write functions to a VTKFile in a way that is robust across dolfinx versions.

    Critical detail:
      - Avoid calling write_mesh(...) first. In some dolfinx builds this produces an
        extra PVD entry that references a dataset file that is never written
        (e.g. p_t000000.pvtu), which then breaks ParaView.
      - Prefer a single write_function([...], t) call so ParaView sees all arrays in
        one dataset and the PVD references only files that actually exist.
    """
    if len(funcs) == 0:
        return

    # Some versions are happier with a single Function than with a 1-long list.
    try:
        if len(funcs) == 1:
            vtk.write_function(funcs[0], t)
        else:
            vtk.write_function(funcs, t)
    except TypeError:
        # Fallback: write individually (still no write_mesh)
        for f in funcs:
            vtk.write_function(f, t)


def write_tissue_bc_label_mesh(
        outdir: Path,
        tissue: Domain3D,
        *,
        time: float = 0.0,
        basename: str = "tissue_bc_labels",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    mesh = tissue.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    DG0 = fem.functionspace(mesh, ("DG", 0))
    bc_type = fem.Function(DG0)
    bc_type.name = "bc_type"
    facet_marker = fem.Function(DG0)
    facet_marker.name = "facet_marker"

    bc_type.x.array[:] = default_scalar_type(0.0)
    facet_marker.x.array[:] = default_scalar_type(0.0)

    if tissue.boundaries is not None:
        mesh.topology.create_connectivity(fdim, tdim)
        f2c = mesh.topology.connectivity(fdim, tdim)

        outlet = getattr(tissue, "outlet_marker", None)

        for facet, tag in zip(tissue.boundaries.indices, tissue.boundaries.values):
            tag_i = int(tag)
            for cell in f2c.links(int(facet)):
                dof = int(DG0.dofmap.cell_dofs(int(cell))[0])
                facet_marker.x.array[dof] = default_scalar_type(tag_i)
                if outlet is not None and tag_i == int(outlet):
                    bc_type.x.array[dof] = default_scalar_type(1.0)

    bc_type.x.scatter_forward()
    facet_marker.x.scatter_forward()

    with dolfinx_io.XDMFFile(mesh.comm, str(outdir / f"{basename}.xdmf"), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(bc_type, time)
        xdmf.write_function(facet_marker, time)
        if tissue.boundaries is not None:
            # Some dolfinx versions require geometry passed explicitly
            xdmf.write_meshtags(tissue.boundaries, mesh.geometry)  # type: ignore[arg-type]


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
    tissue_fields: list[Any] = [tissue_pressure]

    tissue_velocity: Any | None = None
    if isinstance(solution, PressureVelocitySolution):
        tissue_velocity = solution.tissue_velocity
        if tissue_velocity is not None:
            tissue_fields.append(tissue_velocity)

    network_fields: list[Any] = [solution.network_pressure]

    if fmt == "vtx":
        # Write tissue pressure + velocity together so ParaView sees both arrays in one dataset
        with dolfinx_io.VTXWriter(
                tissue.mesh.comm,
                str(outdir / f"{names.tissue_pressure}.bp"),
                tissue_fields,
        ) as vtx:
            vtx.write(options.time)

        with dolfinx_io.VTXWriter(
                network.mesh.comm,
                str(outdir / f"{names.network_vtx}.bp"),
                network_fields,
        ) as vtx:
            vtx.write(options.time)

    elif fmt == "vtk":
        # IMPORTANT (fix):
        # Do NOT call vtk.write_mesh(...) first.
        #
        # Reason: depending on dolfinx version/build, write_mesh can create an
        # additional PVD entry referencing e.g. p_t000000.pvtu that is not actually
        # written, which makes ParaView fail with:
        #   vtkXMLPUnstructuredGridReader: Error opening file .../p_t000000.pvtu
        #
        # Instead, write functions directly; dolfinx will include the mesh in the
        # dataset it writes, and the PVD will reference only existing files.

        # Tissue fields
        tissue_fields_out: list[Any] = []
        try:
            tissue_pressure.name = names.tissue_pressure
        except Exception:
            pass
        try:
            tissue_pressure.x.scatter_forward()
        except Exception:
            pass
        tissue_fields_out.append(tissue_pressure)

        if tissue_velocity is not None:
            try:
                tissue_velocity.name = names.tissue_velocity
            except Exception:
                pass
            try:
                tissue_velocity.x.scatter_forward()
            except Exception:
                pass
            tissue_fields_out.append(tissue_velocity)

        tissue_pvd = outdir / f"{names.tissue_pressure}.pvd"
        with dolfinx_io.VTKFile(tissue.mesh.comm, str(tissue_pvd), "w") as vtk:
            _vtk_write_functions(vtk, tissue_fields_out, options.time)

        # Make sure all ranks have finished writing before moving on
        try:
            tissue.mesh.comm.Barrier()
        except Exception:
            pass

        # Network fields
        for f in network_fields:
            try:
                f.name = names.network_vtx
            except Exception:
                pass
            try:
                f.x.scatter_forward()
            except Exception:
                pass

        network_pvd = outdir / f"{names.network}.pvd"
        with dolfinx_io.VTKFile(network.mesh.comm, str(network_pvd), "w") as vtk:
            _vtk_write_functions(vtk, network_fields, options.time)

        try:
            network.mesh.comm.Barrier()
        except Exception:
            pass

    elif fmt == "xdmf":
        # Tissue: pressure + velocity in ONE .xdmf
        with dolfinx_io.XDMFFile(tissue.mesh.comm, str(outdir / f"{names.tissue_pressure}.xdmf"), "w") as xdmf:
            xdmf.write_mesh(tissue.mesh)
            for f in tissue_fields:
                xdmf.write_function(f, options.time)

            if options.write_meshtags:
                try:
                    if getattr(tissue, "boundaries", None) is not None:
                        xdmf.write_meshtags(tissue.boundaries, tissue.mesh.geometry)  # type: ignore[arg-type]
                except Exception:
                    pass

        with dolfinx_io.XDMFFile(network.mesh.comm, str(outdir / f"{names.network}.xdmf"), "w") as xdmf:
            xdmf.write_mesh(network.mesh)
            for f in network_fields:
                xdmf.write_function(f, options.time)

            if options.write_meshtags:
                try:
                    if network.subdomains is not None:
                        xdmf.write_meshtags(network.subdomains, network.mesh.geometry)  # type: ignore[arg-type]
                except Exception:
                    pass
                try:
                    xdmf.write_meshtags(network.boundaries, network.mesh.geometry)
                except Exception:
                    pass

    else:
        raise ValueError(f"Unknown format={options.format!r}. Use 'xdmf', 'vtk', or 'vtx'.")

    # Keep this as-is: it writes a small XDMF helper mesh for debugging BC regions
    write_tissue_bc_label_mesh(outdir, tissue, time=options.time)
