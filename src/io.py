from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dolfinx.io as dolfinx_io
import numpy as np
from dolfinx import default_scalar_type, fem
from dolfinx import mesh as dmesh

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


def _write_tissue_vtk_meshio(
    filepath: Path,
    tissue_mesh: Any,
    *,
    time: float,
    point_data: dict[str, np.ndarray],
) -> None:
    import meshio

    x = tissue_mesh.geometry.x
    tdim = tissue_mesh.topology.dim
    tissue_mesh.topology.create_connectivity(tdim, 0)
    tissue_mesh.topology.create_connectivity(tdim, tdim)

    # Extract cell connectivity for the topological dimension
    topology = tissue_mesh.topology
    c_map = topology.index_map(tdim)
    num_cells = int(c_map.size_local)

    topology.create_connectivity(tdim, 0)
    c2v = topology.connectivity(tdim, 0)
    cells = np.vstack([c2v.links(c) for c in range(num_cells)]).astype(np.int64)

    # Map DOLFINx cell type -> meshio cell type name
    ct = tissue_mesh.topology.cell_type
    if ct == dmesh.CellType.tetrahedron:
        cell_type = "tetra"
    elif ct == dmesh.CellType.hexahedron:
        cell_type = "hexahedron"
    elif ct == dmesh.CellType.triangle:
        cell_type = "triangle"
    elif ct == dmesh.CellType.quadrilateral:
        cell_type = "quad"
    else:
        raise ValueError(f"Unsupported cell type for meshio writer: {ct}")

    mesh = meshio.Mesh(
        points=x,
        cells=[(cell_type, cells)],
        point_data=point_data,
    )

    # Write VTU
    vtu_path = filepath.with_suffix(".vtu")
    meshio.write(vtu_path, mesh)

    # Write minimal PVD that points to the VTU
    pvd_path = filepath.with_suffix(".pvd")
    pvd_xml = f"""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
    <DataSet timestep="{time}" group="" part="0" file="{vtu_path.name}"/>
  </Collection>
</VTKFile>
"""
    pvd_path.write_text(pvd_xml)


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

    with dolfinx_io.XDMFFile(mesh.comm, outdir / f"{basename}.xdmf", "w") as xdmf:
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
            outdir / f"{names.tissue_pressure}.bp",
            tissue_fields,
        ) as vtx:
            vtx.write(options.time)

        with dolfinx_io.VTXWriter(
            network.mesh.comm,
            outdir / f"{names.network_vtx}.bp",
            network_fields,
        ) as vtx:
            vtx.write(options.time)

    elif fmt == "vtk":
        p = tissue_pressure
        p.x.scatter_forward()

        point_data: dict[str, np.ndarray] = {}

        Vp = p.function_space
        xp = Vp.element.interpolation_points
        p_expr = fem.Expression(p, xp)
        p_at_pts = fem.Function(Vp)
        p_at_pts.interpolate(p_expr)
        p_at_pts.x.scatter_forward()

        point_data[names.tissue_pressure] = np.asarray(p_at_pts.x.array, dtype=np.float64)

        if tissue_velocity is not None:
            tissue_velocity.x.scatter_forward()

            Vv = tissue_velocity.function_space
            xv = Vv.element.interpolation_points
            v_expr = fem.Expression(tissue_velocity, xv)
            v_at_pts = fem.Function(Vv)
            v_at_pts.interpolate(v_expr)
            v_at_pts.x.scatter_forward()

            # v_at_pts.x.array is flattened; reshape to (num_points, gdim)
            gdim = tissue.mesh.geometry.dim
            v_arr = np.asarray(v_at_pts.x.array, dtype=np.float64).reshape((-1, gdim))
            point_data[names.tissue_velocity] = v_arr

        _write_tissue_vtk_meshio(
            outdir / f"{names.tissue_pressure}",  # will create .vtu + .pvd
            tissue.mesh,
            time=options.time,
            point_data=point_data,
        )

        with dolfinx_io.VTKFile(network.mesh.comm, outdir / f"{names.network}.pvd", "w") as vtk:
            vtk.write_mesh(network.mesh, options.time)
            for f in network_fields:
                vtk.write_function(f, options.time)

    elif fmt == "xdmf":
        # Tissue: pressure + velocity in ONE .xdmf
        with dolfinx_io.XDMFFile(tissue.mesh.comm, outdir / f"{names.tissue_pressure}.xdmf", "w") as xdmf:
            xdmf.write_mesh(tissue.mesh)
            for f in tissue_fields:
                xdmf.write_function(f, options.time)

            if options.write_meshtags:
                try:
                    if getattr(tissue, "boundaries", None) is not None:
                        xdmf.write_meshtags(tissue.boundaries, tissue.mesh.geometry)  # type: ignore[arg-type]
                except Exception:
                    pass

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

    write_tissue_bc_label_mesh(outdir, tissue, time=options.time)
