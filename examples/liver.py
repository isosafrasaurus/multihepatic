from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dolfinx.fem as fem
import dolfinx.io as io
import dolfinx.mesh as dmesh
import numpy as np
import ufl
from dolfinx import default_scalar_type
from fenicsx_ii import Average, Circle, LinearProblem, assemble_scalar
from mpi4py import MPI

TISSUE_VTK = Path(__file__).resolve().parent.parent / "data" / Path("nii2mesh_liver_mask.vtk")
NETWORK_VTK = Path(__file__).resolve().parent.parent / "data" / Path("sortedVesselNetwork.vtk")

# Treat this ORIGINAL VTK node index as inlet; all other endpoints become outlets
INLET_NODE_VTK_INDEX = 59

OUTPUT_FORMAT = "vtk"  # "vtk", "xdmf", or "vtx"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Network data names (used for radius only; inlet/outlet selection is forced by INLET_NODE_VTK_INDEX)
NETWORK_RADIUS_CELL_DATA = "radius"  # cell_data name
NETWORK_RADIUS_POINT_DATA: Optional[str] = None  # optional fallback point_data name

INLET_MARKER = 1
OUTLET_MARKER = 2
DEFAULT_RADIUS = 1.0e-3

DEGREE_3D = 1
DEGREE_1D = 1
CIRCLE_QUAD_DEGREE = 20


# -----------------------------------------------------------------------------
# Parameters (same as your original)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    gamma: float = 3.6145827741262347e-05
    gamma_a: float = 8.225197366649115e-08
    gamma_R: float = 8.620057937882969e-08
    mu: float = 1.0e-3
    k_t: float = 1.0e-10
    P_in: float = 100.0 * 133.322
    P_cvp: float = 1.0 * 133.322


# -----------------------------------------------------------------------------
# Utilities: VTK -> (meshio) -> XDMF -> dolfinx.Mesh
# -----------------------------------------------------------------------------

def _ensure_3d_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f"Expected points to have shape (N, gdim). Got {pts.shape}.")
    if pts.shape[1] == 3:
        return pts
    if pts.shape[1] == 2:
        return np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)])
    if pts.shape[1] == 1:
        return np.hstack([pts, np.zeros((pts.shape[0], 2), dtype=pts.dtype)])
    raise ValueError(f"Unsupported point dimension {pts.shape[1]}; expected 1, 2, or 3.")


def _pick_first_existing_celltype(cells_dict: dict, preferred: tuple[str, ...]) -> str:
    for ct in preferred:
        if ct in cells_dict and len(cells_dict[ct]) > 0:
            return ct
    available = ", ".join(sorted(cells_dict.keys()))
    raise ValueError(f"Could not find any of {preferred} in mesh. Available cell types: {available}")


def _polyline_to_lines(poly_cells: np.ndarray) -> np.ndarray:
    segments: list[tuple[int, int]] = []
    for row in poly_cells:
        chain = np.asarray(row, dtype=np.int64).ravel()
        if chain.size < 2:
            continue
        for i in range(chain.size - 1):
            segments.append((int(chain[i]), int(chain[i + 1])))
    if len(segments) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(segments, dtype=np.int64)


def _replicate_polyline_cell_data_to_segments(poly_cells: np.ndarray, poly_data: np.ndarray) -> np.ndarray:
    out: list[float] = []
    if len(poly_data) != len(poly_cells):
        raise ValueError("polyline cell_data length mismatch with number of polylines.")
    for row, val in zip(poly_cells, poly_data):
        chain = np.asarray(row, dtype=np.int64).ravel()
        nseg = max(0, chain.size - 1)
        out.extend([float(val)] * nseg)
    return np.asarray(out, dtype=np.float64)


def _write_xdmf_from_meshio(comm: MPI.Comm, meshio_mesh, xdmf_path: Path) -> None:
    # rank-0 writes, all ranks barrier
    if comm.rank == 0:
        xdmf_path.parent.mkdir(parents=True, exist_ok=True)
        import meshio
        meshio.write(str(xdmf_path), meshio_mesh)
    comm.barrier()


def _read_dolfinx_mesh_from_xdmf(
        comm: MPI.Comm,
        xdmf_path: Path,
        *,
        max_facet_to_cell_links: int,
) -> dmesh.Mesh:
    with io.XDMFFile(comm, xdmf_path, "r") as xdmf:
        try:
            return xdmf.read_mesh(name="mesh", max_facet_to_cell_links=max_facet_to_cell_links)
        except Exception:
            return xdmf.read_mesh(name="Grid", max_facet_to_cell_links=max_facet_to_cell_links)


# -----------------------------------------------------------------------------
# Network loading with forced inlet node (original VTK index)
# -----------------------------------------------------------------------------

@dataclass
class NetworkVTKData:
    n_points: int
    n_cells: int
    radius_global: np.ndarray  # (n_cells,)
    bmarker_global: np.ndarray  # (n_points,)
    max_facet_to_cell_links: int  # for branching 1D meshes


def load_network_from_vtk_forced_inlet(
        comm: MPI.Comm,
        network_vtk: Path,
        xdmf_path: Path,
        *,
        inlet_node_vtk_index: int,
        radius_cell_data: str = "radius",
        radius_point_data: Optional[str] = None,
        inlet_marker: int = 1,
        outlet_marker: int = 2,
        default_radius: float = 1.0e-3,
) -> tuple[dmesh.Mesh, dmesh.MeshTags, NetworkVTKData]:
    """
    Reads 1D VTK, forces inlet at a specified ORIGINAL-VTK node index, and marks
    all other endpoints as outlets. Non-endpoints get marker 0.
    """
    if inlet_marker == outlet_marker:
        raise ValueError("inlet_marker and outlet_marker must be different.")

    if comm.rank == 0:
        import meshio

        m = meshio.read(str(network_vtk))
        pts = _ensure_3d_points(m.points)

        cells_dict = m.cells_dict
        if "line" in cells_dict and len(cells_dict["line"]) > 0:
            line_cells = np.asarray(cells_dict["line"], dtype=np.int64)
            had_polyline = False
            poly_cells = None
        elif "polyline" in cells_dict and len(cells_dict["polyline"]) > 0:
            had_polyline = True
            poly_cells = cells_dict["polyline"]
            line_cells = _polyline_to_lines(poly_cells)
        else:
            raise ValueError(
                f"Network VTK must contain 'line' or 'polyline' cells. Found {list(cells_dict.keys())}"
            )

        n_points = pts.shape[0]
        n_cells = line_cells.shape[0]

        if inlet_node_vtk_index < 0 or inlet_node_vtk_index >= n_points:
            raise ValueError(
                f"inlet_node_vtk_index={inlet_node_vtk_index} out of range for VTK points [0, {n_points - 1}]"
            )

        # Degree / endpoints
        deg = np.zeros((n_points,), dtype=np.int32)
        if n_cells > 0:
            np.add.at(deg, line_cells[:, 0], 1)
            np.add.at(deg, line_cells[:, 1], 1)
        endpoints = np.flatnonzero(deg == 1)

        # Force boundary markers from inlet choice
        bmarker_global = np.zeros((n_points,), dtype=np.int32)
        if endpoints.size > 0:
            bmarker_global[endpoints] = int(outlet_marker)
        bmarker_global[int(inlet_node_vtk_index)] = int(inlet_marker)

        if int(inlet_node_vtk_index) not in set(endpoints.tolist()):
            print(
                f"[warning] Forced inlet node {inlet_node_vtk_index} is NOT an endpoint (degree={deg[inlet_node_vtk_index]}). "
                f"It will still be marked as inlet, but your model may expect inlet to be an endpoint."
            )

        # Radius per cell
        radius_global = None

        # 1) cell_data radius
        if radius_cell_data is not None and radius_cell_data in m.cell_data_dict:
            try:
                r = m.get_cell_data(radius_cell_data, "line")
                radius_global = np.asarray(r, dtype=np.float64)
            except Exception:
                if had_polyline and poly_cells is not None:
                    try:
                        poly_r = m.get_cell_data(radius_cell_data, "polyline")
                        radius_global = _replicate_polyline_cell_data_to_segments(poly_cells, np.asarray(poly_r))
                    except Exception:
                        radius_global = None

        # 2) point_data radius -> average endpoints
        if radius_global is None and radius_point_data and radius_point_data in m.point_data:
            rp = np.asarray(m.point_data[radius_point_data], dtype=np.float64).ravel()
            if rp.size != n_points:
                raise ValueError(f"point_data['{radius_point_data}'] has size {rp.size}, expected {n_points}.")
            if n_cells == 0:
                radius_global = np.zeros((0,), dtype=np.float64)
            else:
                radius_global = 0.5 * (rp[line_cells[:, 0]] + rp[line_cells[:, 1]])

        # 3) fallback constant
        if radius_global is None:
            radius_global = np.full((n_cells,), float(default_radius), dtype=np.float64)

        # For branching 1D meshes, allow larger vertex-to-cell link counts
        max_links = int(max(2, int(deg.max(initial=2))))

        # Write XDMF containing the line mesh; also include arrays (handy for inspection)
        out = meshio.Mesh(
            points=pts,
            cells={"line": line_cells},
            point_data={"boundary_marker_forced": bmarker_global},
            cell_data={radius_cell_data: [radius_global]},
        )
        _write_xdmf_from_meshio(comm, out, xdmf_path)

        data = NetworkVTKData(
            n_points=n_points,
            n_cells=n_cells,
            radius_global=radius_global,
            bmarker_global=bmarker_global,
            max_facet_to_cell_links=max_links,
        )
    else:
        data = None
        _write_xdmf_from_meshio(comm, None, xdmf_path)

    data = comm.bcast(data, root=0)

    lmbda = _read_dolfinx_mesh_from_xdmf(comm, xdmf_path, max_facet_to_cell_links=int(data.max_facet_to_cell_links))
    lmbda.topology.create_connectivity(lmbda.topology.dim - 1, lmbda.topology.dim)
    lmbda.topology.create_connectivity(0, lmbda.topology.dim)

    # Build vertex MeshTags (dim=0) from ORIGINAL input indices
    boundaries = build_boundary_meshtags_from_global_markers(lmbda, data.bmarker_global)

    return lmbda, boundaries, data


def load_tissue_from_vtk(comm: MPI.Comm, tissue_vtk: Path, xdmf_path: Path) -> dmesh.Mesh:
    if comm.rank == 0:
        import meshio

        m = meshio.read(str(tissue_vtk))
        pts = _ensure_3d_points(m.points)
        cells_dict = m.cells_dict

        cell_type = _pick_first_existing_celltype(cells_dict, preferred=("tetra", "hexahedron", "wedge", "pyramid"))
        cells = np.asarray(cells_dict[cell_type], dtype=np.int64)

        out = meshio.Mesh(points=pts, cells={cell_type: cells})
        _write_xdmf_from_meshio(comm, out, xdmf_path)
    else:
        _write_xdmf_from_meshio(comm, None, xdmf_path)

    omega = _read_dolfinx_mesh_from_xdmf(comm, xdmf_path, max_facet_to_cell_links=2)
    omega.topology.create_connectivity(omega.topology.dim - 1, omega.topology.dim)
    return omega


# -----------------------------------------------------------------------------
# Radius helpers (same idea as your original)
# -----------------------------------------------------------------------------

def build_radius_fields_from_global(lmbda: dmesh.Mesh, radius_global: np.ndarray) -> tuple[fem.Function, np.ndarray]:
    DG0 = fem.functionspace(lmbda, ("DG", 0))
    r_cell = fem.Function(DG0)
    r_cell.name = "radius_cell"

    tdim = lmbda.topology.dim
    cell_map = lmbda.topology.index_map(tdim)
    num_cells_local = cell_map.size_local
    num_cells = num_cells_local + cell_map.num_ghosts

    oc = np.asarray(lmbda.topology.original_cell_index, dtype=np.int64)
    if oc.size < num_cells:
        raise RuntimeError("mesh.topology.original_cell_index is smaller than local+ghost cell count.")

    radius_per_cell = np.empty((num_cells,), dtype=np.float64)
    for c in range(num_cells):
        g = int(oc[c])
        if g < 0 or g >= radius_global.size:
            raise IndexError(f"original_cell_index[{c}]={g} out of bounds for radius_global size={radius_global.size}")
        radius_per_cell[c] = float(radius_global[g])

    r_cell.x.array[:num_cells_local] = radius_per_cell[:num_cells_local].astype(r_cell.x.array.dtype, copy=False)
    r_cell.x.scatter_forward()
    return r_cell, radius_per_cell


def make_radius_callable(lmbda: dmesh.Mesh, radius_per_cell: np.ndarray, default_radius: float):
    from dolfinx import geometry

    tdim = lmbda.topology.dim
    tree = geometry.bb_tree(lmbda, tdim)

    def radius_fun(xT: np.ndarray) -> np.ndarray:
        pts = np.asarray(xT.T, dtype=np.float64)
        if pts.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        candidates = geometry.compute_collisions_points(tree, pts)
        colliding = geometry.compute_colliding_cells(lmbda, candidates, pts)

        out = np.full((pts.shape[0],), float(default_radius), dtype=np.float64)
        for i in range(pts.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                out[i] = float(radius_per_cell[int(links[0])])
        return out

    return radius_fun


def build_boundary_meshtags_from_global_markers(lmbda: dmesh.Mesh, bmarker_global: np.ndarray) -> dmesh.MeshTags:
    n_owned = lmbda.geometry.index_map.size_local
    input_ids = np.asarray(lmbda.geometry.input_global_indices[:n_owned], dtype=np.int64)

    vals = bmarker_global[input_ids].astype(np.int32, copy=False)
    idx = np.arange(n_owned, dtype=np.int32)

    mask = vals != 0
    indices = idx[mask]
    values = vals[mask]

    return dmesh.meshtags(lmbda, 0, indices, values)


# -----------------------------------------------------------------------------
# Solve (same formulation)
# -----------------------------------------------------------------------------

def solve_coupled_from_vtk_forced_inlet(
        outdir: Path,
        tissue_vtk: Path,
        network_vtk: Path,
        inlet_node_vtk_index: int,
        params: Params = Params(),
):
    comm = MPI.COMM_WORLD
    outdir.mkdir(parents=True, exist_ok=True)

    tmp = outdir / "_converted_xdmf"
    tmp.mkdir(parents=True, exist_ok=True)

    omega = load_tissue_from_vtk(comm, tissue_vtk, tmp / "tissue.xdmf")
    lmbda, boundaries, netdata = load_network_from_vtk_forced_inlet(
        comm,
        network_vtk,
        tmp / "network.xdmf",
        inlet_node_vtk_index=inlet_node_vtk_index,
        radius_cell_data=NETWORK_RADIUS_CELL_DATA,
        radius_point_data=NETWORK_RADIUS_POINT_DATA,
        inlet_marker=INLET_MARKER,
        outlet_marker=OUTLET_MARKER,
        default_radius=DEFAULT_RADIUS,
    )

    inlet_vertices = boundaries.indices[boundaries.values == int(INLET_MARKER)].astype(np.int32)
    outlet_vertices = boundaries.indices[boundaries.values == int(OUTLET_MARKER)].astype(np.int32)

    if comm.rank == 0:
        print(f"[input] tissue_vtk={tissue_vtk}")
        print(f"[input] network_vtk={network_vtk}")
        print(f"[network] forced inlet VTK node index = {inlet_node_vtk_index}")
        print(f"[network] branching-safe max_facet_to_cell_links = {netdata.max_facet_to_cell_links}")
        print(f"[network] inlet vertices (local on rank0)  = {inlet_vertices.tolist()}")
        print(f"[network] outlet vertices (local on rank0) = {outlet_vertices.tolist()}")

    # Spaces
    V3 = fem.functionspace(omega, ("Lagrange", DEGREE_3D))
    V1 = fem.functionspace(lmbda, ("Lagrange", DEGREE_1D))

    W = ufl.MixedFunctionSpace(V3, V1)
    (p_t, P) = ufl.TrialFunctions(W)
    (v_t, w) = ufl.TestFunctions(W)

    # Radii
    r_cell, radius_per_cell = build_radius_fields_from_global(lmbda, netdata.radius_global)
    max_r = float(np.max(netdata.radius_global)) if netdata.radius_global.size > 0 else float(DEFAULT_RADIUS)

    r_vertex = fem.Function(V1)
    r_vertex.name = "radius_vertex"
    r_vertex.x.array[:] = float(max_r)

    lmbda.topology.create_connectivity(0, 1)
    v2c = lmbda.topology.connectivity(0, 1)
    DG0 = r_cell.function_space

    def radius_at_vertex(vtx: int) -> float:
        cells = v2c.links(vtx)
        if len(cells) == 0:
            return float(max_r)
        c = int(cells[0])
        dof_c = int(DG0.dofmap.cell_dofs(c)[0])
        return float(r_cell.x.array[dof_c])

    for vtx in inlet_vertices.tolist():
        dofs = fem.locate_dofs_topological(V1, 0, np.array([vtx], dtype=np.int32))
        if len(dofs) > 0:
            r_vertex.x.array[dofs[0]] = radius_at_vertex(int(vtx))

    for vtx in outlet_vertices.tolist():
        dofs = fem.locate_dofs_topological(V1, 0, np.array([vtx], dtype=np.int32))
        if len(dofs) > 0:
            r_vertex.x.array[dofs[0]] = radius_at_vertex(int(vtx))

    r_vertex.x.scatter_forward()

    # Circle/Average coupling
    radius_fun = make_radius_callable(lmbda, radius_per_cell, default_radius=max_r)
    circle_trial = Circle(lmbda, radius=radius_fun, degree=CIRCLE_QUAD_DEGREE)
    circle_test = Circle(lmbda, radius=radius_fun, degree=CIRCLE_QUAD_DEGREE)

    Rs = V1
    p_avg = Average(p_t, circle_trial, Rs)
    v_avg = Average(v_t, circle_test, Rs)

    # Measures
    dxOmega = ufl.Measure("dx", domain=omega)
    dxLambda = ufl.Measure("dx", domain=lmbda)

    dsOmega = ufl.Measure("ds", domain=omega)
    dsLambda = ufl.Measure("ds", domain=lmbda, subdomain_data=boundaries)
    dsLambdaOutlet = dsLambda(int(OUTLET_MARKER))

    # Constants
    k_t = fem.Constant(omega, default_scalar_type(params.k_t))
    mu3 = fem.Constant(omega, default_scalar_type(params.mu))
    gamma_R = fem.Constant(omega, default_scalar_type(params.gamma_R))
    P_cvp3 = fem.Constant(omega, default_scalar_type(params.P_cvp))

    mu1 = fem.Constant(lmbda, default_scalar_type(params.mu))
    gamma = fem.Constant(lmbda, default_scalar_type(params.gamma))
    gamma_a = fem.Constant(lmbda, default_scalar_type(params.gamma_a))
    P_cvp1 = fem.Constant(lmbda, default_scalar_type(params.P_cvp))

    # Geometry factors
    D_area_cell = ufl.pi * r_cell ** 2
    D_peri_cell = 2.0 * ufl.pi * r_cell
    k_v_cell = (r_cell ** 2) / 8.0
    D_area_vertex = ufl.pi * r_vertex ** 2

    # Bilinear forms (same structure)
    a00 = (k_t / mu3) * ufl.inner(ufl.grad(p_t), ufl.grad(v_t)) * dxOmega
    a00 += gamma_R * p_t * v_t * dsOmega
    a00 += gamma * p_avg * v_avg * D_peri_cell * dxLambda

    a01 = -gamma * P * v_avg * D_peri_cell * dxLambda
    a01 += -(gamma_a / mu1) * P * v_avg * D_area_vertex * dsLambdaOutlet

    a10 = -gamma * p_avg * w * D_peri_cell * dxLambda

    a11 = (k_v_cell / mu1) * D_area_cell * ufl.inner(ufl.grad(P), ufl.grad(w)) * dxLambda
    a11 += gamma * P * w * D_peri_cell * dxLambda
    a11 += (gamma_a / mu1) * P * w * D_area_vertex * dsLambdaOutlet

    a = a00 + a01 + a10 + a11

    # Linear forms
    L0 = gamma_R * P_cvp3 * v_t * dsOmega
    L0 += (gamma_a / mu1) * P_cvp1 * v_avg * D_area_vertex * dsLambdaOutlet
    L1 = (gamma_a / mu1) * P_cvp1 * w * D_area_vertex * dsLambdaOutlet
    L = L0 + L1

    # Inlet BC
    inlet_dofs = fem.locate_dofs_topological(V1, 0, inlet_vertices)
    P_in_fun = fem.Function(V1)
    P_in_fun.x.array[:] = default_scalar_type(params.P_in)
    bc_inlet = fem.dirichletbc(P_in_fun, inlet_dofs)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    }
    problem = LinearProblem(
        a,
        L,
        bcs=[bc_inlet],
        petsc_options_prefix="la_from_vtk_forced_inlet",
        petsc_options=petsc_options,
    )
    p_sol, P_sol = problem.solve()
    p_sol.name = "p_t"
    P_sol.name = "P"

    # Diagnostics
    exchange_wall_form = gamma * D_peri_cell * (P_sol - Average(p_sol, circle_trial, Rs)) * dxLambda
    exchange_terminal_form = (gamma_a / mu1) * D_area_vertex * (P_sol - P_cvp1) * dsLambdaOutlet

    Q_wall = assemble_scalar(exchange_wall_form, op=MPI.SUM)
    Q_term = assemble_scalar(exchange_terminal_form, op=MPI.SUM)

    if comm.rank == 0:
        print("Solved coupled problem from VTK inputs (forced inlet)!")
        print(f"Total vessel-wall exchange = {Q_wall:.6e}")
        print(f"Total terminal exchange    = {Q_term:.6e}")

    # Output
    fmt = OUTPUT_FORMAT.lower()
    if fmt == "vtx":
        with io.VTXWriter(omega.comm, outdir / "p_t.bp", [p_sol]) as vtx:
            vtx.write(0.0)
        with io.VTXWriter(lmbda.comm, outdir / "P.bp", [P_sol, r_cell, r_vertex]) as vtx:
            vtx.write(0.0)

    elif fmt == "vtk":
        with io.VTKFile(omega.comm, outdir / "p_t.pvd", "w") as vtk:
            vtk.write_mesh(omega, 0.0)
            vtk.write_function(p_sol, 0.0)
        with io.VTKFile(lmbda.comm, outdir / "network.pvd", "w") as vtk:
            vtk.write_mesh(lmbda, 0.0)
            vtk.write_function(P_sol, 0.0)
            vtk.write_function(r_cell, 0.0)
            vtk.write_function(r_vertex, 0.0)

    else:  # xdmf
        with io.XDMFFile(omega.comm, outdir / "p_t.xdmf", "w") as xdmf:
            xdmf.write_mesh(omega)
            xdmf.write_function(p_sol, 0.0)

        with io.XDMFFile(lmbda.comm, outdir / "network.xdmf", "w") as xdmf:
            xdmf.write_mesh(lmbda)
            xdmf.write_function(P_sol, 0.0)
            xdmf.write_function(r_cell, 0.0)
            xdmf.write_function(r_vertex, 0.0)
            try:
                xdmf.write_meshtags(boundaries, lmbda.geometry)
            except Exception as e:
                if comm.rank == 0:
                    print(f"[warning] Could not write boundary meshtags: {type(e).__name__}: {e}")

    if comm.rank == 0:
        print(f"Results written to: {outdir} (format={fmt})")

    return p_sol, P_sol


# -----------------------------------------------------------------------------
# Main (no CLI; hard-coded inputs)
# -----------------------------------------------------------------------------

def main():
    from datetime import datetime

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        if not TISSUE_VTK.exists():
            raise FileNotFoundError(f"Missing tissue VTK: {TISSUE_VTK.resolve()}")
        if not NETWORK_VTK.exists():
            raise FileNotFoundError(f"Missing network VTK: {NETWORK_VTK.resolve()}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outdir = RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    solve_coupled_from_vtk_forced_inlet(
        outdir=outdir,
        tissue_vtk=TISSUE_VTK,
        network_vtk=NETWORK_VTK,
        inlet_node_vtk_index=INLET_NODE_VTK_INDEX,
        params=Params(),
    )


if __name__ == "__main__":
    main()
