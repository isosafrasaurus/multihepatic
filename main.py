from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ufl
from mpi4py import MPI

import dolfinx.fem as fem
import dolfinx.io as io
import dolfinx.mesh as dmesh
from dolfinx import default_scalar_type

from fenicsx_ii import Average, Circle, LinearProblem, assemble_scalar

import networkx as nx
from networks_fenicsx import NetworkMesh

TEST_GRAPH_NODES: Dict[int, List[float]] = {
    0: [0.000, 0.020, 0.015],
    1: [0.010, 0.020, 0.015],
    2: [0.022, 0.013, 0.015],
    3: [0.022, 0.028, 0.015],
    4: [0.015, 0.005, 0.015],
    5: [0.015, 0.035, 0.015],
    6: [0.038, 0.005, 0.015],
    7: [0.038, 0.035, 0.015],
}

TEST_GRAPH_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 0.004),
    (1, 2, 0.003),
    (1, 3, 0.003),
    (2, 4, 0.002),
    (2, 6, 0.003),
    (3, 5, 0.002),
    (3, 7, 0.003),
]


@dataclass(frozen=True)
class Params:
    gamma: float = 3.6145827741262347e-05
    gamma_a: float = 8.225197366649115e-08
    gamma_R: float = 8.620057937882969e-08
    mu: float = 1.0e-3
    k_t: float = 1.0e-10
    P_in: float = 100.0 * 133.322
    P_cvp: float = 1.0 * 133.322


def bbox_from_points(points: np.ndarray, pad: float) -> Tuple[np.ndarray, np.ndarray]:
    mn = points.min(axis=0) - pad
    mx = points.max(axis=0) + pad
    return mn, mx


def create_tissue_box_mesh(
        comm: MPI.Comm,
        mn: np.ndarray,
        mx: np.ndarray,
        h: float,
) -> dmesh.Mesh:
    ext = mx - mn

    n = [max(2, int(np.ceil(ext[i] / h))) for i in range(3)]
    omega = dmesh.create_box(
        comm,
        [mn.tolist(), mx.tolist()],
        n,
        cell_type=dmesh.CellType.tetrahedron,
    )
    omega.topology.create_connectivity(omega.topology.dim - 1, omega.topology.dim)
    return omega


def build_test_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for nid, pos in TEST_GRAPH_NODES.items():
        G.add_node(int(nid), pos=tuple(float(x) for x in pos))
    for u, v, r in TEST_GRAPH_EDGES:
        G.add_edge(int(u), int(v), radius=float(r))
    return G


def infer_sources_sinks(G: nx.DiGraph) -> Tuple[List[int], List[int]]:
    sources = [n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) > 0]
    sinks = [n for n in G.nodes if G.out_degree(n) == 0 and G.in_degree(n) > 0]
    return sources, sinks


def select_inlet_outlet_markers(network: NetworkMesh, sources: List[int], sinks: List[int]) -> Tuple[int, int]:
    bnd = network.boundaries
    in_verts = set(bnd.indices[bnd.values == network.in_marker].tolist())
    out_verts = set(bnd.indices[bnd.values == network.out_marker].tolist())

    src_set = set(sources)
    sink_set = set(sinks)

    if out_verts == src_set and in_verts == sink_set:
        return network.out_marker, network.in_marker

    if in_verts == src_set and out_verts == sink_set:
        return network.in_marker, network.out_marker

    return network.out_marker, network.in_marker


def edge_color_mapping(G: nx.DiGraph) -> Dict[Tuple[int, int], int]:
    return {edge: i for i, edge in enumerate(G.edges)}


def build_radius_fields(
        lmbda: dmesh.Mesh,
        subdomains: dmesh.MeshTags,
        radius_by_color: np.ndarray,
) -> Tuple[fem.Function, np.ndarray]:
    DG0 = fem.functionspace(lmbda, ("DG", 0))
    r_cell = fem.Function(DG0)
    r_cell.name = "radius_cell"

    tdim = lmbda.topology.dim
    cell_map = lmbda.topology.index_map(tdim)
    num_cells_local = cell_map.size_local
    num_cells = num_cells_local + cell_map.num_ghosts

    cell_marker = np.zeros(num_cells, dtype=np.int32)
    cell_marker[subdomains.indices] = subdomains.values

    radius_per_cell = radius_by_color[cell_marker]

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


def solve_coupled_test_graph(
        outdir: Path,
        params: Params = Params(),
        N_per_edge: int = 12,
        tissue_h: float = 0.004,
        degree_3d: int = 1,
        degree_1d: int = 1,
        circle_quad_degree: int = 20,
        output_format: str = "xdmf",
):
    comm = MPI.COMM_WORLD
    outdir.mkdir(parents=True, exist_ok=True)

    G = build_test_graph()
    sources, sinks = infer_sources_sinks(G)

    network = NetworkMesh(G, N=N_per_edge, color_strategy=None, comm=comm, graph_rank=0)
    lmbda = network.mesh

    inlet_marker = network.out_marker
    outlet_marker = network.in_marker

    bnd = network.boundaries
    inlet_vertices = bnd.indices[bnd.values == inlet_marker].astype(np.int32)
    outlet_vertices = bnd.indices[bnd.values == outlet_marker].astype(np.int32)

    if comm.rank == 0:
        print(f"Graph sources: {sources}")
        print(f"Graph sinks: {sinks}")
        print(f"Using inlet_marker={inlet_marker}, outlet_marker={outlet_marker}")
        print(f"Inlet vertices on mesh: {inlet_vertices.tolist()}")
        print(f"Outlet vertices on mesh: {outlet_vertices.tolist()}")

    node_xyz = np.array([G.nodes[n]["pos"] for n in sorted(G.nodes)], dtype=np.float64)
    max_r = max(r for _, _, r in TEST_GRAPH_EDGES)
    pad = 8.0 * max_r + 0.005

    mn, mx = bbox_from_points(node_xyz, pad=pad)
    omega = create_tissue_box_mesh(comm, mn=mn, mx=mx, h=tissue_h)

    V3 = fem.functionspace(omega, ("Lagrange", degree_3d))
    V1 = fem.functionspace(lmbda, ("Lagrange", degree_1d))

    W = ufl.MixedFunctionSpace(V3, V1)
    (p_t, P) = ufl.TrialFunctions(W)
    (v_t, w) = ufl.TestFunctions(W)

    edge_color = edge_color_mapping(G)
    num_colors = len(edge_color)
    radius_by_color = np.zeros((num_colors,), dtype=np.float64)
    for edge, c in edge_color.items():
        radius_by_color[c] = float(G.edges[edge]["radius"])

    r_cell, radius_per_cell = build_radius_fields(lmbda, network.subdomains, radius_by_color)

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

    radius_fun = make_radius_callable(lmbda, radius_per_cell, default_radius=max_r)
    circle_trial = Circle(lmbda, radius=radius_fun, degree=circle_quad_degree)
    circle_test = Circle(lmbda, radius=radius_fun, degree=circle_quad_degree)

    Rs = V1
    p_avg = Average(p_t, circle_trial, Rs)
    v_avg = Average(v_t, circle_test, Rs)

    dxOmega = ufl.Measure("dx", domain=omega)
    dxLambda = ufl.Measure("dx", domain=lmbda)

    dsOmega = ufl.Measure("ds", domain=omega)

    dsLambda = ufl.Measure("ds", domain=lmbda, subdomain_data=network.boundaries)
    dsLambdaOutlet = dsLambda(outlet_marker)

    k_t = fem.Constant(omega, default_scalar_type(params.k_t))
    mu3 = fem.Constant(omega, default_scalar_type(params.mu))
    gamma_R = fem.Constant(omega, default_scalar_type(params.gamma_R))
    P_cvp3 = fem.Constant(omega, default_scalar_type(params.P_cvp))

    mu1 = fem.Constant(lmbda, default_scalar_type(params.mu))
    gamma = fem.Constant(lmbda, default_scalar_type(params.gamma))
    gamma_a = fem.Constant(lmbda, default_scalar_type(params.gamma_a))
    P_cvp1 = fem.Constant(lmbda, default_scalar_type(params.P_cvp))

    D_area_cell = ufl.pi * r_cell ** 2
    D_peri_cell = 2.0 * ufl.pi * r_cell

    k_v_cell = (r_cell ** 2) / 8.0

    D_area_vertex = ufl.pi * r_vertex ** 2

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

    L0 = gamma_R * P_cvp3 * v_t * dsOmega
    L0 += (gamma_a / mu1) * P_cvp1 * v_avg * D_area_vertex * dsLambdaOutlet
    L1 = (gamma_a / mu1) * P_cvp1 * w * D_area_vertex * dsLambdaOutlet
    L = L0 + L1

    inlet_dofs = fem.locate_dofs_topological(V1, 0, inlet_vertices)

    P_in_fun = fem.Function(V1)
    P_in_fun.x.array[:] = default_scalar_type(params.P_in)
    bc_inlet = fem.dirichletbc(P_in_fun, inlet_dofs)
    bcs = [bc_inlet]

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    }
    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="la_test_graph",
        petsc_options=petsc_options,
    )
    p_sol, P_sol = problem.solve()
    p_sol.name = "p_t"
    P_sol.name = "P"

    exchange_wall_form = gamma * D_peri_cell * (P_sol - Average(p_sol, circle_trial, Rs)) * dxLambda
    exchange_terminal_form = (gamma_a / mu1) * D_area_vertex * (P_sol - P_cvp1) * dsLambdaOutlet

    Q_wall = assemble_scalar(exchange_wall_form, op=MPI.SUM)
    Q_term = assemble_scalar(exchange_terminal_form, op=MPI.SUM)

    if comm.rank == 0:
        print("LAM test solved!")
        print(f"Total vessel-wall exchange $int_{{Lambda}} gamma,|partial Theta|,(P-Pi(p)),mathrm{{d}}s$ = {Q_wall:.6e}")
        print(f"Total terminal exchange $sum_{{A}} |Theta| gamma_a/mu,(P-P_mathrm{{cvp}})$ = {Q_term:.6e}")

    fmt = output_format.lower()
    if fmt not in {"xdmf", "vtk", "vtx"}:
        raise ValueError(f"Unknown output_format={output_format!r}. Use 'xdmf', 'vtk', or 'vtx'.")

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
                xdmf.write_meshtags(network.subdomains, lmbda.geometry)
                xdmf.write_meshtags(network.boundaries, lmbda.geometry)
            except Exception as e:
                if comm.rank == 0:
                    print(f"[warning] Could not write meshtags to XDMF: {type(e).__name__}: {e}")

    if comm.rank == 0:
        print(f"Results written to: {outdir} (format={fmt})")
        print("Open in ParaView:")
        if fmt == "xdmf":
            print(f"  - {outdir / 'p_t.xdmf'}")
            print(f"  - {outdir / 'network.xdmf'}")
        elif fmt == "vtk":
            print(f"  - {outdir / 'p_t.pvd'}")
            print(f"  - {outdir / 'network.pvd'}")
        else:
            print(f"  - {outdir / 'p_t.bp'}")
            print(f"  - {outdir / 'P.bp'}")

    return p_sol, P_sol


def main():
    import os

    outdir = Path(__file__).resolve().parent / "results_test_graph"
    outdir.mkdir(parents=True, exist_ok=True)
    fmt = os.environ.get("DOLFINX_OUTPUT_FORMAT", "vtk")
    solve_coupled_test_graph(outdir=outdir, output_format=fmt)


if __name__ == "__main__":
    main()
