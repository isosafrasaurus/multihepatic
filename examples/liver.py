from __future__ import annotations

import os
import sys
import time
import socket
import traceback
import faulthandler
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ufl
from mpi4py import MPI

import dolfinx
import dolfinx.fem as fem
import dolfinx.io as io
import dolfinx.mesh as dmesh
from dolfinx import default_scalar_type

from fenicsx_ii import Average, Circle, LinearProblem, assemble_scalar

import networkx as nx
from networks_fenicsx import NetworkMesh

def _now() -> str:
    return time.strftime("%H:%M:%S")

def _envpick(keys: List[str]) -> Dict[str, str]:
    out = {}
    for k in keys:
        if k in os.environ:
            out[k] = os.environ[k]
    return out


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

    # Debug sanity (no prints here; caller prints)
    radius_per_cell = radius_by_color[cell_marker]

    r_cell.x.array[:num_cells_local] = radius_per_cell[:num_cells_local].astype(
        r_cell.x.array.dtype, copy=False
    )
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
        tissue_h: float = 0.002,
        degree_3d: int = 1,
        degree_1d: int = 1,
        circle_quad_degree: int = 20,
        output_format: str = "xdmf",
):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Make stdout/stderr as immediate as possible
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    host = socket.gethostname()
    pid = os.getpid()

    def rprint(msg: str) -> None:
        print(f"[{_now()}] [rank {rank}/{size}] [pid {pid}] {msg}", flush=True)

    def barrier(tag: str) -> None:
        if os.environ.get("DEBUG_BARRIERS", "1") == "1":
            rprint(f"ENTER BARRIER: {tag}")
            comm.Barrier()
            rprint(f"EXIT  BARRIER: {tag}")

    # Periodic stack dumps if we hang anywhere
    if os.environ.get("DEBUG_FAULTHANDLER", "1") == "1":
        try:
            faulthandler.enable()
            # Dump every 30s if stuck (tune with env var)
            sec = float(os.environ.get("DEBUG_DUMP_EVERY", "30"))
            faulthandler.dump_traceback_later(sec, repeat=True, file=sys.stderr)
            rprint(f"faulthandler enabled; will dump tracebacks every {sec}s if hung.")
        except Exception as e:
            rprint(f"[warning] faulthandler setup failed: {type(e).__name__}: {e}")

    # Make MPI errors return (best effort)
    try:
        comm.Set_errhandler(MPI.ERRORS_RETURN)
    except Exception:
        pass

    # Print basic environment once per rank
    rprint(f"Host={host}, dolfinx={getattr(dolfinx, '__version__', 'unknown')}")
    rprint(f"MPI library: {MPI.Get_library_version().strip()}")
    rprint(f"Env threads: {_envpick(['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])}")
    rprint(f"Env JIT-ish: {_envpick([k for k in os.environ.keys() if 'JIT' in k or 'FFCX' in k or 'UFL' in k or 'DOLFINX' in k])}")

    barrier("start")

    try:
        rprint(f"outdir={outdir}")
        # NOTE: mkdir on all ranks can be OK, but log it
        t0 = time.time()
        outdir.mkdir(parents=True, exist_ok=True)
        rprint(f"outdir.mkdir done in {time.time()-t0:.3f}s")
        barrier("after mkdir")

        rprint("Building test graph...")
        G = build_test_graph()
        sources, sinks = infer_sources_sinks(G)
        rprint(f"Graph built. nodes={G.number_of_nodes()} edges={G.number_of_edges()} sources={sources} sinks={sinks}")
        barrier("after graph")

        rprint("ABOUT TO CREATE NetworkMesh(...)")
        t0 = time.time()
        network = NetworkMesh(G, N=N_per_edge, color_strategy=None, comm=comm, graph_rank=0)
        rprint(f"NetworkMesh created in {time.time()-t0:.3f}s")
        barrier("after NetworkMesh")

        lmbda = network.mesh
        rprint(f"Network mesh: tdim={lmbda.topology.dim}, gdim={lmbda.geometry.dim}, comm.size={lmbda.comm.size}")

        # Mesh entity counts
        for dim in [0, lmbda.topology.dim]:
            im = lmbda.topology.index_map(dim)
            rprint(f"index_map(dim={dim}): size_local={im.size_local}, num_ghosts={im.num_ghosts}")

        # Tags sanity
        bnd = network.boundaries
        rprint(f"boundaries: indices.shape={bnd.indices.shape}, values.shape={bnd.values.shape}, "
               f"values_unique={np.unique(bnd.values) if bnd.values.size else 'EMPTY'}")
        if getattr(network, "subdomains", None) is None:
            rprint("subdomains: None  (!!! this would break radius build)")
        else:
            sd = network.subdomains
            rprint(f"subdomains: indices.shape={sd.indices.shape}, values.shape={sd.values.shape}, "
                   f"values_unique={np.unique(sd.values) if sd.values.size else 'EMPTY'}")

        inlet_marker = network.out_marker
        outlet_marker = network.in_marker
        inlet_vertices = bnd.indices[bnd.values == inlet_marker].astype(np.int32)
        outlet_vertices = bnd.indices[bnd.values == outlet_marker].astype(np.int32)

        rprint(f"Markers: in_marker={network.in_marker}, out_marker={network.out_marker} "
               f"=> inlet_marker={inlet_marker}, outlet_marker={outlet_marker}")
        rprint(f"inlet_vertices(local view)={inlet_vertices.tolist()}")
        rprint(f"outlet_vertices(local view)={outlet_vertices.tolist()}")

        barrier("after markers/vertices")

        node_xyz = np.array([G.nodes[n]["pos"] for n in sorted(G.nodes)], dtype=np.float64)
        max_r = max(r for _, _, r in TEST_GRAPH_EDGES)

        # Fixed tissue bounding box: [0,0,0] to [0.040, 0.040, 0.030]
        mn = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        mx = np.array([0.040, 0.040, 0.030], dtype=np.float64)

        rprint(f"Building tissue mesh: mn={mn.tolist()} mx={mx.tolist()} h={tissue_h}")
        t0 = time.time()
        omega = create_tissue_box_mesh(comm, mn=mn, mx=mx, h=tissue_h)
        rprint(f"Tissue mesh created in {time.time()-t0:.3f}s (tdim={omega.topology.dim}, gdim={omega.geometry.dim})")
        im3 = omega.topology.index_map(omega.topology.dim)
        rprint(f"omega cells: local={im3.size_local}, ghosts={im3.num_ghosts}")
        barrier("after tissue mesh")

        rprint("Creating function spaces V3/V1 and mixed space W...")
        t0 = time.time()
        V3 = fem.functionspace(omega, ("Lagrange", degree_3d))
        V1 = fem.functionspace(lmbda, ("Lagrange", degree_1d))
        W = ufl.MixedFunctionSpace(V3, V1)
        rprint(f"Function spaces created in {time.time()-t0:.3f}s")

        # DOF counts
        try:
            imV3 = V3.dofmap.index_map
            imV1 = V1.dofmap.index_map
            rprint(f"V3 dofs: local={imV3.size_local}, ghosts={imV3.num_ghosts}")
            rprint(f"V1 dofs: local={imV1.size_local}, ghosts={imV1.num_ghosts}")
        except Exception as e:
            rprint(f"[warning] Could not print dofmap index_map info: {type(e).__name__}: {e}")

        barrier("after spaces")

        rprint("Building edge color mapping + radius_by_color...")
        edge_color = edge_color_mapping(G)
        num_colors = len(edge_color)
        radius_by_color = np.zeros((num_colors,), dtype=np.float64)
        for edge, c in edge_color.items():
            radius_by_color[c] = float(G.edges[edge]["radius"])
        rprint(f"radius_by_color (len={len(radius_by_color)}): {radius_by_color.tolist()}")

        # Check subdomains compatibility
        sd = network.subdomains
        if sd is None:
            raise RuntimeError("network.subdomains is None; cannot build DG0 cell radii")

        # Before building, check if any tag exceeds array bounds (per rank)
        if sd.values.size:
            max_tag_local = int(np.max(sd.values))
        else:
            max_tag_local = -1
        max_tag_global = comm.allreduce(max_tag_local, op=MPI.MAX)
        rprint(f"subdomain max_tag_local={max_tag_local}, max_tag_global={max_tag_global}, radius_by_color_max_index={len(radius_by_color)-1}")
        if max_tag_global >= len(radius_by_color):
            rprint("!!! ERROR: subdomain tag exceeds radius_by_color size; would crash indexing.")
        barrier("before build_radius_fields")

        rprint("Calling build_radius_fields(...)")
        t0 = time.time()
        r_cell, radius_per_cell = build_radius_fields(lmbda, sd, radius_by_color)
        rprint(f"build_radius_fields done in {time.time()-t0:.3f}s; radius_per_cell.shape={radius_per_cell.shape}")
        barrier("after build_radius_fields")

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

        # IMPORTANT: call locate_dofs_topological exactly once per marker on every rank
        inlet_dofs = fem.locate_dofs_topological(V1, 0, inlet_vertices)
        outlet_dofs = fem.locate_dofs_topological(V1, 0, outlet_vertices)

        # For scalar CG1 on a 1D mesh, this is typically 1 dof per vertex.
        if inlet_vertices.size:
            assert len(inlet_dofs) == len(inlet_vertices)
            r_vertex.x.array[inlet_dofs] = np.array(
                [radius_at_vertex(int(v)) for v in inlet_vertices], dtype=r_vertex.x.array.dtype
            )

        if outlet_vertices.size:
            assert len(outlet_dofs) == len(outlet_vertices)
            r_vertex.x.array[outlet_dofs] = np.array(
                [radius_at_vertex(int(v)) for v in outlet_vertices], dtype=r_vertex.x.array.dtype
            )

        comm.Barrier()

        r_vertex.x.scatter_forward()
        rprint("r_vertex scatter_forward done.")
        barrier("after r_vertex")

        rprint("Building radius callable (bb_tree) ...")
        t0 = time.time()
        radius_fun = make_radius_callable(lmbda, radius_per_cell, default_radius=max_r)
        rprint(f"radius callable created in {time.time()-t0:.3f}s")

        # Smoke test radius_fun on rank-local single point
        test_pt = np.array([[float(TEST_GRAPH_NODES[0][0])],
                            [float(TEST_GRAPH_NODES[0][1])],
                            [float(TEST_GRAPH_NODES[0][2])]], dtype=np.float64)  # shape (3,1)
        t0 = time.time()
        try:
            rv = radius_fun(test_pt)
            rprint(f"radius_fun(smoke test) -> {rv} (in {time.time()-t0:.3f}s)")
        except Exception as e:
            rprint(f"[warning] radius_fun smoke test failed: {type(e).__name__}: {e}")

        barrier("before Circle")

        rprint("Creating Circle objects...")
        t0 = time.time()
        circle_trial = Circle(lmbda, radius=radius_fun, degree=circle_quad_degree)
        circle_test = Circle(lmbda, radius=radius_fun, degree=circle_quad_degree)
        rprint(f"Circle objects created in {time.time()-t0:.3f}s")
        barrier("after Circle")

        rprint("Building UFL trial/test functions, Average(...) ...")
        (p_t, P) = ufl.TrialFunctions(W)
        (v_t, w) = ufl.TestFunctions(W)

        Rs = V1
        t0 = time.time()
        p_avg = Average(p_t, circle_trial, Rs)
        v_avg = Average(v_t, circle_test, Rs)
        rprint(f"Average(...) created in {time.time()-t0:.3f}s")
        barrier("after Average")

        dxOmega = ufl.Measure("dx", domain=omega)
        dxLambda = ufl.Measure("dx", domain=lmbda)
        dsOmega = ufl.Measure("ds", domain=omega)
        dsLambda = ufl.Measure("ds", domain=lmbda, subdomain_data=network.boundaries)
        dsLambdaOutlet = dsLambda(outlet_marker)

        # Constants
        rprint("Creating Constants...")
        k_t = fem.Constant(omega, default_scalar_type(params.k_t))
        mu3 = fem.Constant(omega, default_scalar_type(params.mu))
        gamma_R = fem.Constant(omega, default_scalar_type(params.gamma_R))
        P_cvp3 = fem.Constant(omega, default_scalar_type(params.P_cvp))

        mu1 = fem.Constant(lmbda, default_scalar_type(params.mu))
        gamma = fem.Constant(lmbda, default_scalar_type(params.gamma))
        gamma_a = fem.Constant(lmbda, default_scalar_type(params.gamma_a))
        P_cvp1 = fem.Constant(lmbda, default_scalar_type(params.P_cvp))
        rprint("Constants created.")
        barrier("after Constants")

        D_area_cell = ufl.pi * r_cell ** 2
        D_peri_cell = 2.0 * ufl.pi * r_cell
        k_v_cell = (r_cell ** 2) / 8.0
        D_area_vertex = ufl.pi * r_vertex ** 2

        rprint("Building bilinear/linear forms a, L ...")
        t0 = time.time()

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

        rprint(f"Forms built in {time.time()-t0:.3f}s")
        barrier("after forms")

        # BCs
        rprint("Locating inlet dofs...")
        inlet_dofs = fem.locate_dofs_topological(V1, 0, inlet_vertices)
        rprint(f"inlet_dofs={inlet_dofs.tolist()} (len={len(inlet_dofs)})")

        P_in_fun = fem.Function(V1)
        P_in_fun.x.array[:] = default_scalar_type(params.P_in)
        bc_inlet = fem.dirichletbc(P_in_fun, inlet_dofs)
        bcs = [bc_inlet]
        barrier("after BCs")

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        }

        rprint("ABOUT TO CONSTRUCT LinearProblem(...) (this may trigger JIT/assembly)")
        t0 = time.time()
        problem = LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="la_test_graph",
            petsc_options=petsc_options,
        )
        rprint(f"LinearProblem constructed in {time.time()-t0:.3f}s")
        barrier("after LinearProblem ctor")

        rprint("ABOUT TO CALL problem.solve()")
        t0 = time.time()
        p_sol, P_sol = problem.solve()
        rprint(f"problem.solve() returned in {time.time()-t0:.3f}s")
        barrier("after problem.solve")

        p_sol.name = "p_t"
        P_sol.name = "P"

        rprint("Assembling exchange integrals...")
        t0 = time.time()
        exchange_wall_form = gamma * D_peri_cell * (P_sol - Average(p_sol, circle_trial, Rs)) * dxLambda
        exchange_terminal_form = (gamma_a / mu1) * D_area_vertex * (P_sol - P_cvp1) * dsLambdaOutlet

        Q_wall = assemble_scalar(exchange_wall_form, op=MPI.SUM)
        Q_term = assemble_scalar(exchange_terminal_form, op=MPI.SUM)
        rprint(f"exchange assembled in {time.time()-t0:.3f}s; Q_wall={Q_wall} Q_term={Q_term}")
        barrier("after assemble_scalar")

        if rank == 0:
            print("LAM test solved!", flush=True)
            print(f"Total vessel-wall exchange = {Q_wall:.6e}", flush=True)
            print(f"Total terminal exchange    = {Q_term:.6e}", flush=True)

        if os.environ.get("SKIP_IO", "0") == "1":
            rprint("SKIP_IO=1 -> skipping all output writing.")
            return p_sol, P_sol

        fmt = output_format.lower()
        rprint(f"Writing output, format={fmt}")

        if fmt == "vtx":
            rprint("Writing VTX omega p_t.bp ...")
            with io.VTXWriter(omega.comm, outdir / "p_t.bp", [p_sol]) as vtx:
                vtx.write(0.0)
            rprint("Wrote VTX omega p_t.bp")

            rprint("Writing VTX lmbda P.bp ...")
            with io.VTXWriter(lmbda.comm, outdir / "P.bp", [P_sol, r_cell, r_vertex]) as vtx:
                vtx.write(0.0)
            rprint("Wrote VTX lmbda P.bp")

        elif fmt == "vtk":
            rprint("Writing VTK omega p_t.pvd ...")
            with io.VTKFile(omega.comm, outdir / "p_t.pvd", "w") as vtk:
                vtk.write_mesh(omega, 0.0)
                vtk.write_function(p_sol, 0.0)
            rprint("Wrote VTK omega p_t.pvd")

            rprint("Writing VTK lmbda network.pvd ...")
            with io.VTKFile(lmbda.comm, outdir / "network.pvd", "w") as vtk:
                vtk.write_mesh(lmbda, 0.0)
                vtk.write_function(P_sol, 0.0)
                vtk.write_function(r_cell, 0.0)
                vtk.write_function(r_vertex, 0.0)
            rprint("Wrote VTK lmbda network.pvd")

        else:  # xdmf
            rprint("Writing XDMF omega p_t.xdmf ...")
            with io.XDMFFile(omega.comm, outdir / "p_t.xdmf", "w") as xdmf:
                xdmf.write_mesh(omega)
                xdmf.write_function(p_sol, 0.0)
            rprint("Wrote XDMF omega p_t.xdmf")

            rprint("Writing XDMF lmbda network.xdmf ...")
            with io.XDMFFile(lmbda.comm, outdir / "network.xdmf", "w") as xdmf:
                xdmf.write_mesh(lmbda)
                xdmf.write_function(P_sol, 0.0)
                xdmf.write_function(r_cell, 0.0)
                xdmf.write_function(r_vertex, 0.0)

                try:
                    xdmf.write_meshtags(network.subdomains, lmbda.geometry)
                    xdmf.write_meshtags(network.boundaries, lmbda.geometry)
                except Exception as e:
                    if rank == 0:
                        print(f"[warning] Could not write meshtags to XDMF: {type(e).__name__}: {e}", flush=True)
            rprint("Wrote XDMF lmbda network.xdmf")

        barrier("after IO")

        if rank == 0:
            print(f"Results written to: {outdir} (format={fmt})", flush=True)

        return p_sol, P_sol

    except Exception as e:
        # Print full traceback on each rank that hits it, then abort to avoid silent deadlock
        rprint(f"!!! EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            comm.Abort(1)
        except Exception:
            raise


def main():
    results_root = Path(__file__).resolve().parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = results_root / timestamp

    fmt = os.environ.get("DOLFINX_OUTPUT_FORMAT", "vtk")
    solve_coupled_test_graph(outdir=outdir, output_format=fmt)


if __name__ == "__main__":
    main()
