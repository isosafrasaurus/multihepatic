from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from mpi4py import MPI
import networkx as nx
import dolfinx.mesh as dmesh

from src import (
    Parameters,
    AssemblyOptions,
    SolverOptions,
    OutputNames,
    OutputOptions,
    write_solution,
)
from src.domain import Domain1D, Domain3D
from src.problem import PressureProblem
from src.system import (
    make_rank_logger,
    print_environment,
    setup_mpi_debug,
    setup_faulthandler,
    barrier as mpi_barrier,
)


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


def solve_coupled_test_graph(
    outdir: Path,
    *,
    params: Parameters = Parameters(),
    N_per_edge: int = 12,
    tissue_h: float = 0.002,
    degree_3d: int = 1,
    degree_1d: int = 1,
    circle_quad_degree: int = 20,
    output_format: str = "xdmf",
):
    comm = MPI.COMM_WORLD
    setup_mpi_debug(comm)

    rprint = make_rank_logger(comm)

    def barrier(tag: str) -> None:
        mpi_barrier(comm, tag, rprint)

    setup_faulthandler(rprint=rprint)
    print_environment(comm, rprint)

    barrier("start")

    try:
        rprint(f"outdir={outdir}")
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
        network = Domain1D.from_network(
            G,
            points_per_edge=N_per_edge,
            comm=comm,
            graph_rank=0,
            color_strategy=None,
        )
        rprint(f"NetworkMesh created in {time.time()-t0:.3f}s")
        barrier("after NetworkMesh")

        lmbda = network.mesh
        rprint(f"Network mesh: tdim={lmbda.topology.dim}, gdim={lmbda.geometry.dim}, comm.size={lmbda.comm.size}")
        for dim in [0, lmbda.topology.dim]:
            im = lmbda.topology.index_map(dim)
            rprint(f"index_map(dim={dim}): size_local={im.size_local}, num_ghosts={im.num_ghosts}")

        bnd = network.boundaries
        rprint(
            f"boundaries: indices.shape={bnd.indices.shape}, values.shape={bnd.values.shape}, "
            f"values_unique={np.unique(bnd.values) if bnd.values.size else 'EMPTY'}"
        )
        if network.subdomains is None:
            rprint("subdomains: None  (!!! this would break radius build)")
        else:
            sd = network.subdomains
            rprint(
                f"subdomains: indices.shape={sd.indices.shape}, values.shape={sd.values.shape}, "
                f"values_unique={np.unique(sd.values) if sd.values.size else 'EMPTY'}"
            )

        # Match the original marker print exactly:
        inlet_marker = network.inlet_marker   # this is NetworkMesh.out_marker
        outlet_marker = network.outlet_marker # this is NetworkMesh.in_marker
        rprint(
            f"Markers: in_marker={outlet_marker}, out_marker={inlet_marker} "
            f"=> inlet_marker={inlet_marker}, outlet_marker={outlet_marker}"
        )
        rprint(f"inlet_vertices(local view)={network.inlet_vertices.tolist()}")
        rprint(f"outlet_vertices(local view)={network.outlet_vertices.tolist()}")
        barrier("after markers/vertices")

        max_r = max(r for _, _, r in TEST_GRAPH_EDGES)

        mn = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        mx = np.array([0.040, 0.040, 0.030], dtype=np.float64)

        rprint(f"Building tissue mesh: mn={mn.tolist()} mx={mx.tolist()} h={tissue_h}")
        t0 = time.time()
        tissue = Domain3D.from_box(
            comm,
            mn,
            mx,
            target_h=tissue_h,
            cell_type=dmesh.CellType.tetrahedron,
        )
        omega = tissue.mesh
        rprint(f"Tissue mesh created in {time.time()-t0:.3f}s (tdim={omega.topology.dim}, gdim={omega.geometry.dim})")
        im3 = omega.topology.index_map(omega.topology.dim)
        rprint(f"omega cells: local={im3.size_local}, ghosts={im3.num_ghosts}")
        barrier("after tissue mesh")

        rprint("Building edge color mapping + radius_by_color...")
        edge_color = edge_color_mapping(G)
        num_colors = len(edge_color)
        radius_by_color = np.zeros((num_colors,), dtype=np.float64)
        for edge, c in edge_color.items():
            radius_by_color[c] = float(G.edges[edge]["radius"])
        rprint(f"radius_by_color (len={len(radius_by_color)}): {radius_by_color.tolist()}")

        sd = network.subdomains
        if sd is None:
            raise RuntimeError("network.subdomains is None; cannot build DG0 cell radii")

        if sd.values.size:
            max_tag_local = int(np.max(sd.values))
        else:
            max_tag_local = -1
        max_tag_global = comm.allreduce(max_tag_local, op=MPI.MAX)
        rprint(
            f"subdomain max_tag_local={max_tag_local}, max_tag_global={max_tag_global}, "
            f"radius_by_color_max_index={len(radius_by_color)-1}"
        )
        if max_tag_global >= len(radius_by_color):
            rprint("!!! ERROR: subdomain tag exceeds radius_by_color size; would crash indexing.")
        barrier("before solve")

        assembly = AssemblyOptions(
            degree_3d=degree_3d,
            degree_1d=degree_1d,
            circle_quadrature_degree=circle_quad_degree,
        )

        solver = SolverOptions(
            petsc_options_prefix="la_test_graph",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "ksp_error_if_not_converged": True,
            },
        )

        with PressureProblem(
            tissue,
            network,
            params=params,
            assembly=assembly,
            solver=solver,
            radius_by_tag=radius_by_color,
            default_radius=float(max_r),
            log=rprint,        # ✅ keeps all the internal prints
            barrier=barrier,   # ✅ keeps all the barrier tags
        ) as prob:
            sol = prob.solve()

        # Match original field names for output
        sol.tissue_pressure.name = "p_t"
        sol.network_pressure.name = "P"

        if comm.rank == 0:
            print("LAM test solved!", flush=True)

        if os.environ.get("SKIP_IO", "0") == "1":
            rprint("SKIP_IO=1 -> skipping all output writing.")
            return sol

        fmt = output_format.lower()
        rprint(f"Writing output, format={fmt}")

        names = OutputNames(
            tissue_pressure="p_t",
            network="network",
            network_vtx="P",
            tissue_velocity="v_tissue",
        )
        write_solution(
            outdir,
            tissue,
            network,
            sol,
            options=OutputOptions(format=fmt, time=0.0, write_meshtags=True, names=names),
        )

        if fmt == "vtx":
            rprint("Wrote VTX omega p_t.bp")
            rprint("Wrote VTX lmbda P.bp")
        elif fmt == "vtk":
            rprint("Wrote VTK omega p_t.pvd")
            rprint("Wrote VTK lmbda network.pvd")
        else:
            rprint("Wrote XDMF omega p_t.xdmf")
            rprint("Wrote XDMF lmbda network.xdmf")

        barrier("after IO")

        if comm.rank == 0:
            print(f"Results written to: {outdir} (format={fmt})", flush=True)

        return sol

    except Exception as e:
        # Mirror the original “abort to avoid deadlock” behavior
        from src.system import abort_on_exception
        abort_on_exception(comm, rprint, e)
        raise  # unreachable after Abort, but keeps linters happy


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
