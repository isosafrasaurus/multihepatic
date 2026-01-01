from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from mpi4py import MPI

from src.core import Parameters, SolverOptions
from src.domains.domains import Domain1D, Domain3D
from src.io import OutputOptions, write_solution
from src.problem import AssemblyOptions, PressureVelocityProblem

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

TISSUE_BOX_MAX = np.array([0.040, 0.040, 0.030], dtype=np.float64)
TISSUE_BOX_MIN = np.array([0.000, 0.000, 0.000], dtype=np.float64)


def build_test_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id, pos in TEST_GRAPH_NODES.items():
        graph.add_node(int(node_id), pos=tuple(float(x) for x in pos))
    for u, v, radius in TEST_GRAPH_EDGES:
        graph.add_edge(int(u), int(v), radius=float(radius))
    return graph


def edge_color_mapping(graph: nx.DiGraph) -> dict[tuple[int, int], int]:
    return {edge: i for i, edge in enumerate(graph.edges)}


def solve_test_graph(
        outdir: Path,
        *,
        params: Parameters = Parameters(),
        points_per_edge: int = 12,
        tissue_h: float = 0.004,
        degree_3d: int = 1,
        degree_1d: int = 1,
        circle_quadrature_degree: int = 20,
        output_format: str = "vtk",
) -> None:
    comm = MPI.COMM_WORLD
    outdir.mkdir(parents=True, exist_ok=True)

    graph = build_test_graph()

    # Build Domain1D via the factory
    network_domain = Domain1D.from_networkx_graph(
        graph,
        points_per_edge=points_per_edge,
        comm=comm,
        graph_rank=0,
    )

    # Build tissue box with fixed dimensions [0,0,0] -> [0.040,0.040,0.030]
    tissue_domain = Domain3D.from_box(
        comm,
        min_corner=TISSUE_BOX_MIN,
        max_corner=TISSUE_BOX_MAX,
        target_h=tissue_h,
    )

    # radius_by_tag: tag index -> radius
    max_radius = max(r for _, _, r in TEST_GRAPH_EDGES)
    color_map = edge_color_mapping(graph)
    radius_by_tag: dict[int, float] = {}
    for edge, color in color_map.items():
        radius_by_tag[int(color)] = float(graph.edges[edge]["radius"])

    assembly = AssemblyOptions(
        degree_3d=degree_3d,
        degree_1d=degree_1d,
        circle_quadrature_degree=circle_quadrature_degree,
    )
    solver = SolverOptions(petsc_options_prefix="la_test_graph")

    # Use PressureVelocityProblem so we also compute v = -(k_t/mu) * grad(p_tissue)
    with PressureVelocityProblem(
            tissue_domain,
            network_domain,
            params=params,
            assembly=assembly,
            solver=solver,
            radius_by_tag=radius_by_tag,
            default_radius=max_radius,
    ) as problem:
        solution = problem.solve()

    if comm.rank == 0:
        print("Toy problem solved!")
        print(f"Total vessel-wall exchange: {solution.total_wall_exchange:.6e}")
        print(f"Total terminal exchange: {solution.total_terminal_exchange:.6e}")

    write_solution(
        outdir,
        tissue_domain,
        network_domain,
        solution,
        options=OutputOptions(format=output_format, time=0.0, write_meshtags=True, write_network_tube=True),
    )

    if comm.rank == 0:
        print(f"Results written to: {outdir} (format={output_format})")


def main() -> None:
    results_root = Path(__file__).resolve().parent.parent / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = results_root / timestamp
    fmt = os.environ.get("DOLFINX_OUTPUT_FORMAT", "vtk")
    solve_test_graph(outdir, output_format=fmt)


if __name__ == "__main__":
    main()
