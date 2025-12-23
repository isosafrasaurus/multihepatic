from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from dolfin import (
    FacetNormal,
    File,
    LogLevel,
    Measure,
    assemble,
    dot,
    set_log_level,
)
from graphnics import FenicsGraph, TubeFile

_THIS_FILE = Path(__file__).resolve()
for parent in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (parent / "src" / "__init__.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from src import (  
    AxisPlane,
    Domain1D,
    Domain3D,
    Parameters,
    PressureVelocityProblem,
    Simulation,
    release_solution,
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

TEST_NUM_NODES_EXP = 5


@dataclass(frozen=True)
class RunConfig:
    num_nodes_exp: int = TEST_NUM_NODES_EXP
    inlet_nodes: Tuple[int, ...] = (0,)
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.05, 0.04, 0.03),
    )
    sink_plane_axis: int = 0
    sink_plane_coordinate: float = 0.0


def build_test_graph() -> FenicsGraph:
    G = FenicsGraph()

    for nid, pos in TEST_GRAPH_NODES.items():
        G.add_node(nid, pos=[float(pos[0]), float(pos[1]), float(pos[2])])

    for u, v, r in TEST_GRAPH_EDGES:
        G.add_edge(u, v, radius=float(r))

    for (u, v) in G.edges():
        duv = np.asarray(G.nodes[v]["pos"], dtype=float) - np.asarray(G.nodes[u]["pos"], dtype=float)
        nrm = float(np.linalg.norm(duv))
        if nrm > 0.0:
            duv /= nrm
        G.edges[u, v]["tangent"] = duv

    return G


def run_case(*, cfg: RunConfig, outdir: Path) -> float:
    outdir.mkdir(parents=True, exist_ok=True)

    G = build_test_graph()

    
    sink = AxisPlane(axis=cfg.sink_plane_axis, coordinate=cfg.sink_plane_coordinate)

    lower, upper = cfg.bounds
    bounds = [list(lower), list(upper)]

    with Domain1D(
            G,
            edge_resolution_exp=cfg.num_nodes_exp,
            inlet_node_idxs=list(cfg.inlet_nodes),
    ) as Lambda, Domain3D.from_graph(G, bounds=bounds) as Omega, Simulation(
        Lambda,
        Omega,
        problem_cls=PressureVelocityProblem,
        Omega_sink_subdomain=sink,
    ) as sim:
        params = Parameters(
            gamma=3.6145827741262347e-05,
            gamma_a=8.225197366649115e-08,
            gamma_R=8.620057937882969e-08,
            mu=1.0e-3,
            k_t=1.0e-10,
            P_in=100.0 * 133.322,
            P_cvp=1.0 * 133.322,
        )

        sol = sim.solve(params)

        (File(str(outdir / "pressure_3d.pvd")) << sol.p3d)
        (TubeFile(G, str(outdir / "pressure_1d.pvd")) << sol.p1d)

        if getattr(sol, "v3d", None) is not None:
            (File(str(outdir / "velocity_3d.pvd")) << sol.v3d)

        n = FacetNormal(Omega.mesh)
        ds = Measure("ds", domain=Omega.mesh)
        net_flow_all = float(assemble(dot(sol.v3d, n) * ds))

        release_solution(sol)

    return net_flow_all


def main() -> float:
    set_log_level(LogLevel.ERROR)

    cfg = RunConfig()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("_results") / timestamp

    net_flow_all = run_case(cfg=cfg, outdir=outdir)
    print("net_flow_all =", net_flow_all)
    return net_flow_all


if __name__ == "__main__":
    raise SystemExit(0 if np.isfinite(main()) else 1)
