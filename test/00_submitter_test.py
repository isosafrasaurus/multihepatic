#!/usr/bin/env python3
import os
import sys

project_root = os.environ.get("PROJECT_ROOT")
if not project_root:
    raise EnvironmentError("Environment variable PROJECT_ROOT is not set.")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graphnics import FenicsGraph
from dolfin import FacetNormal, Measure, dot, assemble

from tissue import AxisPlane
from src import Domain1D, Domain3D, Simulation, release_solution
from src.problem import PressureVelocityProblem
from src.composition import Parameters

TEST_NUM_NODES_EXP = 5

TEST_GRAPH_NODES = {
    0: [0.000, 0.020, 0.015],
    1: [0.010, 0.020, 0.015],
    2: [0.022, 0.013, 0.015],
    3: [0.022, 0.028, 0.015],
    4: [0.015, 0.005, 0.015],
    5: [0.015, 0.035, 0.015],
    6: [0.038, 0.005, 0.015],
    7: [0.038, 0.035, 0.015],
}
TEST_GRAPH_EDGES = [
    (0, 1, 0.004),
    (1, 2, 0.003),
    (1, 3, 0.003),
    (2, 4, 0.002),
    (2, 6, 0.003),
    (3, 5, 0.002),
    (3, 7, 0.003),
]


def build_test_graph() -> FenicsGraph:
    G = FenicsGraph()
    for node_id, pos in TEST_GRAPH_NODES.items():
        G.add_node(node_id, pos=pos)
    for (u, v, radius) in TEST_GRAPH_EDGES:
        G.add_edge(u, v, radius=radius)

    try:
        G.make_mesh(num_nodes_exp=TEST_NUM_NODES_EXP)
    except TypeError:
        G.make_mesh(n=TEST_NUM_NODES_EXP)
    G.make_submeshes()
    if hasattr(G, "compute_tangents"):
        G.compute_tangents()
    return G


def main() -> float:
    G = build_test_graph()

    X_ZERO_PLANE = AxisPlane(axis=0, coordinate=0.0)
    bounds = [[0.0, 0.0, 0.0], [0.05, 0.04, 0.03]]

    with Domain1D(G, Lambda_num_nodes_exp=TEST_NUM_NODES_EXP, inlet_nodes=[0]) as Lambda, \
            Domain3D.from_graph(G, bounds=bounds) as Omega, \
            Simulation(
                Lambda,
                Omega,
                problem_cls=PressureVelocityProblem,
                Omega_sink_subdomain=X_ZERO_PLANE,
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

        n = FacetNormal(Omega.Omega)
        ds = Measure("ds", domain=Omega.Omega)
        net_flow_all = assemble(dot(sol.v3d, n) * ds)

        release_solution(sol)

    print("net_flow_all =", float(net_flow_all))
    return float(net_flow_all)


if __name__ == "__main__":
    main()
