import sys
import os
import pytz
import datetime
import numpy as np
from scipy.optimize import minimize
from graphnics import FenicsGraph

WORK_PATH   = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
sys.path.append(SOURCE_PATH)

import fem
import tissue

TEST_NUM_NODES_EXP = 5

TEST_GRAPH = FenicsGraph()

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

for node_id, pos in TEST_GRAPH_NODES.items():
    TEST_GRAPH.add_node(node_id, pos=pos)
for u, v, radius in TEST_GRAPH_EDGES:
    TEST_GRAPH.add_edge(u, v, radius=radius)

TEST_GRAPH.make_mesh(n=TEST_NUM_NODES_EXP)
TEST_GRAPH.make_submeshes()

OMEGA_BOUNDS = [[0.0, 0.0, 0.0], [0.05, 0.04, 0.03]]
TEST_OMEGA_BUILD = tissue.OmegaBuild(TEST_GRAPH, bounds=OMEGA_BOUNDS)
X_ZERO_PLANE = tissue.AxisPlane(0, 0.0)
TEST_DOMAIN = tissue.DomainBuild(
    TEST_GRAPH,
    TEST_OMEGA_BUILD,
    Lambda_inlet_nodes=[0],
    Omega_sink_subdomain=X_ZERO_PLANE,
)

X_DEFAULT   = [4.855e-05, 3.568e-08, 1.952e-07]  
TARGET_FLOW = 5.0e-6

SOLVER = fem.SubCubes(
    domain=TEST_DOMAIN,
    lower_cube_bounds=[[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
    upper_cube_bounds=[[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]]
)

SOLVER.solve(
    gamma=X_DEFAULT[0],
    gamma_a=X_DEFAULT[1],
    gamma_R=X_DEFAULT[2],
    mu=1.0e-3,
    k_t=1.0e-10,
    P_in=100.0 * 133.322,
    P_cvp=1.0 * 133.322,
)

def compute_flow(params: np.ndarray, solver: fem.SubCubes):
    
    solver.solve(
        gamma=params[0],
        gamma_a=params[1],
        gamma_R=params[2],
        mu=1.0e-3,
        k_t=1.0e-10,
        P_in=100.0 * 133.322,
        P_cvp=1.0 * 133.322,
    )
    return solver.compute_net_flow_all_dolfin()

def objective_log(y: np.ndarray, solver: fem.SubCubes, target: float) -> float:
    x = np.exp(y)
    net_flow = compute_flow(x, solver)
    return (net_flow - target) ** 2

def optimization_callback(yk: np.ndarray):
    params = np.exp(yk)
    flow = compute_flow(params, SOLVER)
    print("Optimization iteration:")
    print(f"  log-params: {yk}")
    print(f"  params    : {params}")
    print(f"  flow      : {flow:.3e} m³/s\n")

if __name__ == "__main__":
    x0 = X_DEFAULT
    log_x0 = np.log(x0)

    result = minimize(
        fun=objective_log,
        x0=log_x0,
        args=(SOLVER, TARGET_FLOW),
        method='Nelder-Mead',
        options={'maxiter': 50},
        callback=optimization_callback
    )

    optimized_log = result.x
    optimized_params = np.exp(optimized_log)
    achieved_flow = compute_flow(optimized_params, SOLVER)

    print("Optimization complete.")
    print(f"  gamma   = {optimized_params[0]:.6e}")
    print(f"  gamma_a = {optimized_params[1]:.6e}")
    print(f"  gamma_R = {optimized_params[2]:.6e}")
    print(f"  net_flow = {achieved_flow:.3e} m³/s")