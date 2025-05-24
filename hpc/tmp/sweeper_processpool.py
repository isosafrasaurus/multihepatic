#!/usr/bin/env python3
import sys, os
import numpy as np
import pytz
import datetime
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
from graphnics import FenicsGraph
from concurrent.futures import ProcessPoolExecutor

import os
from multiprocessing import get_context


os.environ["DIJITSO_CACHE_DIR"] = f"/scratch/{os.environ['USER']}/dijitso_cache_{os.getpid()}"

WORK_PATH   = os.path.join(os.getcwd(), "3d-1d")
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join(WORK_PATH, "export")

sys.path.append(SOURCE_PATH)

import fem, tissue

TEST_NUM_NODES_EXP = 5

TEST_GRAPH_NODES = {
    0: [0.000, 0.020, 0.015],
    1: [0.010, 0.020, 0.015],
    2: [0.022, 0.013, 0.015],
    3: [0.022, 0.028, 0.015],
    4: [0.015, 0.005, 0.015],
    5: [0.015, 0.035, 0.015],
    6: [0.038, 0.005, 0.015],
    7: [0.038, 0.035, 0.015]
}

TEST_GRAPH_EDGES = [
    (0, 1, 0.004),
    (1, 2, 0.003),
    (1, 3, 0.003),
    (2, 4, 0.002),
    (2, 6, 0.003),
    (3, 5, 0.002),
    (3, 7, 0.003)
]

LOWER_CUBE_BOUNDS = [[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]]
UPPER_CUBE_BOUNDS = [[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]]
LAMBDA_INLET_NODES = [0]
X_ZERO_PLANE = tissue.AxisPlane(0, 0.0)


def create_solver():
    graph = FenicsGraph()
    for node_id, pos in TEST_GRAPH_NODES.items():
        graph.add_node(node_id, pos=pos)
    for u, v, radius in TEST_GRAPH_EDGES:
        graph.add_edge(u, v, radius=radius)
    graph.make_mesh(n=TEST_NUM_NODES_EXP)
    graph.make_submeshes()

    Omega, _ = tissue.get_Omega_rect_from_res(
        graph,
        bounds=[[0, 0, 0], [0.05, 0.04, 0.03]],
        voxel_res=0.001
    )

    solver = fem.SubCubes(
        graph,
        Omega,
        Lambda_inlet_nodes=LAMBDA_INLET_NODES,
        Omega_sink_subdomain=X_ZERO_PLANE,
        lower_cube_bounds=LOWER_CUBE_BOUNDS,
        upper_cube_bounds=UPPER_CUBE_BOUNDS
    )
    return solver


TEST_CUBES_SOLVER = create_solver()
TEST_CUBES_SOLVER.solve(
    gamma=3.6145827741262347e-05,
    gamma_a=8.225197366649115e-08,
    gamma_R=8.620057937882969e-08,
    mu=1.0e-3,
    k_t=1.0e-10,
    P_in=100.0 * 133.322,
    P_cvp=1.0 * 133.322
)


def compute_flow(x, solver):
    solver.solve(
        gamma=x[0],
        gamma_a=x[1],
        gamma_R=x[2],
        mu=1.0e-3,
        k_t=1.0e-10,
        P_in=100.0 * 133.322,
        P_cvp=1.0 * 133.322
    )
    return [
        solver.compute_net_flow_all_dolfin(),
        solver.compute_lower_cube_flux_out(),
        solver.compute_upper_cube_flux_in(),
        solver.compute_upper_cube_flux_out(),
        solver.compute_upper_cube_flux()
    ]


def objective_log_free(y_free, solver, target, fixed_index, fixed_value, free_indices):
    log_x = np.zeros(3)
    log_x[fixed_index] = np.log(fixed_value)
    for i, idx in enumerate(free_indices):
        log_x[idx] = y_free[i]
    x = np.exp(log_x)
    net_flow = compute_flow(x, solver)[0]
    return (net_flow - target) ** 2


def optimization_callback(yk, fixed_index, fixed_value, free_indices):
    log_x = np.zeros(3)
    log_x[fixed_index] = np.log(fixed_value)
    for i, idx in enumerate(free_indices):
        log_x[idx] = yk[i]
    current_params = np.exp(log_x)
    print("Optimization iteration:")
    print(f"  Fixed index {fixed_index} at value {fixed_value}")
    print(f"  Current log_params: {log_x}")
    print(f"  Current parameters (gamma, gamma_a, gamma_R): {current_params}")


def _sweep_variable_worker(args):
    variable_name, variable_index, free_indices, default, target_flow, value = args

    
    solver = create_solver()

    fixed_value = value
    y0 = np.log(default)[free_indices]

    result = scipy.optimize.minimize(
        objective_log_free,
        y0,
        args=(solver, target_flow, variable_index, fixed_value, free_indices),
        method='Nelder-Mead',
        options={'maxiter': 20},
        callback=lambda yk, fi=variable_index, fv=fixed_value, fi_list=free_indices:
                 optimization_callback(yk, fi, fv, fi_list)
    )

    log_x_opt = np.zeros(3)
    log_x_opt[variable_index] = np.log(fixed_value)
    for i, idx in enumerate(free_indices):
        log_x_opt[idx] = result.x[i]
    x_opt = np.exp(log_x_opt)

    flows = compute_flow(x_opt, solver)
    return {
        variable_name:           value,
        "net_flow":              flows[0],
        "lower_cube_flux_out":   flows[1],
        "upper_cube_flux_in":    flows[2],
        "upper_cube_flux_out":   flows[3],
        "upper_cube_flux":       flows[4],
        "gamma_opt":             x_opt[0],
        "gamma_a_opt":           x_opt[1],
        "gamma_R_opt":           x_opt[2],
    }


def sweep_variable(variable_name, variable_values, default, solver,
                   directory=None, target_flow=5.0e-6):
    match variable_name:
        case "gamma":
            variable_index = 0
        case "gamma_a":
            variable_index = 1
        case "gamma_R":
            variable_index = 2
        case _:
            raise ValueError("Invalid variable choice")

    free_indices = [i for i in range(3) if i != variable_index]

    args_list = [
        (variable_name, variable_index, free_indices, default, target_flow, value)
        for value in variable_values
    ]

    
    ctx = get_context("spawn")
    with ProcessPoolExecutor(mp_context=ctx) as executor:
        rows = list(executor.map(_sweep_variable_worker, args_list))

    df = pd.DataFrame(rows).set_index(variable_name)

    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        cst = pytz.timezone("America/Chicago")
        now = datetime.datetime.now(cst)
        timestamp = now.strftime("%Y%m%d_%H%M")

        
        csv_path = os.path.join(directory, f"{variable_name}_sweeps_{timestamp}.csv")
        df.to_csv(csv_path)

        
        plots_dir = os.path.join(directory, f"{variable_name}_plots_{timestamp}")
        os.makedirs(plots_dir, exist_ok=True)

        
        x = df.index.values
        plt.figure(figsize=(8, 6))
        plt.semilogx(x, df['lower_cube_flux_out'], marker='o', linestyle='-')
        plt.xlabel(f"{variable_name} (log scale)")
        plt.ylabel('Lower Cube Flux Out')
        plt.title(f"{variable_name} vs Lower Cube Flux Out")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{variable_name}_lower_cube_flux_out.png"))
        plt.close()

        
        plt.figure(figsize=(8, 6))
        plt.semilogx(x, df['upper_cube_flux_in'],  marker='s', linestyle='-',  label='Upper Cube Flux In')
        plt.semilogx(x, df['upper_cube_flux_out'], marker='^', linestyle='--', label='Upper Cube Flux Out')
        plt.semilogx(x, df['upper_cube_flux'],     marker='d', linestyle='-.', label='Upper Cube Net Flux')
        plt.xlabel(f"{variable_name} (log scale)")
        plt.ylabel('Flux')
        plt.title(f"{variable_name} vs Upper Cube Flux (In, Out, Net)")
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{variable_name}_upper_cube_flux.png"))
        plt.close()

    return df


if __name__ == "__main__":
    x_default = [
        3.587472583336982e-05,
        8.220701444028143e-08,
        8.587334091365098e-08
    ]

    data = sweep_variable(
        "gamma",
        np.logspace(-10, 2, 50),
        x_default,
        TEST_CUBES_SOLVER,
        directory=EXPORT_PATH,
        target_flow=5.0e-6
    )

    print(data.head())

