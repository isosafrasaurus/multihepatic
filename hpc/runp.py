import sys
import os
import datetime
import pytz
import datetime
import scipy.optimize
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from graphnics import FenicsGraph

try:
    import fem
    import tissue
except ImportError:
    raise ImportError("Could not import fem or tissue. Make sure to add src to PYTHONPATH.")

X_DIMENSION = 3

def compute_flow(x, solver):
    """
    Computes the flow based on the given parameters and solver.
    Args:
        x (np.ndarray): Array containing gamma, gamma_a, and gamma_R values.
        solver (fem.SubCubes): The SubCubes solver object.
    Returns:
        list: A list of computed flow values.
    """
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

def error(x_log_free, sweep_i, sweep_log, target, solver):
    """
    Computes the squared error between the computed flow and the target.
    Args:
        x_log_free (np.ndarray): Subarray of free log-transformed parameters.
        sweep_i (int): The index of the swept parameter.
        sweep_log (float): Value of sweep parameter.
        target (float): The target net flow value.
        solver (fem.SubCubes): The SubCubes solver object.
    Returns:
        float: The squared error.
    """
    x_log = np.zeros(X_DIMENSION)
    free_idx = 0
    for i in range(X_DIMENSION):
        if i == sweep_i:
            x_log[i] = sweep_log
        else:
            x_log[i] = x_log_free[free_idx]
            free_idx += 1
    x = np.exp(x_log)
    return (compute_flow(x, solver)[0] - target) ** 2

def sweep_variable(sweep_name, sweep_i, sweep_values, x_default, target, solver, maxiter):
    """
    Sweeps a single variable, optimizing the others to match the target flow.
    Args:
        sweep_name (str): The name of the variable being swept.
        sweep_i (int): The index of the variable being swept in the parameter vector.
        sweep_values (list): A list of values to sweep over.
        x_default (list): The initial guess template for the parameter vector.
        target (float): The target net flow value.
        solver (fem.SubCubes): The SubCubes solver object.
    Returns:
        pandas.DataFrame: A DataFrame containing the results of the sweep.
    """
    x_default_log = np.log(x_default)
    df_rows = []
    for sweep_value in sweep_values:
        x0_log_free = np.delete(x_default_log, sweep_i)
        sweep_log = np.log(sweep_value)
        iter_count = [0]
        result = scipy.optimize.minimize(
            error,
            x0_log_free,
            args=(sweep_i, sweep_log, target, solver),
            method='Nelder-Mead',
            options={'maxiter': maxiter},
            callback=lambda xk: (
                iter_count.__setitem__(0, iter_count[0] + 1),
                print(f"[iter {iter_count[0]}] free-log params = {xk}")
            )
        )
        x_opt_log = np.zeros(X_DIMENSION)
        free_idx = 0
        for i in range(X_DIMENSION):
            if i == sweep_i:
                x_opt_log[i] = np.log(sweep_value)
            else:
                x_opt_log[i] = result.x[free_idx]
                free_idx += 1
        x_opt = np.exp(x_opt_log)
        flows = compute_flow(x_opt, solver)
        df_rows.append({
            sweep_name: sweep_value,
            "net_flow": flows[0],
            "lower_cube_flow_out": flows[1],
            "upper_cube_flow_in": flows[2],
            "upper_cube_flow_out": flows[3],
            "upper_cube_flow": flows[4],
            "gamma_opt": x_opt[0],
            "gamma_a_opt": x_opt[1],
            "gamma_R_opt": x_opt[2],
        })
    df = pd.DataFrame(df_rows)
    return df

def optimize_all_parameters(x0, target, solver, maxiter):
    """
    Optimizes gamma, gamma_a, and gamma_R so that the net flow matches the target.
    Args:
        x0 (array-like): Initial guess for [gamma, gamma_a, gamma_R].
        target (float): Desired net flow.
        solver (fem.SubCubes): The SubCubes solver object.
        maxiter (int): Maximum Nelder-Mead iterations.
    Returns:
        np.ndarray: Optimized [gamma, gamma_a, gamma_R].
    """
    x0_log = np.log(x0)

    def obj(log_x):
        x = np.exp(log_x)
        net_flow = compute_flow(x, solver)[0]
        return (net_flow - target) ** 2

    iter_count = [0]
    result = scipy.optimize.minimize(
        obj,
        x0_log,
        method='Nelder-Mead',
        options={'maxiter': maxiter},
        callback=lambda xk: (
            iter_count.__setitem__(0, iter_count[0] + 1),
            print(f"[iter {iter_count[0]}] free-log params = {xk}")
        )
    )
    x_opt = np.exp(result.x)
    return x_opt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a parameter sweep and save results.")
    parser.add_argument("--sweep_name", required=True, help="Name of the variable to sweep ('gamma', 'gamma_a', or 'gamma_R').")
    parser.add_argument("--sweep_values", required=True, help="Python expression for the list of sweep values.")
    parser.add_argument("--target", type=float, default=5.0e-6, help="Target net flow value (default: 5.0e-6).")
    parser.add_argument("--x_default", default=None, help="Python expression for the default [gamma, gamma_a, gamma_R] guess.")
    parser.add_argument("--maxiter_o", type=int, default=50, help="Maximum Nelder-Mead iterations for the sweep (default: 50).")
    parser.add_argument("--maxiter_c", type=int, default=50, help="Maximum Nelder-Mead iterations for the initial optimization (default: 50).")
    parser.add_argument("--directory", default=None, help="Directory where the plots and CSV will be saved. If omitted, results are not written to disk.")
    parser.add_argument("--num_parts", type=int, default=1, help="Total number of equal parts into which sweep_values is divided.")
    parser.add_argument("--part_idx", type=int, default=0, help="Index of the part to execute in this run.")
    parser.add_argument("--voxel_res", type=float, default=0.002, help="Resolution of 3D mesh voxel to use (default: 0.002).")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        raise ValueError(f"{args.directory} is not a valid directory.")

    # Build consensus mesh and solver
    test_num_nodes_exp = 5
    test_graph = FenicsGraph()
    test_graph_nodes = {
        0: [0.000, 0.020, 0.015],
        1: [0.010, 0.020, 0.015],
        2: [0.022, 0.013, 0.015],
        3: [0.022, 0.028, 0.015],
        4: [0.015, 0.005, 0.015],
        5: [0.015, 0.035, 0.015],
        6: [0.038, 0.005, 0.015],
        7: [0.038, 0.035, 0.015]
    }
    test_graph_edges = [
        (0, 1, 0.004),
        (1, 2, 0.003),
        (1, 3, 0.003),
        (2, 4, 0.002),
        (2, 6, 0.003),
        (3, 5, 0.002),
        (3, 7, 0.003)
    ]
    for node_id, pos in test_graph_nodes.items():
        test_graph.add_node(node_id, pos=pos)
    for u, v, radius in test_graph_edges:
        test_graph.add_edge(u, v, radius=radius)
    test_graph.make_mesh(n=test_num_nodes_exp)
    test_graph.make_submeshes()
    test_Omega, _ = tissue.get_Omega_rect_from_res(
        test_graph,
        bounds=[[0, 0, 0], [0.05, 0.04, 0.03]],
        voxel_res=args.voxel_res
    )
    x_zero_plane = tissue.AxisPlane(0, 0.0)
    test_cubes_solver = fem.SubCubes(
        test_graph,
        test_Omega,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=x_zero_plane,
        lower_cube_bounds=[[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
        upper_cube_bounds=[[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]]
    )

    # Evaluate string args that are actually Python expressions
    sweep_values = eval(args.sweep_values)
    x_default = eval(args.x_default) if args.x_default is not None else None

    if args.num_parts < 1:
        raise ValueError("num_parts must be at least 1")
    if not 0 <= args.part_idx < args.num_parts:
        raise ValueError("part_idx must be between 0 and num_parts-1")

    # Obtain subarray based on num_parts and part_idx
    n_total = len(sweep_values)
    base_size = n_total // args.num_parts
    remainder = n_total % args.num_parts
    start = args.part_idx * base_size + min(args.part_idx, remainder)
    end = start + base_size + (1 if args.part_idx < remainder else 0)
    sweep_values = sweep_values[start:end]

    # If no x_default, optimize to find one
    if x_default is None:
        x_default = optimize_all_parameters(
            [1e-1] * X_DIMENSION,
            args.target,
            test_cubes_solver,
            maxiter=args.maxiter_c
        )

    # Resolve variable name to index
    sweep_i = -1
    match args.sweep_name:
        case "gamma":
            sweep_i = 0
        case "gamma_a":
            sweep_i = 1
        case "gamma_R":
            sweep_i = 2
        case _:
            raise ValueError("Invalid sweep_name variable choice")

    df = sweep_variable(
        args.sweep_name,
        sweep_i,
        sweep_values,
        x_default,
        args.target,
        test_cubes_solver,
        args.maxiter_o
    )

    # Save df to file
    df.to_csv(os.path.join(args.directory, f"part_{args.part_idx}_of_{args.num_parts - 1}.csv"), index = False)
