# consensus_sweep.py

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
        gamma = x[0],
        gamma_a = x[1],
        gamma_R = x[2],
        mu = 1.0e-3,
        k_t = 1.0e-10,
        P_in = 100.0 * 133.322,
        P_cvp = 1.0 * 133.322
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
    # Rebuild full x parameter vector
    x_log = np.zeros(X_DIMENSION)
    free_idx = 0
    for i in range(X_DIMENSION):
        if i == sweep_i:
            x_log[i] = sweep_log
        else:
            x_log[i] = x_log_free[free_idx]
            free_idx += 1
    # Log-space to real space
    x = np.exp(x_log)
    # Return squared error
    return (compute_flow(x, solver)[0] - target) ** 2

def sweep_variable(sweep_name, sweep_i, sweep_values, x_default, target, solver, maxiter, save_path):
    """
    Sweeps a single variable, optimizing the others to match the target flow.
    Args:
        sweep_name (str): The name of the variable being swept.
        sweep_i (int): The index of the variable being swept in the parameter vector.
        sweep_values (list): A list of values to sweep over.
        x_default (list): The initial guess template for the parameter vector.
        target (float): The target net flow value.
        solver (fem.SubCubes): The SubCubes solver object.
        directory (str, optional): Directory to save the results. Defaults to None.
    Returns:
        pandas.DataFrame: A DataFrame containing the results of the sweep.
    """
    # Convert default template values into log-space
    x_default_log = np.log(x_default)
    df_rows = []

    # For each sweep value
    for sweep_value in sweep_values:
        x0_log_free = np.delete(x_default_log, sweep_i)
        sweep_log = np.log(sweep_value)
        # Optimize only the free logs
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

        # Rebuild full optimized x
        x_opt_log = np.zeros(X_DIMENSION)
        free_idx = 0
        for i in range(X_DIMENSION):
            if i == sweep_i:
                x_opt_log[i] = np.log(sweep_value)
            else:
                x_opt_log[i] = result.x[free_idx]
                free_idx += 1
        x_opt = np.exp(x_opt_log)

        # Compute flows
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

    # Save to file and return
    df = pd.DataFrame(df_rows)
    if save_path is not None:
        df.to_csv(save_path, index = False)
    return df

def optimize_all_parameters(x0, target, solver, maxiter):
    """
    Optimizes gamma, gamma_a, and gamma_R so that the net flow matches the target.
    
    Args:
        x0 (array-like, optional): Initial guess for [gamma, gamma_a, gamma_R].
        target (float): Desired net flow.
        solver (fem.SubCubes): The SubCubes solver object.
        maxiter (int, optional): Maximum Nelder-Mead iterations.
    
    Returns:
        np.ndarray: Optimized [gamma, gamma_a, gamma_R].
    """
    x0_log = np.log(x0)

    # Objective: squared error on net flow
    def obj(log_x):
        x = np.exp(log_x)
        net_flow = compute_flow(x, solver)[0]
        return (net_flow - target) ** 2
    
    iter_count = [0]
    # Run Nelder-Mead in log-space
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

def run_sweep(sweep_name, sweep_values, target, x_default = None, maxiter_o = 50, maxiter_c = 50, directory = None):
    """
    Main driver. Runs a sweep of a specified variable and saves the results.
    Args:
        sweep_name (str): The name of the variable to sweep.
        sweep_values (list): A list of values to sweep.
        target (float): The target net flow value.
        directory (str, optional): The directory to save the results. Defaults to None.
    Returns:
        pandas.DataFrame: A DataFrame containing the results of the sweep.
    """    
    # Consensus mesh resolution
    test_num_nodes_exp = 5
    
    # Consensus mesh
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
        test_graph.add_node(node_id, pos = pos)
    for (u, v, radius) in test_graph_edges:
        test_graph.add_edge(u, v, radius = radius)

    # Build mesh and submeshes
    test_graph.make_mesh(n = test_num_nodes_exp)
    test_graph.make_submeshes()

    # Get Omega mesh from Lambda
    test_Omega, _ = tissue.get_Omega_rect(test_graph, bounds = [[0, 0, 0], [0.05, 0.04, 0.03]])
    
    x_zero_plane = tissue.AxisPlane(0, 0.0)

    # Instantiate solver based on consensus simulation
    test_cubes_solver = fem.SubCubes(
        test_graph,
        test_Omega,
        Lambda_inlet_nodes = [0],
        Omega_sink_subdomain = x_zero_plane,
        lower_cube_bounds = [[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
        upper_cube_bounds = [[0.033, 0.030, 0.010],[0.043, 0.040, 0.020]]
    )

    # Get default
    if x_default is None:
        x_default = optimize_all_parameters(
            [1e-1] * X_DIMENSION,
            target,
            test_cubes_solver,
            maxiter = maxiter_c
        )
    
    # Resolve sweep variable's name to index
    sweep_i = -1
    match sweep_name:
        case "gamma":
            sweep_i = 0
        case "gamma_a":
            sweep_i = 1
        case "gamma_R":
            sweep_i = 2
        case _:
            raise ValueError("Invalid sweep_name variable choice")

    if directory is not None:
        # Get CST timestamp
        tz = pytz.timezone("America/Chicago")
        ts = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")

        # Make sub-directory within directory for results
        real_directory = os.path.join(directory, f"{sweep_name}_sweep_{ts}")
        os.makedirs(real_directory, exist_ok = True)

        df = sweep_variable(sweep_name, sweep_i, sweep_values, x_default, target, test_cubes_solver, maxiter_o, os.path.join(real_directory, "data.csv"))
        
        x = df[sweep_name]
        lower_out = df["lower_cube_flow_out"]

        plt.figure(figsize=(8,5))
        plt.plot(x, lower_out, marker='o', linestyle='-', label='Lower cube flow out')
        plt.xscale('log')
        plt.xlabel(sweep_name)
        plt.ylabel('Lower cube flow out')
        plt.title(f'Lower cube flux vs {sweep_name}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.1)
        plt.tight_layout()
        lower_path = os.path.join(real_directory, f"lower_cube_plot.pdf")
        plt.savefig(lower_path)
        plt.close()
        print(f"Lower-cube plot saved to {lower_path}")

        upper_in  = df["upper_cube_flow_in"]
        upper_out = df["upper_cube_flow_out"]
        upper_tot = df["upper_cube_flow"]

        plt.figure(figsize=(8,5))
        plt.plot(x, upper_in,  marker='s', linestyle='--', label='Flux in')
        plt.plot(x, upper_out, marker='^', linestyle='-.', label='Flux out')
        plt.plot(x, upper_tot, marker='d', linestyle=':',  label='Total flux')
        plt.xscale('log')
        plt.xlabel(sweep_name)
        plt.ylabel('Upper cube flux [units]')
        plt.title(f'Upper cube fluxes vs {sweep_name}')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.1)
        plt.tight_layout()
        upper_path = os.path.join(real_directory, f"upper_cube_plot.pdf")
        plt.savefig(upper_path)
        plt.close()
        print(f"Upper-cube plot saved to {upper_path}")
    else:
        # Run sweep and get df, then return
        df = sweep_variable(sweep_name, sweep_i, sweep_values, x_default, target, test_cubes_solver, maxiter_o, save_path)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a parameter sweep and save results."
    )
    parser.add_argument(
        "--sweep_name",
        required=True,
        help="Name of the variable to sweep ('gamma', 'gamma_a', or 'gamma_R')."
    )
    parser.add_argument(
        "--sweep_values",
        required=True,
        help=(
            "Python expression for the list of sweep values, "
        )
    )
    parser.add_argument(
        "--target",
        type=float,
        default=5.0e-6,
        help="Target net flow value (default: 5.0e-6)."
    )
    parser.add_argument(
        "--x_default",
        default=None,
        help=(
            "Python expression for the default [gamma, gamma_a, gamma_R] guess."
        )
    )
    parser.add_argument(
        "--maxiter_o",
        type=int,
        default=50,
        help="Maximum Nelder-Mead iterations for the sweep (default: 50)."
    )
    parser.add_argument(
        "--maxiter_c",
        type=int,
        default=50,
        help="Maximum Nelder–Mead iterations for the initial optimization (default: 50)."
    )
    parser.add_argument(
        "--directory",
        default=None,
        help=(
            "Directory where the plots and CSV will be saved. If omitted, results are not written to disk."
        )
    )

    args = parser.parse_args()

    # Evaluate the code strings into real Python objects
    sweep_values = eval(args.sweep_values)
    x_default = eval(args.x_default) if args.x_default is not None else None

    # Run the sweep
    run_sweep(
        sweep_name = args.sweep_name,
        sweep_values = sweep_values,
        target = args.target,
        x_default = x_default,
        maxiter_o = args.maxiter_o,
        maxiter_c = args.maxiter_c,
        directory = args.directory
    )