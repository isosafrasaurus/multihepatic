import sys
import os
import datetime
import pytz
import datetime
import scipy.optimize
import pandas as pd
import numpy as np
from graphnics import FenicsGraph

def compute_flow(x, solver):
    """Computes the flow based on the given parameters and solver.
    Args:
        x (list): List containing gamma, gamma_a, and gamma_R values.
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

def error(x_log_free, target, solver, sweep_i, x0_log_fixed):
    """Computes the squared error between the computed flow and the target.
    Args:
        x_log_free (np.ndarray): Array of free log-transformed parameters.
        target (float): The target net flow value.
        solver (fem.SubCubes): The SubCubes solver object.
        sweep_i (int): The index of the swept parameter.
        x0_log_fixed (np.ndarray): Array of fixed log-transformed parameters.
    Returns:
        float: The squared error.
    """
    # Rebuild full log-vector
    x_log_full = np.zeros_like(x0_log_fixed)
    free_idx = 0
    for i in range(len(x0_log_fixed)):
        if i == sweep_i:
            x_log_full[i] = x0_log_fixed[i]
        else:
            x_log_full[i] = x_log_free[free_idx]
            free_idx += 1
    # Log-space to real space
    x = np.exp(x_log_full)
    # Return squared error
    return (compute_flow(x, solver)[0] - target) ** 2

def sweep_variable(sweep_name, sweep_i, sweep_values, x0, target, solver, directory = None):
    """Sweeps a single variable, optimizing the others to match the target flow.
    Args:
        sweep_name (str): The name of the variable being swept.
        sweep_i (int): The index of the variable being swept in the parameter vector.
        sweep_values (list): A list of values to sweep over.
        x0 (list): The initial guess for the parameter vector.
        target (float): The target net flow value.
        solver (fem.SubCubes): The SubCubes solver object.
        directory (str, optional): Directory to save the results. Defaults to None.
    Returns:
        pandas.DataFrame: A DataFrame containing the results of the sweep.
    """
    # Convert initial guess true values into logscape
    x0_log = np.log(x0)
    df_rows = []

    # For each sweep value
    for value in sweep_values:
        # Set sweep coordinate to fixed value specified by sweep_values
        x0_log_fixed = x0_log.copy()
        x0_log_fixed[sweep_i] = np.log(value)

        # Extract just the free coordinates from x0
        x0_log_free_init = np.delete(x0_log, sweep_i)

        # Optimize only the free logs
        iter_count = [0]
        result = scipy.optimize.minimize(
            error,
            x0_log_free_init,
            args=(target, solver, sweep_i, x0_log_fixed),
            method='Nelder-Mead',
            options={'maxiter': 5},
            callback=lambda xk: (
                iter_count.__setitem__(0, iter_count[0] + 1),
                print(f"[iter {iter_count[0]}] free-log params = {xk}")
            )
        )

        # Rebuild full optimized x
        full_log = np.zeros_like(x0_log)
        free_idx = 0
        for i in range(len(full_log)):
            if i == sweep_i:
                full_log[i] = np.log(value)
            else:
                full_log[i] = result.x[free_idx]
                free_idx += 1
        x_opt = np.exp(full_log)

        # Compute flows
        flows = compute_flow(x_opt, solver)

        df_rows.append({
            sweep_name: value,
            "net_flow": flows[0],
            "lower_cube_flux_out": flows[1],
            "upper_cube_flux_in": flows[2],
            "upper_cube_flux_out": flows[3],
            "upper_cube_flux": flows[4],
            "gamma_opt": x_opt[0],
            "gamma_a_opt": x_opt[1],
            "gamma_R_opt": x_opt[2],
        })

    # Save to file and return
    df = pd.DataFrame(df_rows)
    if directory is not None:
        os.makedirs(directory, exist_ok = True)
        df.to_csv(directory, index = False)
    return df

def run_sweep(sweep_name, sweep_values, target, repo_path, directory = None):
    """Runs a sweep of a specified variable and saves the results.
    Args:
        sweep_name (str): The name of the variable to sweep.
        sweep_values (list): A list of values to sweep.
        target (float): The target net flow value.
        repo_path (str): The path to the repository containing necessary modules.
        directory (str, optional): The directory to save the results. Defaults to None.
    Returns:
        pandas.DataFrame: A DataFrame containing the results of the sweep.
    """
    # Use repo_path to import necessary modules
    SOURCE_PATH = os.path.join(repo_path, "src")
    sys.path.append(SOURCE_PATH)
    import fem
    import tissue
    
    X_DEFAULT = [
        3.587472583336982e-05,
        8.220701444028143e-08,
        8.587334091365098e-08
    ]

    # Consensus mesh resolution
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
    
    for node_id, pos in TEST_GRAPH_NODES.items():
        TEST_GRAPH.add_node(node_id, pos = pos)
    
    for (u, v, radius) in TEST_GRAPH_EDGES:
        TEST_GRAPH.add_edge(u, v, radius = radius)

    # Build mesh and submeshes
    TEST_GRAPH.make_mesh(n = TEST_NUM_NODES_EXP)
    
    TEST_GRAPH.make_submeshes()

    # Get Omega mesh from Lambda
    TEST_OMEGA, _ = tissue.get_Omega_rect(TEST_GRAPH, bounds = [[0, 0, 0], [0.05, 0.04, 0.03]])
    
    X_ZERO_PLANE = tissue.AxisPlane(0, 0.0)

    # Instantiate solver based on consensus simulation
    TEST_CUBES_SOLVER = fem.SubCubes(
        TEST_GRAPH,
        TEST_OMEGA,
        Lambda_inlet_nodes = [0],
        Omega_sink_subdomain = X_ZERO_PLANE,
        lower_cube_bounds = [[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
        upper_cube_bounds = [[0.033, 0.030, 0.010],[0.043, 0.040, 0.020]]
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

    # Run sweep and get df, then return
    df = sweep_variable(sweep_name, sweep_i, sweep_values, X_DEFAULT, target, TEST_CUBES_SOLVER, directory)
    return df