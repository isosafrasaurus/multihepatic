import sys, os, pytz, datetime, math, concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphnics import FenicsGraph
from scipy.optimize import minimize

WORK_PATH = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join("..", "export")
DATA_PATH = os.path.join("..", "data")
sys.path.append(SOURCE_PATH)

import fem, tissue, visualize

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

X_DEFAULT = [2.570e-06, 1.412e-07, 3.147e-07]
X_NAME_TO_INDEX = {
    "gamma" : 0,
    "gamma_a" : 1,
    "gamma_R" : 2
}
TARGET_FLOW = 5.0e-6

def optimize_for_target(fixed_index, fixed_value, default, target_flow, lambda_reg = 1e-18, maxiter = 30):
    free_init = [val for i, val in enumerate(default) if i != fixed_index]
    free_init_log = np.log(free_init)
    y_ref = free_init_log.copy()

    # Define the objective function in log-space with regularization
    def objective_log(y):
        x = default[:]
        j = 0
        for i in range(len(x)):
            if i == fixed_index:
                x[i] = fixed_value
            else:
                x[i] = np.exp(y[j])
                j += 1
        # Objective: square difference between net flow and target
        net_flow = compute_flow(x)[0]
        cost_primary = (net_flow - target_flow)**2
        # Regularization penalty to keep y near y_ref
        cost_reg = lambda_reg * np.sum((y - y_ref)**2)
        return cost_primary + cost_reg

    result = minimize(
        objective_log,
        free_init_log,
        method='Nelder-Mead',
        options={'maxiter': maxiter}
    )
    
    x_opt = default[:]
    j = 0
    for i in range(len(x_opt)):
        if i == fixed_index:
            x_opt[i] = fixed_value
        else:
            x_opt[i] = np.exp(result.x[j])
            j += 1
    return x_opt

def compute_flow(x):
    solution = fem.SubCubes(
        domain = WORK_DOMAIN,
        gamma = x[0],
        gamma_a = x[1],
        gamma_R = x[2],
        mu = 1.0e-3,
        k_t = 1.0e-10,
        k_v = 1.543e-10,
        P_in = 100.0 * 133.322,
        P_cvp = 1.0 * 133.322,
        lower_cube_bounds = [[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
        upper_cube_bounds = [[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]]
    )
    data = [
        solution.compute_net_flow_all_dolfin(),
        solution.compute_lower_cube_flux_out(),
        solution.compute_upper_cube_flux_in(),
        solution.compute_upper_cube_flux_out(),
        solution.compute_upper_cube_flux()
    ]
    # Explicitly delete the solution to free memory used by it.
    del solution
    return data

def process_chunk(args):
    chunk, variable_index, default, variable_name = args
    return compute_chunk(chunk, variable_index, default, variable_name)

def compute_chunk(chunk, variable_index, default, variable_name):
    chunk_rows = []
    for value in chunk:
        x_opt = optimize_for_target(variable_index, value, default, TARGET_FLOW, lambda_reg = 1e-18, maxiter = 30)
        results = compute_flow(x_opt)
        data_row = {name : x_opt[X_NAME_TO_INDEX[name]] for name in X_NAME_TO_INDEX.keys()}
        data_row.update({
            "net_flow": results[0],
            "lower_cube_flux_out": results[1],
            "upper_cube_flux_in": results[2],
            "upper_cube_flux_out": results[3],
            "upper_cube_flux": results[4]
        })
        chunk_rows.append(data_row)
    return chunk_rows

def sweep_variable(variable_name, variable_values, default, directory=None, n_workers=None):
    mapping = {"gamma": 0, "gamma_a": 1, "gamma_R": 2}
    if variable_name not in mapping:
        raise ValueError("Invalid variable choice")
    variable_index = mapping[variable_name]
    
    if n_workers is None:
        n_workers = os.cpu_count() or 1
        
    # Split the list of values into chunks.
    chunk_size = math.ceil(len(variable_values) / n_workers)
    def chunkify(lst, n):
        return [lst[i:i+n] for i in range(0, len(lst), n)]
    chunks_list = chunkify(variable_values, chunk_size)
    
    rows = []
    tasks = [(chunk, variable_index, default, variable_name) for chunk in chunks_list]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init) as executor:
        for chunk_rows in executor.map(process_chunk, tasks):
            rows.extend(chunk_rows)
    
    df = pd.DataFrame(rows).set_index(variable_name)
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        cst = pytz.timezone("America/Chicago")
        now = datetime.datetime.now(cst)
        timestamp = now.strftime("%Y%m%d_%H%M")
        filename = os.path.join(directory, f"{variable_name}_sweeps_{timestamp}.csv")
        df.to_csv(filename)
    return df

def plot_flow_data_semilog(df, directory=None):
    # Create a subdirectory for saving plots.
    cst = pytz.timezone("America/Chicago")
    now = datetime.datetime.now(cst)
    timestamp = now.strftime("%Y%m%d_%H%M")

    plot_dir = os.path.join(directory, f"plot_flow_data_semilog_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)

    variable_name = df.index.name
    variable = df.index.values

    # First plot: Lower Cube Flux Out.
    plt.figure(figsize=(8, 6))
    plt.semilogx(variable, df['lower_cube_flux_out'], marker='o', linestyle='-')
    plt.xlabel(f'{variable_name} (log scale)')
    plt.ylabel('Lower Cube Flux Out')
    plt.title(f'{variable_name} vs Lower Cube Flux Out')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plot1_filename = os.path.join(plot_dir, f"lower_cube_flux_out_{timestamp}.png")
    plt.savefig(plot1_filename)
    plt.close()

    # Second plot: Upper Cube Fluxes.
    plt.figure(figsize=(8, 6))
    plt.semilogx(variable, df['upper_cube_flux_in'], marker='s', linestyle='-', label='Upper Cube Flux In')
    plt.semilogx(variable, df['upper_cube_flux_out'], marker='^', linestyle='--', label='Upper Cube Flux Out')
    plt.semilogx(variable, df['upper_cube_flux'], marker='d', linestyle='-.', label='Upper Cube Net Flux')
    plt.xlabel(f'{variable_name} (log scale)')
    plt.ylabel('Flux')
    plt.title(f'{variable_name} vs Upper Cube Flux (In, Out, and Net)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plot2_filename = os.path.join(plot_dir, f"upper_cube_flux_{timestamp}.png")
    plt.savefig(plot2_filename)
    plt.close()

def worker_init():
    global WORK_DOMAIN
    graph = FenicsGraph()
    for node_id, pos in TEST_GRAPH_NODES.items():
        graph.add_node(node_id, pos = pos)
    for (u, v, radius) in TEST_GRAPH_EDGES:
        graph.add_edge(u, v, radius = radius)
    graph.make_mesh(n = TEST_NUM_NODES_EXP)
    graph.make_submeshes()
    omega_build = tissue.OmegaBuild(graph, bounds = [[0, 0, 0], [0.05, 0.04, 0.03]])
    x_zero_plane = tissue.AxisPlane(0, 0.0)
    WORK_DOMAIN = tissue.DomainBuild(
        graph,
        omega_build,
        Lambda_inlet_nodes = [0],
        Omega_sink_subdomain = x_zero_plane
    )

if __name__ == '__main__':
    values = np.logspace(-9, 2, 100)
    data = sweep_variable("gamma", values, X_DEFAULT, directory = EXPORT_PATH)
    plot_flow_data_semilog(data, directory = EXPORT_PATH)
