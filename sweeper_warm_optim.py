import sys, os, pytz, datetime, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphnics import FenicsGraph
from scipy.optimize import minimize

# Paths and settings.
WORK_PATH = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join("..", "export2")
DATA_PATH = os.path.join("..", "data")
sys.path.append(SOURCE_PATH)

import fem, tissue, visualize

# Experiment settings.
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

# Default parameter values:
# X_DEFAULT = [gamma, gamma_a, gamma_R, k_v]
X_DEFAULT = [2.570e-06, 1.412e-07, 3.147e-07, 1.543e-10]

# Target net flow for optimization.
TARGET_FLOW = 5.0e-6

def create_domain():
    """Recreate the domain based on the provided graph and submeshes."""
    # Create graph.
    graph = FenicsGraph()
    for node_id, pos in TEST_GRAPH_NODES.items():
        graph.add_node(node_id, pos=pos)
    for (u, v, radius) in TEST_GRAPH_EDGES:
        graph.add_edge(u, v, radius=radius)
    
    graph.make_mesh(n=TEST_NUM_NODES_EXP)
    graph.make_submeshes()
    
    omega_build = tissue.OmegaBuild(graph, bounds=[[0,0,0], [0.05, 0.04, 0.03]])
    x_zero_plane = tissue.AxisPlane(0, 0.0)
    
    domain = tissue.DomainBuild(
        graph,
        omega_build,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=x_zero_plane
    )
    return domain

# Initialize the global domain.
WORK_DOMAIN = create_domain()

def optimize_for_target(fixed_index, fixed_value, default, target, y_prev=None, lambda_reg=1e-3, maxiter=30):
    # Identify free parameters (those not being fixed).
    free_params = [default[i] for i in range(len(default)) if i != fixed_index]
    if y_prev is None:
        # If no previous parameter is provided, initialize using the default values (converted to log-space).
        y_prev = np.log(free_params)
    # Use y_prev as the initial guess.
    free_init_log = y_prev.copy()
    
    def objective_log(y):
        # Reconstruct the full parameter vector from free parameters in log-space.
        x = default[:]
        j = 0
        for i in range(len(x)):
            if i == fixed_index:
                x[i] = fixed_value
            else:
                x[i] = np.exp(y[j])
                j += 1
        net_flow = compute_flow(x)[0]
        # The cost: the squared error in the net flow plus a regularization term to promote continuity.
        return (net_flow - target)**2 + lambda_reg * np.sum((y - y_prev)**2)
    
    result = minimize(
        objective_log,
        free_init_log,
        method='Nelder-Mead',
        options={'maxiter': maxiter}
    )
    
    # Reconstruct the optimized full parameter vector.
    x_opt = default[:]
    j = 0
    optimized_y = []
    for i in range(len(x_opt)):
        if i == fixed_index:
            x_opt[i] = fixed_value
        else:
            val = np.exp(result.x[j])
            x_opt[i] = val
            optimized_y.append(result.x[j])
            j += 1
    return x_opt, np.array(optimized_y)

def compute_flow(x):
    solution = fem.SubCubes(
        domain=WORK_DOMAIN,
        gamma=x[0],
        gamma_a=x[1],
        gamma_R=x[2],
        mu=1.0e-3,
        k_t=1.0e-10,
        k_v=x[3],
        P_in=100.0 * 133.322,
        P_cvp=1.0 * 133.322,
        lower_cube_bounds=[[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
        upper_cube_bounds=[[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]]
    )
    data = [
        solution.compute_net_flow_all_dolfin(),
        solution.compute_lower_cube_flux_out(),
        solution.compute_upper_cube_flux_in(),
        solution.compute_upper_cube_flux_out(),
        solution.compute_upper_cube_flux()
    ]
    return data

def sweep_variable_sequential(variable_name, variable_values, default, directory=None):
    mapping = {"gamma": 0, "gamma_a": 1, "gamma_R": 2, "k_v": 3}
    if variable_name not in mapping:
        raise ValueError("Invalid variable choice")
    variable_index = mapping[variable_name]
    
    # Initialize for warm starting.
    y_prev = None
    results_rows = []
    
    for value in variable_values:
        # Optimize free parameters given the current fixed value,
        # warm-starting with y_prev and using regularization.
        x_opt, y_prev = optimize_for_target(variable_index, value, default, TARGET_FLOW,
                                            y_prev=y_prev, lambda_reg=1e-3, maxiter=30)
        flow_results = compute_flow(x_opt)
        results_rows.append({
            variable_name: value,
            "net_flow": flow_results[0],
            "lower_cube_flux_out": flow_results[1],
            "upper_cube_flux_in": flow_results[2],
            "upper_cube_flux_out": flow_results[3],
            "upper_cube_flux": flow_results[4]
        })
        print(f"Swept {variable_name} = {value:.3e} -> net_flow = {flow_results[0]:.3e}")
    
    df = pd.DataFrame(results_rows).set_index(variable_name)
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        cst = pytz.timezone("America/Chicago")
        now = datetime.datetime.now(cst)
        timestamp = now.strftime("%Y%m%d_%H%M")
        filename = os.path.join(directory, f"{variable_name}_sweeps_{timestamp}.csv")
        df.to_csv(filename)
    return df

def plot_flow_data_semilog(df, directory=None):
    plot_dir = os.path.join(directory, "plot_flow_data_semilog")
    os.makedirs(plot_dir, exist_ok=True)
    cst = pytz.timezone("America/Chicago")
    now = datetime.datetime.now(cst)
    timestamp = now.strftime("%Y%m%d_%H%M")

    variable_name = df.index.name
    variable = df.index.values

    # Plot: Lower Cube Flux Out.
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

    # Plot: Upper Cube Fluxes.
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

if __name__ == '__main__':
    # Sweep the "gamma" parameter over a logarithmic space.
    values = np.logspace(-10, 2, 20)
    df = sweep_variable_sequential("gamma", values, X_DEFAULT, directory=EXPORT_PATH)
    plot_flow_data_semilog(df, directory=EXPORT_PATH)
