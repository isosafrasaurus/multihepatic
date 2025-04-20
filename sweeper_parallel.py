#!/usr/bin/env python3
import os
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
jobid = os.environ.get("SLURM_JOB_ID", "nojob")
user = os.environ.get("USER", "user")
cache_dir = f"/scratch/{user}/dijitso_cache_{jobid}_{rank}"
os.makedirs(cache_dir, exist_ok=True)
os.environ["DIJITSO_CACHE_DIR"] = cache_dir

import sys
import numpy as np
import pandas as pd
import datetime
import pytz
import matplotlib.pyplot as plt


WORK_PATH   = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join("..", "export")
sys.path.append(SOURCE_PATH)

import fem
import tissue

from graphnics import FenicsGraph
import scipy.optimize
import dolfin  

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
OMEGA_BOUNDS = [[0.0, 0.0, 0.0], [0.05, 0.04, 0.03]]
LOWER_BOUNDS = [[0.0,    0.0,    0.0],   [0.010, 0.010, 0.010]]
UPPER_BOUNDS = [[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]]
X_DEFAULT    = [4.855e-05, 3.568e-08, 1.952e-07]  
TARGET_FLOW  = 5.0e-6
LAMBDA_REG   = 1e-3

def build_test_graph(n_nodes_exp: int) -> FenicsGraph:
    g = FenicsGraph()
    for nid, pos in TEST_GRAPH_NODES.items():
        g.add_node(nid, pos=pos)
    for u, v, r in TEST_GRAPH_EDGES:
        g.add_edge(u, v, radius=r)
    g.make_mesh(n=n_nodes_exp)
    g.make_submeshes()
    return g

def worker_init(domain, lower_bounds, upper_bounds):
    global SOLVER
    SOLVER = fem.SubCubes(
        domain=domain,
        lower_cube_bounds=lower_bounds,
        upper_cube_bounds=upper_bounds,
        order=2,
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

def compute_flow(params: np.ndarray) -> float:
    SOLVER.solve(
        gamma=params[0],
        gamma_a=params[1],
        gamma_R=params[2],
        mu=1.0e-3,
        k_t=1.0e-10,
        P_in=100.0 * 133.322,
        P_cvp=1.0 * 133.322,
    )
    try:
        dolfin.cpp.la.clear_petsc()
    except AttributeError:
        pass
    return SOLVER.compute_net_flow_all_dolfin()

def sweep_job(args):
    fixed_index, fixed_value, default = args
    y0 = np.log(default)

    def obj(y):
        x = np.exp(y)
        x[fixed_index] = fixed_value
        f = compute_flow(x)
        return (f - TARGET_FLOW)**2 + LAMBDA_REG * np.sum(y**2)

    def callback(yk):
        xk = np.exp(yk)
        xk[fixed_index] = fixed_value
        flow_k = compute_flow(xk)
        print(f"[rank={rank}] [val={fixed_value:.3e}] iter log-params={yk}")
        print(f"                   params={xk}")
        print(f"                   flow  ={flow_k:.3e}\n")

    res = scipy.optimize.minimize(
        fun=obj,
        x0=y0,
        method='Nelder-Mead',
        options={'maxiter': 20},
        callback=callback
    )

    y_opt = res.x
    x_opt = np.exp(y_opt)
    x_opt[fixed_index] = fixed_value

    
    flow      = compute_flow(x_opt)
    lower     = SOLVER.compute_lower_cube_flux_out()
    upper_in  = SOLVER.compute_upper_cube_flux_in()
    upper_out = SOLVER.compute_upper_cube_flux_out()
    upper_net = SOLVER.compute_upper_cube_flux()

    return {
        'value':      fixed_value,
        'opt_gamma':   x_opt[0],
        'opt_gamma_a': x_opt[1],
        'opt_gamma_R': x_opt[2],
        'net_flow':    flow,
        'lower_out':   lower,
        'upper_in':    upper_in,
        'upper_out':   upper_out,
        'upper_net':   upper_net,
    }

def plot_flow_data_semilog(df: pd.DataFrame, directory: str):
    plot_dir = os.path.join(directory, "plot_flow_data_semilog")
    os.makedirs(plot_dir, exist_ok=True)
    tz = pytz.timezone("America/Chicago")
    timestamp = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")

    var  = df.index.values
    name = df.index.name

    plt.figure(figsize=(8, 6))
    plt.semilogx(var, df['lower_out'], marker='o')
    plt.xlabel(f"{name} (log scale)")
    plt.ylabel("Lower Cube Flux Out")
    plt.grid(which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"lower_out_{timestamp}.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.semilogx(var, df['upper_in'],  marker='s', label='Upper In')
    plt.semilogx(var, df['upper_out'], marker='^', label='Upper Out')
    plt.semilogx(var, df['upper_net'], marker='d', label='Upper Net')
    plt.xlabel(f"{name} (log scale)")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"upper_flux_{timestamp}.png"))
    plt.close()

if __name__ == "__main__":
    
    TEST_GRAPH = build_test_graph(TEST_NUM_NODES_EXP)
    OMEGA_BUILD = tissue.OmegaBuild(TEST_GRAPH, bounds=OMEGA_BOUNDS)
    X_ZERO_PLANE = tissue.AxisPlane(0, 0.0)
    TEST_DOMAIN = tissue.DomainBuild(
        TEST_GRAPH,
        OMEGA_BUILD,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=X_ZERO_PLANE,
    )

    
    worker_init(TEST_DOMAIN, LOWER_BOUNDS, UPPER_BOUNDS)

    
    varname    = 'gamma'
    all_values = np.logspace(-10, 2, 20)
    my_values  = all_values[rank::size]
    jobs       = [(0, v, X_DEFAULT) for v in my_values]

    
    my_results = [sweep_job(args) for args in jobs]

    
    gathered = MPI.COMM_WORLD.gather(my_results, root=0)

    if rank == 0:
        
        all_results = [r for subset in gathered for r in subset]
        df = pd.DataFrame(all_results).set_index('value')
        df.index.name = varname

        
        export = os.path.join("..", "export")
        os.makedirs(export, exist_ok=True)
        tz = pytz.timezone("America/Chicago")
        ts = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")
        df.to_csv(os.path.join(export, f"{varname}_mpi_{ts}.csv"))
        plot_flow_data_semilog(df, export)

    
    MPI.COMM_WORLD.Barrier()


