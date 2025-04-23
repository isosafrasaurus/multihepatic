#!/usr/bin/env python3
import os
import sys
import tempfile
import datetime
import pytz
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import dolfin
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from mpi4py import MPI
from graphnics import FenicsGraph




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()




cache_dir = tempfile.mkdtemp(prefix=f"dijitso_cache_{os.getpid()}_")
os.environ['DIJITSO_CACHE_DIR'] = cache_dir
os.environ['FFC_CACHE_DIR']    = cache_dir




WORK_PATH   = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join("..", "export")
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
for nid, pos in TEST_GRAPH_NODES.items():
    TEST_GRAPH.add_node(nid, pos=pos)
for u, v, r in TEST_GRAPH_EDGES:
    TEST_GRAPH.add_edge(u, v, radius=r)
TEST_GRAPH.make_mesh(n=TEST_NUM_NODES_EXP)
TEST_GRAPH.make_submeshes()

OMEGA_BOUNDS  = [[0.0, 0.0, 0.0], [0.05, 0.04, 0.03]]
OMEGA_BUILD   = tissue.OmegaBuild(TEST_GRAPH, bounds=OMEGA_BOUNDS)
X_ZERO_PLANE = tissue.AxisPlane(0, 0.0)
TEST_DOMAIN   = tissue.DomainBuild(
    TEST_GRAPH,
    OMEGA_BUILD,
    Lambda_inlet_nodes=[0],
    Omega_sink_subdomain=X_ZERO_PLANE,
)




X_DEFAULT   = [4.855e-05, 3.568e-08, 1.952e-07]  
TARGET_FLOW = 5.0e-6
LAMBDA_REG  = 1e-3
LOWER_BOUNDS = [[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]]
UPPER_BOUNDS = [[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]]




def worker_init():
    global SOLVER
    
    SOLVER = fem.SubCubes(
        domain=TEST_DOMAIN,
        lower_cube_bounds=LOWER_BOUNDS,
        upper_cube_bounds=UPPER_BOUNDS,
        order=2,
    )
    
    if os.environ.get("FENICS_PRECOMPILE", "0") == "1":
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

    res = scipy.optimize.minimize(
        fun=obj,
        x0=y0,
        method='Nelder-Mead',
        options={'maxiter': 20}
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




def sweep_variable_parallel(variable_name: str,
                            values: np.ndarray,
                            default: list,
                            max_workers=None):
    mapping = {'gamma': 0, 'gamma_a': 1, 'gamma_R': 2}
    if variable_name not in mapping:
        raise ValueError(f"Invalid variable '{variable_name}' for sweep")
    idx = mapping[variable_name]
    jobs = [(idx, v, default) for v in values]

    
    ctx = get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=worker_init
    ) as exe:
        results = list(exe.map(sweep_job, jobs))

    df = pd.DataFrame(results).set_index('value')
    df.index.name = variable_name
    return df




def plot_flow_data_semilog(df: pd.DataFrame, directory: str):
    plot_dir = os.path.join(directory, "plot_flow_data_semilog")
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime("%Y%m%d_%H%M")

    var = df.index.values
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
    varname = 'gamma'
    full_values = np.logspace(-10, 2, 20)

    
    os.environ["FENICS_PRECOMPILE"] = "1"
    worker_init()
    os.environ["FENICS_PRECOMPILE"] = "0"

    
    local_values = full_values[rank::size]
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    
    df_local = sweep_variable_parallel(varname, local_values, X_DEFAULT,
                                       max_workers=cpus)
    dict_local = df_local.to_dict('index')

    
    all_dicts = comm.gather(dict_local, root=0)

    if rank == 0:
        merged = {}
        for part in all_dicts:
            merged.update(part)
        df = pd.DataFrame.from_dict(merged, orient='index').sort_index()

        os.makedirs(EXPORT_PATH, exist_ok=True)
        ts = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime("%Y%m%d_%H%M")
        df.to_csv(os.path.join(EXPORT_PATH, f"{varname}_mpi_parallel_{ts}.csv"))

        plot_flow_data_semilog(df, EXPORT_PATH)
