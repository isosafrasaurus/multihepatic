#!/usr/bin/env python3
import os
import sys
import tempfile
import datetime
import pytz
import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
import dolfin


WORK_PATH   = os.path.join(os.getcwd(), "3d-1d")
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join("..", "export")
DATA_PATH   = os.path.join("..", "data")
sys.path.append(SOURCE_PATH)

import fem
import tissue
from graphnics import FenicsGraph


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


cores_per_solver = 4
if size % cores_per_solver != 0:
    raise RuntimeError(f"MPI world size {size} not divisible by {cores_per_solver}")


n_solvers = size // cores_per_solver


solver_id   = rank // cores_per_solver
solver_comm = comm.Split(color=solver_id, key=rank)
sol_rank    = solver_comm.Get_rank()
sol_size    = solver_comm.Get_size()


dolfin.MPI.comm_world = solver_comm


os.environ.update({
    "OMP_NUM_THREADS":        "1",
    "MKL_NUM_THREADS":        "1",
    "OPENBLAS_NUM_THREADS":   "4",
    "VECLIB_MAXIMUM_THREADS": "4",
})

jobid     = os.environ.get("SLURM_JOB_ID", "nojob")
user      = os.environ.get("USER", "")
cache_dir = tempfile.mkdtemp(prefix=f"dijitso_cache_{user}_{jobid}_{rank}_")
os.environ["DIJITSO_CACHE_DIR"] = cache_dir
os.environ["FFC_CACHE_DIR"]    = cache_dir


TARGET_FLOW   = 5.0e-6
LAMBDA_REG    = 1e-10
X_DEFAULT     = [4.855e-05, 3.568e-08, 1.952e-07]
mu, k_t       = 1.0e-3, 1.0e-10
P_in          = 100.0 * 133.322
P_cvp         =   1.0 * 133.322

gamma_values = np.logspace(-10, 2, 50)

gamma_chunks = np.array_split(gamma_values, n_solvers)
my_gammas    = gamma_chunks[solver_id]

TEST_NUM_NODES_EXP = 7
NODE_POSITIONS = {
    0: [0.000, 0.020, 0.015],
    1: [0.010, 0.020, 0.015],
    2: [0.022, 0.013, 0.015],
    3: [0.022, 0.028, 0.015],
    4: [0.015, 0.005, 0.015],
    5: [0.015, 0.035, 0.015],
    6: [0.038, 0.005, 0.015],
    7: [0.038, 0.035, 0.015],
}
EDGE_LIST = [
    (0, 1, 0.004),
    (1, 2, 0.003),
    (1, 3, 0.003),
    (2, 4, 0.002),
    (2, 6, 0.003),
    (3, 5, 0.002),
    (3, 7, 0.003),
]
OMEGA_BOUNDS     = [[0.0,0.0,0.0], [0.05,0.04,0.03]]
LOWER_CUBE_BOUNDS = [[0.0,0.0,0.0], [0.010,0.010,0.010]]
UPPER_CUBE_BOUNDS = [[0.033,0.030,0.010], [0.043,0.040,0.020]]

def build_graph_and_omega(n_exp):
    G = FenicsGraph()
    for nid, pos in NODE_POSITIONS.items():
        G.add_node(nid, pos=pos)
    for u, v, r in EDGE_LIST:
        G.add_edge(u, v, radius=r)
    G.make_mesh(n=n_exp)
    G.make_submeshes()
    Omega, _ = tissue.get_Omega_rect(G, bounds=OMEGA_BOUNDS)
    return G, Omega

def plot_flow_data_semilog(df: pd.DataFrame, directory: str):
    plot_dir = os.path.join(directory, "plot_flow_data_semilog")
    os.makedirs(plot_dir, exist_ok=True)
    tz = pytz.timezone("America/Chicago")
    ts = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")

    plt.figure(figsize=(8,6))
    plt.semilogx(df.index, df["lower_out"], marker="o")
    plt.xlabel("γ (log scale)")
    plt.ylabel("Lower Cube Flux Out")
    plt.grid(which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"lower_out_{ts}.png"))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.semilogx(df.index, df["upper_in"],  marker="s", label="Upper In")
    plt.semilogx(df.index, df["upper_out"], marker="^", label="Upper Out")
    plt.semilogx(df.index, df["upper_net"], marker="d", label="Upper Net")
    plt.xlabel("γ (log scale)")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"upper_flux_{ts}.png"))
    plt.close()

def main():
    G, Omega = build_graph_and_omega(TEST_NUM_NODES_EXP)
    inlet_plane = tissue.AxisPlane(0, 0.0)

    solver = fem.SubCubes(
        G,
        Omega,
        Lambda_inlet_nodes    = [0],
        Omega_sink_subdomain  = inlet_plane,
        lower_cube_bounds     = LOWER_CUBE_BOUNDS,
        upper_cube_bounds     = UPPER_CUBE_BOUNDS,
        order                 = 2
    )

    local_results = []

    for gamma in my_gammas:
        y0 = np.log([X_DEFAULT[1], X_DEFAULT[2]])

        def obj(y):
            ga, gR = np.exp(y)
            solver.solve(
                gamma   = gamma,
                gamma_a = ga,
                gamma_R = gR,
                mu      = mu,
                k_t     = k_t,
                P_in    = P_in,
                P_cvp   = P_cvp
            )
            try:
                dolfin.cpp.la.clear_petsc()
            except AttributeError:
                pass
            flow = solver.compute_net_flow_all_dolfin()
            return (flow - TARGET_FLOW)**2 + LAMBDA_REG * np.sum(y**2)

        res = scipy.optimize.minimize(
            obj,
            y0,
            method  = "Nelder-Mead",
            options = {"maxiter": 30}
        )
        ga_opt, gR_opt = np.exp(res.x)

        
        solver.solve(
            gamma   = gamma,
            gamma_a = ga_opt,
            gamma_R = gR_opt,
            mu      = mu,
            k_t     = k_t,
            P_in    = P_in,
            P_cvp   = P_cvp
        )
        try:
            dolfin.cpp.la.clear_petsc()
        except AttributeError:
            pass

        
        flow      = solver.compute_net_flow_all_dolfin()
        lower_out = solver.compute_lower_cube_flux_out()
        upper_in  = solver.compute_upper_cube_flux_in()
        upper_out = solver.compute_upper_cube_flux_out()
        upper_net = solver.compute_upper_cube_flux()

        local_results.append({
            "gamma":     gamma,
            "gamma_a":   ga_opt,
            "gamma_R":   gR_opt,
            "net_flow":  flow,
            "lower_out": lower_out,
            "upper_in":  upper_in,
            "upper_out": upper_out,
            "upper_net": upper_net,
        })

    
    if sol_rank == 0:
        group_results = local_results
    else:
        group_results = None

    
    all_groups = comm.gather(group_results, root=0)

    if rank == 0:
        flat = [entry for grp in all_groups if grp for entry in grp]
        df   = pd.DataFrame(flat).set_index("gamma")

        EXPORT_PATH = os.path.join(os.getcwd(), "export")
        os.makedirs(EXPORT_PATH, exist_ok=True)
        tz  = pytz.timezone("America/Chicago")
        ts  = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")
        csv_path = os.path.join(EXPORT_PATH, f"gamma_mpi_{ts}.csv")
        df.to_csv(csv_path)
        print(f"Wrote results to {csv_path}")

        plot_flow_data_semilog(df, EXPORT_PATH)

if __name__ == "__main__":
    main()

