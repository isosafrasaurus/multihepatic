#!/usr/bin/env python3
import os, sys, tempfile, time, glob
import numpy as np
import scipy.optimize
import pandas as pd
import dolfin
from graphnics import FenicsGraph
import matplotlib.pyplot as plt
import datetime, pytz

# Slurm array env
TASK_ID   = int(os.environ['SLURM_ARRAY_TASK_ID'])
NUM_TASKS = int(os.environ['SLURM_ARRAY_TASK_COUNT'])

# Unique cache per task
dir_here = os.path.abspath(os.path.dirname(__file__))
jobid    = os.environ.get("SLURM_JOB_ID", "nojob")
user     = os.environ.get("USER", "")
cache    = tempfile.mkdtemp(prefix=f"dijitso_cache_{user}_{jobid}_{TASK_ID}_")
os.environ['DIJITSO_CACHE_DIR'] = cache
os.environ['FFC_CACHE_DIR']    = cache

# Paths
HERE        = dir_here
SOURCE_PATH = os.path.join(HERE, "src")
EXPORT_PATH = os.path.join(HERE, "..", "export")
sys.path.append(SOURCE_PATH)

import fem, tissue

# Constants
TARGET_FLOW = 5e-6
LAMBDA_REG  = 1e-3
X_DEFAULT   = [4.855e-05, 3.568e-08, 1.952e-07]
mu, k_t     = 1e-3, 1e-10
P_in        = 100.0 * 133.322
P_cvp       = 1.0   * 133.322

# Build network and domain
def build_graph_and_domain(exp=7):
    nodes = {
        0:[0.000,0.020,0.015], 1:[0.010,0.020,0.015],
        2:[0.022,0.013,0.015], 3:[0.022,0.028,0.015],
        4:[0.015,0.005,0.015], 5:[0.015,0.035,0.015],
        6:[0.038,0.005,0.015], 7:[0.038,0.035,0.015]
    }
    edges = [
        (0,1,0.004),(1,2,0.003),(1,3,0.003),
        (2,4,0.002),(2,6,0.003),(3,5,0.002),(3,7,0.003)
    ]
    G = FenicsGraph()
    for nid, pos in nodes.items():
        G.add_node(nid, pos=pos)
    for u, v, r in edges:
        G.add_edge(u, v, radius=r)

    # Try mesh refinement, fallback on manual connectivity init
    try:
        G.make_mesh(n=exp)
    except RuntimeError:
        mesh, mf = G.get_mesh(exp)
        mesh.init(1)
        G.mesh = mesh
        G.mesh_function = mf
    G.make_submeshes()

    omega = tissue.OmegaBuild(
        G,
        bounds=[[0,0,0],[0.05,0.04,0.03]],
        voxel_dim=(32,32,32)
    )
    plane = tissue.AxisPlane(0, 0.0)
    return tissue.DomainBuild(
        G, omega,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=plane
    )

# Full gamma grid and chunking
gamma_values = np.logspace(-10, 2, 20)
chunks       = np.array_split(gamma_values, NUM_TASKS)
my_gammas    = chunks[TASK_ID]

# Initialize solver
domain = build_graph_and_domain()
solver = fem.SubCubes(
    domain=domain,
    lower_cube_bounds=[[0,0,0],[0.010,0.010,0.010]],
    upper_cube_bounds=[[0.033,0.030,0.010],[0.043,0.040,0.020]],
    order=2
)
solver.solve(
    gamma=X_DEFAULT[0], gamma_a=X_DEFAULT[1], gamma_R=X_DEFAULT[2],
    mu=mu, k_t=k_t, P_in=P_in, P_cvp=P_cvp
)

# Helper to compute flow
def compute_flow(params):
    solver.solve(*params, mu=mu, k_t=k_t, P_in=P_in, P_cvp=P_cvp)
    try:
        dolfin.cpp.la.clear_petsc()
    except AttributeError:
        pass
    return solver.compute_net_flow_all_dolfin()

# Sweep + optimization
results = []
for gamma in my_gammas:
    # optimize gamma_a & gamma_R in log-space
    y0 = np.log([X_DEFAULT[1], X_DEFAULT[2]])
    def obj(y):
        ga, gR = np.exp(y)
        flow = compute_flow([gamma, ga, gR])
        return (flow - TARGET_FLOW)**2 + LAMBDA_REG*np.sum(y**2)
    res = scipy.optimize.minimize(
        obj, y0,
        method='Nelder-Mead',
        options={'maxiter':20}
    )
    ga_opt, gR_opt = np.exp(res.x)

    # final solves and metrics
    flow      = compute_flow([gamma, ga_opt, gR_opt])
    lower_out = solver.compute_lower_cube_flux_out()
    upper_in  = solver.compute_upper_cube_flux_in()
    upper_out = solver.compute_upper_cube_flux_out()
    upper_net = solver.compute_upper_cube_flux()

    results.append({
        'gamma':     gamma,
        'gamma_a':   ga_opt,
        'gamma_R':   gR_opt,
        'net_flow':  flow,
        'lower_out': lower_out,
        'upper_in':  upper_in,
        'upper_out': upper_out,
        'upper_net': upper_net
    })

# Write local CSV
os.makedirs(EXPORT_PATH, exist_ok=True)
local_csv = os.path.join(EXPORT_PATH, f"gamma_array_{TASK_ID}.csv")
pd.DataFrame(results).set_index('gamma').to_csv(local_csv)
print(f"[TASK {TASK_ID}] wrote {local_csv}")

# TASK 0: wait for all tasks to finish, then merge + plot
if TASK_ID == 0:
    pattern = os.path.join(EXPORT_PATH, "gamma_array_*.csv")
    while len(glob.glob(pattern)) < NUM_TASKS:
        time.sleep(5)

    df = pd.concat([
        pd.read_csv(fname, index_col='gamma')
        for fname in sorted(glob.glob(pattern))
    ]).sort_index()

    # Combined CSV
    tz = pytz.timezone("America/Chicago")
    ts = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")
    combined = os.path.join(EXPORT_PATH, f"gamma_combined_{ts}.csv")
    df.to_csv(combined)
    print(f"Wrote combined CSV: {combined}")

    # Plot helpers
def plot(df, name, ylabel, cols):
    plt.figure(figsize=(8,6))
    for c, marker in zip(cols, ['o','s','^','d']):
        plt.semilogx(df.index, df[c], marker=marker, label=c)
    plt.xlabel("gamma (log scale)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(which='both', ls='--')
    plt.tight_layout()
    out = os.path.join(EXPORT_PATH, f"{name}_{ts}.png")
    plt.savefig(out)
    plt.close()

# Generate plots
plot(df, 'lower_out', 'Lower Cube Flux Out', ['lower_out'])
plot(df, 'upper_flux', 'Flux', ['upper_in','upper_out','upper_net'])

