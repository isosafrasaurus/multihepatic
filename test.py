#!/usr/bin/env python3
import os
import sys
import tempfile
import numpy as np
from mpi4py import MPI
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pytz


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


dir_here = os.path.abspath(os.path.dirname(__file__))
jobid    = os.environ.get("SLURM_JOB_ID", "nojob")
user     = os.environ.get("USER", "")
cache_dir = tempfile.mkdtemp(prefix=f"dijitso_cache_{user}_{jobid}_{rank}_")
os.environ['DIJITSO_CACHE_DIR'] = cache_dir
os.environ['FFC_CACHE_DIR']    = cache_dir


WORK_PATH   = dir_here
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join(WORK_PATH, "export")
sys.path.append(SOURCE_PATH)

import dolfin
from graphnics import FenicsGraph
import fem
import tissue


TARGET_FLOW = 5.0e-6
LAMBDA_REG  = 1e-3
X_DEFAULT   = [4.855e-05, 3.568e-08, 1.952e-07]
mu, k_t     = 1.0e-3, 1.0e-10
P_in        = 100.0 * 133.322
P_cvp       = 1.0   * 133.322


gamma_values = np.logspace(-10, 2, 20)
my_gammas     = gamma_values[rank::size]

def plot_flow_data_semilog(df: pd.DataFrame, directory: str):
    plot_dir = os.path.join(directory, "plot_flow_data_semilog")
    os.makedirs(plot_dir, exist_ok=True)
    tz = pytz.timezone("America/Chicago")
    timestamp = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")

    
    plt.figure(figsize=(8, 6))
    plt.semilogx(df.index.values, df['lower_out'], marker='o')
    plt.xlabel("gamma (log scale)")
    plt.ylabel("Lower Cube Flux Out")
    plt.grid(which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"lower_out_{timestamp}.png"))
    plt.close()

    
    plt.figure(figsize=(8, 6))
    plt.semilogx(df.index.values, df['upper_in'],  marker='s', label='Upper In')
    plt.semilogx(df.index.values, df['upper_out'], marker='^', label='Upper Out')
    plt.semilogx(df.index.values, df['upper_net'], marker='d', label='Upper Net')
    plt.xlabel("gamma (log scale)")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"upper_flux_{timestamp}.png"))
    plt.close()


def build_graph_and_domain(exp):
    nodes = {
        0: [0.000,0.020,0.015], 1: [0.010,0.020,0.015],
        2: [0.022,0.013,0.015], 3: [0.022,0.028,0.015],
        4: [0.015,0.005,0.015], 5: [0.015,0.035,0.015],
        6: [0.038,0.005,0.015], 7: [0.038,0.035,0.015]
    }
    edges = [
        (0,1,0.004),(1,2,0.003),(1,3,0.003),
        (2,4,0.002),(2,6,0.003),(3,5,0.002),(3,7,0.003)
    ]
    G = FenicsGraph()
    for nid,pos in nodes.items(): G.add_node(nid, pos=pos)
    for u,v,r in edges: G.add_edge(u,v,radius=r)
    G.make_mesh(n=exp)
    G.make_submeshes()
    omega = tissue.OmegaBuild(G, bounds=[[0,0,0],[0.05,0.04,0.03]], voxel_dim=(16, 16, 16))
    inlet_plane = tissue.AxisPlane(0,0.0)
    return tissue.DomainBuild(
        G, omega,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=inlet_plane
    )

if __name__ == "__main__":
    
    domain = build_graph_and_domain(exp=5)

    
    solver = fem.SubCubes(
        domain=domain,
        lower_cube_bounds=[[0,0,0],[0.010,0.010,0.010]],
        upper_cube_bounds=[[0.033,0.030,0.010],[0.043,0.040,0.020]],
        order=2
    )
    solver.solve(
        gamma   = X_DEFAULT[0], gamma_a = X_DEFAULT[1], gamma_R = X_DEFAULT[2],
        mu=mu, k_t=k_t, P_in=P_in, P_cvp=P_cvp
    )

    
    local_results = []
    for gamma in my_gammas:
        
        y0 = np.log([X_DEFAULT[1], X_DEFAULT[2]])
        def obj(y):
            ga, gR = np.exp(y)
            params = [gamma, ga, gR]
            solver.solve(*params, mu=mu, k_t=k_t, P_in=P_in, P_cvp=P_cvp)
            try: dolfin.cpp.la.clear_petsc()
            except AttributeError: pass
            flow = solver.compute_net_flow_all_dolfin()
            return (flow - TARGET_FLOW)**2 + LAMBDA_REG*np.sum(y**2)

        res = scipy.optimize.minimize(
            obj, y0, method='Nelder-Mead', options={'maxiter':20}
        )
        ga_opt, gR_opt = np.exp(res.x)

        
        solver.solve(
            gamma=gamma, gamma_a=ga_opt, gamma_R=gR_opt,
            mu=mu, k_t=k_t, P_in=P_in, P_cvp=P_cvp
        )
        try: dolfin.cpp.la.clear_petsc()
        except AttributeError: pass

        flow   = solver.compute_net_flow_all_dolfin()
        lower  = solver.compute_lower_cube_flux_out()
        upper_in  = solver.compute_upper_cube_flux_in()
        upper_out = solver.compute_upper_cube_flux_out()
        upper_net = solver.compute_upper_cube_flux()

        print(f"[rank {rank}] gamma={gamma:.3e} → γₐ={ga_opt:.3e}, γ_R={gR_opt:.3e}, flow={flow:.6e}")
        local_results.append({
            'gamma':     gamma,
            'gamma_a':   ga_opt,
            'gamma_R':   gR_opt,
            'net_flow':  flow,
            'lower_out': lower,
            'upper_in':  upper_in,
            'upper_out': upper_out,
            'upper_net': upper_net
        })

    
    all_res = comm.gather(local_results, root=0)
    if rank == 0:
        flat = [d for sub in all_res for d in sub]
        df   = pd.DataFrame(flat).set_index('gamma')

        
        os.makedirs(EXPORT_PATH, exist_ok=True)
        tz = pytz.timezone("America/Chicago")
        ts = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M")
        csv_path = os.path.join(EXPORT_PATH, f"gamma_mpi_{ts}.csv")
        df.to_csv(csv_path)
        print(f"Wrote results to {csv_path}", flush=True)

        
        plot_flow_data_semilog(df, EXPORT_PATH)

