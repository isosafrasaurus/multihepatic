#!/usr/bin/env python3
import os
import sys
import tempfile
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

WORK_PATH   = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join("..", "export")
sys.path.append(SOURCE_PATH)


jobid    = os.environ.get("SLURM_JOB_ID", "nojob")
user     = os.environ.get("USER", "")
cache_dir = tempfile.mkdtemp(prefix=f"dijitso_cache_{user}_{jobid}_{rank}_")
os.environ['DIJITSO_CACHE_DIR'] = cache_dir
os.environ['FFC_CACHE_DIR']    = cache_dir

import dolfin
from graphnics import FenicsGraph
import fem
import tissue

def build_graph_and_domain(exp):
    
    nodes = {
        0: [0.000, 0.020, 0.015],
        1: [0.010, 0.020, 0.015],
        2: [0.022, 0.013, 0.015],
        3: [0.022, 0.028, 0.015],
        4: [0.015, 0.005, 0.015],
        5: [0.015, 0.035, 0.015],
        6: [0.038, 0.005, 0.015],
        7: [0.038, 0.035, 0.015],
    }
    edges = [
        (0,1,0.004),(1,2,0.003),(1,3,0.003),
        (2,4,0.002),(2,6,0.003),(3,5,0.002),(3,7,0.003),
    ]
    G = FenicsGraph()
    for nid,pos in nodes.items():
        G.add_node(nid, pos=pos)
    for u,v,r in edges:
        G.add_edge(u, v, radius=r)
    G.make_mesh(n=exp)
    G.make_submeshes()

    omega = tissue.OmegaBuild(G, bounds=[[0,0,0],[0.05,0.04,0.03]], voxel_dim=(32,32,32))
    inlet_plane = tissue.AxisPlane(0, 0.0)
    domain = tissue.DomainBuild(
        G, omega,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=inlet_plane
    )
    return domain


gamma_values = np.logspace(-10, 2, 20)

my_gammas   = gamma_values[rank::size]


X_DEFAULT = [4.855e-05, 3.568e-08, 1.952e-07]
mu, k_t = 1.0e-3, 1.0e-10
P_in  = 100.0 * 133.322
P_cvp =   1.0 * 133.322

def main():
    
    domain = build_graph_and_domain(exp=7)

    
    solver = fem.SubCubes(
        domain=domain,
        lower_cube_bounds=[[0,0,0],[0.010,0.010,0.010]],
        upper_cube_bounds=[[0.033,0.030,0.010],[0.043,0.040,0.020]],
        order=2
    )
    
    solver.solve(
        gamma   = X_DEFAULT[0],
        gamma_a = X_DEFAULT[1],
        gamma_R = X_DEFAULT[2],
        mu      = mu, k_t=k_t,
        P_in    = P_in, P_cvp=P_cvp,
    )

    
    results = []
    for gamma in my_gammas:
        solver.solve(
            gamma   = gamma,
            gamma_a = X_DEFAULT[1],
            gamma_R = X_DEFAULT[2],
            mu      = mu, k_t=k_t,
            P_in    = P_in, P_cvp=P_cvp,
        )
        
        try: dolfin.cpp.la.clear_petsc()
        except AttributeError: pass

        flow = solver.compute_net_flow_all_dolfin()
        print(f"[rank {rank}] gamma = {gamma:.3e} → flow = {flow:.6e}")
        results.append((gamma, flow))

    
    all_results = comm.gather(results, root=0)

    if rank == 0:
        
        flat = [item for sub in all_results for item in sub]
        flat.sort(key=lambda x: x[0])
        print("\n=== All sweep results ===")
        for gamma,flow in flat:
            print(f"{gamma:.3e}  →  {flow:.6e}")
        

if __name__=="__main__":
    main()

