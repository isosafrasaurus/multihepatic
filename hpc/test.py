#!/usr/bin/env python3
import os
import sys
import tempfile
import numpy as np




cache_dir = tempfile.mkdtemp(prefix=f"dijitso_cache_{os.getpid()}_")
os.environ['DIJITSO_CACHE_DIR'] = cache_dir
os.environ['FFC_CACHE_DIR']    = cache_dir




root = "../"
sys.path.append(os.path.join(root, "src"))

import dolfin
from graphnics import FenicsGraph
import fem
import tissue

def main():
    TEST_NUM_NODES_EXP = 5
    G = FenicsGraph()
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
        (0, 1, 0.004),
        (1, 2, 0.003),
        (1, 3, 0.003),
        (2, 4, 0.002),
        (2, 6, 0.003),
        (3, 5, 0.002),
        (3, 7, 0.003),
    ]
    for nid, pos in nodes.items():
        G.add_node(nid, pos=pos)
    for u, v, r in edges:
        G.add_edge(u, v, radius=r)
    G.make_mesh(n=TEST_NUM_NODES_EXP)
    G.make_submeshes()

    OMEGA_BOUNDS = [[0.0, 0.0, 0.0], [0.05, 0.04, 0.03]]
    omega = tissue.OmegaBuild(G, bounds=OMEGA_BOUNDS)
    x0plane = tissue.AxisPlane(0, 0.0)
    domain = tissue.DomainBuild(
        G,
        omega,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=x0plane,
    )

    solver = fem.SubCubes(
        domain=domain,
        lower_cube_bounds=[[0.0, 0.0, 0.0], [0.010, 0.010, 0.010]],
        upper_cube_bounds=[[0.033, 0.030, 0.010], [0.043, 0.040, 0.020]],
        order=2,
    )

    
    X_DEFAULT = [4.855e-05, 3.568e-08, 1.952e-07]
    
    mu   = 1.0e-3
    k_t  = 1.0e-10
    P_in = 100.0 * 133.322
    P_cvp=   1.0 * 133.322

    solver.solve(
        gamma    = X_DEFAULT[0],
        gamma_a  = X_DEFAULT[1],
        gamma_R  = X_DEFAULT[2],
        mu        = mu,
        k_t       = k_t,
        P_in      = P_in,
        P_cvp     = P_cvp,
    )

    flow = solver.compute_net_flow_all_dolfin()
    print(f"Default net flow = {flow:.6e} m³/s")

if __name__ == "__main__":
    main()
