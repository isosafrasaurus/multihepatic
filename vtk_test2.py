import os
from datetime import datetime

import numpy as np
from dolfin import MPI, File
from graphnics import TubeFile

from src import (
    Domain1D,
    Domain3D,
    Parameters,
    Simulation,
    release_solution,
    PressureVelocityProblem,
)


def main():
    CONTAINER_DATA_ROOT = "_data"

    
    vtk_1d = os.path.join(CONTAINER_DATA_ROOT, "cropped", "dataNew.vtk")

    out_root = "_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(out_root, timestamp)

    comm = MPI.comm_world

    
    if MPI.rank(comm) == 0:
        os.makedirs(outdir, exist_ok=True)
    comm.barrier()

    
    with Domain1D.from_vtk(
            vtk_1d,
            radius_field="Radius",
            inlet_nodes=[89]
    ) as Lambda:

        G = Lambda.G
        positions = [data["pos"] for _, data in G.nodes(data=True)]
        pos = np.asarray(positions, dtype=float)

        lam_min = pos.min(axis=0)
        lam_max = pos.max(axis=0)

        center = 0.5 * (lam_min + lam_max)
        extent = lam_max - lam_min
        L = float(extent.max())

        padding = 8e-3
        L_padded = L + 2.0 * padding

        lower = center - 0.5 * L_padded
        upper = center + 0.5 * L_padded
        bounds = (lower, upper)

        
        voxel_dim = (16, 16, 16)

        with Domain3D.from_graph(
                G,
                bounds=bounds,
                voxel_dim=voxel_dim,
                padding=0.0,
                enforce_graph_in_bounds=True,
        ) as Omega:
            with Simulation(
                    Lambda=Lambda,
                    Omega=Omega,
                    problem_cls=PressureVelocityProblem,
                    Omega_sink_subdomain=None,
                    linear_solver="mumps",
            ) as sim:
                params = Parameters(
                    gamma=1.0,
                    gamma_a=1.0,
                    gamma_R=1.0,
                    mu=3.5e-3,
                    k_t=1.0e-12,
                    P_in=20e3,
                    P_cvp=5e3,
                )

                sol = sim.run(params)

                
                p3d_path = os.path.join(outdir, "pressure_3d.pvd")
                p1d_path = os.path.join(outdir, "pressure_1d.pvd")
                v3d_path = os.path.join(outdir, "velocity_3d.pvd")

                
                p3d_file = File(p3d_path)
                p3d_file << sol.p3d

                
                p1d_file = TubeFile(G, p1d_path)
                p1d_file << sol.p1d

                
                if getattr(sol, "v3d", None) is not None:
                    v3d_file = File(v3d_path)
                    v3d_file << sol.v3d

                
                release_solution(sol)


if __name__ == "__main__":
    main()
