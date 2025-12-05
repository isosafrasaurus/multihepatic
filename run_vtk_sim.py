#!/usr/bin/env python
from __future__ import annotations

import os
from datetime import datetime

from dolfin import MPI, XDMFFile


from src import (
    Domain1D,
    Domain3D,
    Parameters,
    Simulation,
    release_solution,
    PressureVelocityProblem,
)
from tissue.meshing import sink_markers_from_surface_vtk


def main() -> None:
    CONTAINER_DATA_ROOT = "_data"

    vtk_1d = os.path.join(CONTAINER_DATA_ROOT, "sortedVesselNetwork.vtk")
    vtk_3d = os.path.join(CONTAINER_DATA_ROOT, "nii2mesh_liver_mask.vtk")
    vtk_sink = os.path.join(CONTAINER_DATA_ROOT, "nii2mesh_liver_sink.vtk")

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
    ) as Lambda, Domain3D.from_vtk(vtk_3d) as Omega:

        
        sink_markers = sink_markers_from_surface_vtk(Omega.Omega, vtk_sink)

        
        with Simulation(
                Lambda=Lambda,
                Omega=Omega,
                problem_cls=PressureVelocityProblem,  
                Omega_sink_subdomain=sink_markers,
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

            
            p3d_path = os.path.join(outdir, "pressure_3d.xdmf")
            p1d_path = os.path.join(outdir, "pressure_1d.xdmf")
            v3d_path = os.path.join(outdir, "velocity_3d.xdmf")

            with XDMFFile(comm, p3d_path) as f:
                f.write(sol.p3d)
            with XDMFFile(comm, p1d_path) as f:
                f.write(sol.p1d)
            if getattr(sol, "v3d", None) is not None:
                with XDMFFile(comm, v3d_path) as f:
                    f.write(sol.v3d)

            
            release_solution(sol)


if __name__ == "__main__":
    main()
