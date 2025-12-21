import os
from datetime import datetime

from dolfin import MPI, File  

from src import (
    Domain1D,
    Domain3D,
    Parameters,
    Simulation,
    release_solution,
    PressureVelocityProblem,
)
from tissue.meshing import sink_markers_from_surface_vtk

import nibabel as nib
import numpy as np

CONTAINER_DATA_ROOT = "_data"

def utility():
    

    in_path = os.path.join(CONTAINER_DATA_ROOT, "newData.nii.gz")

    
    out_liver = os.path.join(CONTAINER_DATA_ROOT, "newData_liver_label1_mask.nii.gz")
    out_vessel = os.path.join(CONTAINER_DATA_ROOT, "newData_vessel_label2_mask.nii.gz")
    out_sink = os.path.join(CONTAINER_DATA_ROOT, "newData_sink_label3_mask.nii.gz")

    
    img = nib.load(in_path)
    data = img.get_fdata(dtype=np.float32)

    
    labels = np.rint(data).astype(np.int32)

    
    liver_mask = (labels == 1).astype(np.uint8)
    vessel_mask = (labels == 2).astype(np.uint8)
    sink_mask = (labels == 3).astype(np.uint8)

    
    liver_img = nib.Nifti1Image(liver_mask, affine=img.affine, header=img.header)
    vessel_img = nib.Nifti1Image(vessel_mask, affine=img.affine, header=img.header)
    sink_img = nib.Nifti1Image(sink_mask, affine=img.affine, header=img.header)

    
    liver_img.set_data_dtype(np.uint8)
    vessel_img.set_data_dtype(np.uint8)
    sink_img.set_data_dtype(np.uint8)

    nib.save(liver_img, out_liver)
    nib.save(vessel_img, out_vessel)
    nib.save(sink_img, out_sink)

    print("Saved:")
    print(f"  Label 1 (liver):  {out_liver}  voxels={int(liver_mask.sum())}")
    print(f"  Label 2 (vessel): {out_vessel} voxels={int(vessel_mask.sum())}")
    print(f"  Label 3 (sink):   {out_sink}   voxels={int(sink_mask.sum())}")

def main():
    vtk_1d = os.path.join(CONTAINER_DATA_ROOT, "sortedVesselNetwork.vtk")
    vtk_3d = os.path.join(CONTAINER_DATA_ROOT, "nii2mesh_liver_mask.vtk")
    vtk_sink = os.path.join(CONTAINER_DATA_ROOT, "nii2mesh_liver_sink.vtk")

    out_root = "_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(out_root, timestamp)

    
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

            
            p3d_path = os.path.join(outdir, "pressure_3d.pvd")
            p1d_path = os.path.join(outdir, "pressure_1d.pvd")
            v3d_path = os.path.join(outdir, "velocity_3d.pvd")

            
            p3d_file = File(p3d_path)
            p3d_file << sol.p3d

            
            p1d_file = File(p1d_path)
            p1d_file << sol.p1d

            
            if getattr(sol, "v3d", None) is not None:
                v3d_file = File(v3d_path)
                v3d_file << sol.v3d

            
            release_solution(sol)


if __name__ == "__main__":
    main()
