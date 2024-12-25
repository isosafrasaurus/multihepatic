import numpy as np
import math
from dolfin import *
import FEMSink2
import importlib

importlib.reload(FEMSink2)


from FEMSinkVelo import FEMSinkVelo  

class FEMSinkCubeFlux(FEMSinkVelo):
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        
        
        coords = self.Omega.coordinates()
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
        
        
        eps = 0.5 * min(x_max - x_min, y_max - y_min, z_max - z_min)
        
        
        self.lower_cube_bounds = (x_min, x_min + eps, y_min, y_min + eps, z_min, z_min + eps)
        
        self.upper_cube_bounds = (x_max - eps, x_max, y_max - eps, y_max, z_max - eps, z_max)
        
        
        
        self.cube_boundaries = MeshFunction("size_t", self.Omega, self.Omega.topology().dim() - 1)
        self.cube_boundaries.set_all(0)
        
        
        class CubeSubBoundary(SubDomain):
            def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, tol=1e-6):
                super().__init__()
                self.x_min = x_min
                self.x_max = x_max
                self.y_min = y_min
                self.y_max = y_max
                self.z_min = z_min
                self.z_max = z_max
                self.tol = tol

            def inside(self, x, on_boundary):
                return on_boundary and (x[0] >= self.x_min - self.tol and x[0] <= self.x_max + self.tol and
                                         x[1] >= self.y_min - self.tol and x[1] <= self.y_max + self.tol and
                                         x[2] >= self.z_min - self.tol and x[2] <= self.z_max + self.tol)
        
        
        self.lower_cube = CubeSubBoundary(*self.lower_cube_bounds)
        self.upper_cube = CubeSubBoundary(*self.upper_cube_bounds)
        
        
        
        self.lower_cube.mark(self.cube_boundaries, 1)
        self.upper_cube.mark(self.cube_boundaries, 2)
        
        
        self.ds_cube = Measure("ds", domain=self.Omega, subdomain_data=self.cube_boundaries)
    
    def compute_lower_cube_flux(self):
        
        n = FacetNormal(self.Omega)
        flux_lower = assemble(dot(self.velocity, n) * self.ds_cube(1))
        return flux_lower

    def compute_upper_cube_flux(self):
        
        n = FacetNormal(self.Omega)
        flux_upper = assemble(dot(self.velocity, n) * self.ds_cube(2))
        return flux_upper

    def save_vtk(self, directory_path: str):
        
        import os
        os.makedirs(directory_path, exist_ok=True)
        
        super().save_vtk(directory_path)
        
        xdmf_file = XDMFFile(os.path.join(directory_path, "cube_boundaries.xdmf"))
        xdmf_file.write(self.cube_boundaries)
