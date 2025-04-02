import numpy as np
from dolfin import UnitCubeMesh

class OmegaBuild:
    def __init__(
        self, 
        fenics_graph,
        bounds = None,
        voxel_dim = (16, 16, 16),
        padding = 0.008
    ):
        if not hasattr(fenics_graph, "mesh"):
            raise ValueError("FenicsGraph object mesh not initialized. Call .make_mesh()")
        
        Lambda_coords = fenics_graph.mesh.coordinates()
        Lambda_min, Lambda_max = np.min(Lambda_coords, axis=0), np.max(Lambda_coords, axis=0)
        self.Omega = UnitCubeMesh(*voxel_dim)
        Omega_coords = self.Omega.coordinates()
        
        if bounds is None:
            scales = Lambda_max - Lambda_min + 2 * padding
            shifts = Lambda_min - padding
            self.bounds = [shifts, shifts + scales]
        else:
            lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
            if not (np.all(Lambda_min >= lower) and np.all(Lambda_max <= upper)):
                raise ValueError("Lambda mesh is not contained within the provided bounds.")
            scales = upper - lower
            shifts = lower
            self.bounds = bounds
        Omega_coords[:] = Omega_coords * scales + shifts
        sizes = np.diff(self.bounds, axis=0)[0]
        self.surface_area = 2 * (sizes[0]*sizes[1] + sizes[0]*sizes[2] + sizes[1]*sizes[2])

    def __str__(self):
        return f"OmegaBuild object: bounds = {self.bounds}, surface_area = {self.surface_area}"