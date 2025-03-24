

import numpy as np
from dolfin import UnitCubeMesh
from .expressions import RadiusMap
from .geometry import AxisPlane

class MeshBuild:
    def __init__(
        self, fenics_graph,
        Omega_bounds = None,
        Omega_mesh_voxel_dim = (16, 16, 16),
        Lambda_padding = 0.008,
        Lambda_num_nodes_exp = 5
    ):
        fenics_graph.make_mesh(n = Lambda_num_nodes_exp)
        fenics_graph.make_submeshes()
        self.Lambda, edge_marker = fenics_graph.get_mesh(n = Lambda_num_nodes_exp)
        
        Lambda_coords = self.Lambda.coordinates()
        lambda_min = np.min(Lambda_coords, axis=0)
        lambda_max = np.max(Lambda_coords, axis=0)

        self.Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = self.Omega.coordinates()

        if Omega_bounds is None:
            scales = lambda_max - lambda_min + 2 * Lambda_padding
            shifts = lambda_min - Lambda_padding
            self.Omega_bounds = np.array([shifts, shifts + scales])
        else:
            lower = np.minimum(Omega_bounds[0], Omega_bounds[1])
            upper = np.maximum(Omega_bounds[0], Omega_bounds[1])
            if not (np.all(lambda_min >= lower) and np.all(lambda_max <= upper)):
                raise ValueError("Lambda mesh is not contained within the provided Omega_bounds.")
            scales = upper - lower
            shifts = lower
            self.Omega_bounds = np.vstack((lower, upper))

        Omega_coords[:] = Omega_coords * scales + shifts
        self.radius_map = RadiusMap(fenics_graph, edge_marker)

    def get_Omega_axis_plane(self, face, tolerance=1e-10) -> AxisPlane:
        face = face.lower()
        match face:
            case "left":
                return AxisPlane(0, self.Omega_bounds[0][0], tolerance)
            case "right":
                return AxisPlane(0, self.Omega_bounds[1][0], tolerance)
            case "bottom":
                return AxisPlane(1, self.Omega_bounds[0][1], tolerance)
            case "top":
                return AxisPlane(1, self.Omega_bounds[1][1], tolerance)
            case "front":
                return AxisPlane(2, self.Omega_bounds[0][2], tolerance)
            case "back":
                return AxisPlane(2, self.Omega_bounds[1][2], tolerance)
            case _:
                raise ValueError(f"Unknown face: {face}")