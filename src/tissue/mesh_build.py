# tissue/mesh_build.py

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
        self.fenics_graph = fenics_graph
        fenics_graph.make_mesh(n = Lambda_num_nodes_exp)
        fenics_graph.make_submeshes()
        self.Lambda, edge_marker = fenics_graph.get_mesh(n = Lambda_num_nodes_exp)
        Lambda_coords = self.Lambda.coordinates()
        Lambda_min = np.min(Lambda_coords, axis=0)
        Lambda_max = np.max(Lambda_coords, axis=0)

        self.Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = self.Omega.coordinates()
        if Omega_bounds is None:
            scales = Lambda_max - Lambda_min + 2 * Lambda_padding
            shifts = Lambda_min - Lambda_padding
            self.Omega_bounds = np.array([shifts, shifts + scales])
        else:
            lower = np.minimum(Omega_bounds[0], Omega_bounds[1])
            upper = np.maximum(Omega_bounds[0], Omega_bounds[1])
            if not (np.all(Lambda_min >= lower) and np.all(Lambda_max <= upper)):
                raise ValueError("Lambda mesh is not contained within the provided Omega_bounds.")
            scales = upper - lower
            shifts = lower
            self.Omega_bounds = np.vstack((lower, upper))
        Omega_coords[:] = Omega_coords * scales + shifts
        self.radius_map = RadiusMap(fenics_graph, edge_marker)

    def get_cells_along_path(self, path):
        global_vertices = []
        global_coords = self.fenics_graph.mesh.coordinates()
    
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.fenics_graph.has_edge(u, v):
                edge, forward = (u, v), True
            elif self.fenics_graph.has_edge(v, u):
                edge, forward = (v, u), False
            else:
                raise ValueError(f"No edge between {u} and {v} in the graph.")
    
            submesh = self.fenics_graph.edges[edge]["submesh"]
            coords = submesh.coordinates()
            if hasattr(submesh, 'entity_map'):
                local_to_global = submesh.entity_map(0)
            else:
                tol = 1e-12
                local_to_global = []
                for local_pt in coords:
                    matches = np.where(np.all(np.isclose(global_coords, local_pt, atol=tol), axis=1))[0]
                    if len(matches) == 0:
                        raise ValueError(f"No matching global vertex for local coordinate: {local_pt}")
                    local_to_global.append(matches[0])
                local_to_global = np.array(local_to_global)
    
            tangent = self.fenics_graph.edges[edge]["tangent"]
            if not forward:
                tangent = -tangent
            proj = np.dot(coords, tangent)
            sorted_local_indices = np.argsort(proj)
            ordered_globals = [local_to_global[idx] for idx in sorted_local_indices]
    
            if i > 0 and ordered_globals[0] == global_vertices[-1]:
                ordered_globals = ordered_globals[1:]
            global_vertices.extend(ordered_globals)
        return global_vertices
    
    def get_surface_area(self) -> float:
        sizes = [self.Omega_bounds[1][i] - self.Omega_bounds[0][i] for i in range(3)]
        print(sizes)
        return 2 * (sizes[0] * sizes[1] + sizes[0] * sizes[2] + sizes[1] * sizes[2])