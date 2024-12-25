from typing import Optional, List
from dolfin import UnitCubeMesh
from graphnics import *
from xii import *
import networkx as nx
import numpy as np

class MeshCreator:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [16, 16, 16],
        Lambda_padding_min: float = 8,
        Lambda_num_nodes_exp: int = 5,
    ):
        
        
        G.make_mesh(n=Lambda_num_nodes_exp)
        G.make_submeshes()
        Lambda, Lambda_edge_marker = G.get_mesh(n=Lambda_num_nodes_exp)

        
        Omega = UnitCubeMesh(*Omega_mesh_voxel_dim)
        Omega_coords = Omega.coordinates()

        
        node_positions = nx.get_node_attributes(G, "pos")
        node_coords = np.array(list(node_positions.values()))

        if Omega_bounds_dim is None:
            xmin, ymin, zmin = np.min(node_coords, axis=0)
            xmax, ymax, zmax = np.max(node_coords, axis=0)
            scales = np.array([
                xmax - xmin + 2 * Lambda_padding_min,
                ymax - ymin + 2 * Lambda_padding_min,
                zmax - zmin + 2 * Lambda_padding_min
            ])
            shifts = np.array([
                xmin - Lambda_padding_min,
                ymin - Lambda_padding_min,
                zmin - Lambda_padding_min
            ])
        else:
            lower = np.array(Omega_bounds_dim[0])
            upper = np.array(Omega_bounds_dim[1])
            scales = upper - lower
            shifts = lower

        
        Omega_coords[:,:] = Omega_coords * scales + shifts

        
        self.G = G
        self.Lambda = Lambda
        self.Omega = Omega
        self.Lambda_edge_marker = Lambda_edge_marker

    def get_cells_along_path(self, path):
        
        G = self.G
        
        global_vertices = []
        
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            
            
            if G.has_edge(u, v):
                edge = (u, v)
                forward = True
            elif G.has_edge(v, u):
                edge = (v, u)
                forward = False
            else:
                raise ValueError(f"No edge between {u} and {v} in the graph.")
                
            
            submesh = G.edges[edge]["submesh"]
            coords = submesh.coordinates()  
            
            
            if hasattr(submesh, 'entity_map'):
                local_to_global = submesh.entity_map(0)
            else:
                
                global_coords = G.mesh.coordinates()
                tol = 1e-12
                local_to_global = []
                for local_pt in coords:
                    matches = np.where(np.all(np.isclose(global_coords, local_pt, atol=tol), axis=1))[0]
                    if len(matches) == 0:
                        raise ValueError("No matching global vertex found for local coordinate: " + str(local_pt))
                    local_to_global.append(matches[0])
                local_to_global = np.array(local_to_global)
                
            
            tangent = G.edges[edge]["tangent"]
            if not forward:
                tangent = -tangent
                
            
            proj = np.dot(coords, tangent)
            sorted_local_indices = np.argsort(proj)
            
            
            ordered_globals = [local_to_global[idx] for idx in sorted_local_indices]
            
            
            if i > 0 and ordered_globals[0] == global_vertices[-1]:
                ordered_globals = ordered_globals[1:]
                
            global_vertices.extend(ordered_globals)
            
        return global_vertices
