from typing import Optional, List
from dolfin import UnitCubeMesh
from graphnics import FenicsGraph
from xii import *
import networkx as nx
import numpy as np

class MeshCreator:
    def __init__(
        self,
        G: FenicsGraph,
        Omega_bounds_dim: Optional[List[List[float]]] = None,
        Omega_mesh_voxel_dim: List[int] = [16, 16, 16],
        Lambda_padding_min: float = 0.008,
        Lambda_num_nodes_exp: int = 8,
    ):
        
        
        G.make_mesh(n=Lambda_num_nodes_exp)
        Lambda, Lambda_edge_marker = G.get_mesh()

        
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

        Omega_min = np.min(Omega_coords, axis=0)
        Omega_max = np.max(Omega_coords, axis=0)
        print("Omega bounding coordinates:")
        print("Min:", Omega_min)
        print("Max:", Omega_max)

        
        self.Lambda = Lambda
        self.Omega = Omega
        self.Lambda_edge_marker = Lambda_edge_marker