# tissue/expressions.py

import numpy as np
from dolfin import UserExpression
from rtree import index as rtree_index
from .geometry import point_in_cylinder

class RadiusMap(UserExpression):
    def __init__(self, graph, edge_marker=None, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
        self.edge_marker = edge_marker
        self.spatial_idx = _build_rtree_3d()
        self.edge_data_list = []
        edge_id = 0
        for (u, v, data) in graph.edges(data=True):
            pos_u = np.array(graph.nodes[u]['pos'])
            pos_v = np.array(graph.nodes[v]['pos'])
            radius = data['radius']
            bbox_min = np.minimum(pos_u, pos_v) - radius
            bbox_max = np.maximum(pos_u, pos_v) + radius
            bbox = tuple(np.concatenate([bbox_min, bbox_max]).tolist())
            self.spatial_idx.insert(edge_id, bbox)
            self.edge_data_list.append((u, v, data))
            edge_id += 1

    def eval(self, value, x):
        point = tuple(x[:3])
        candidates = self.spatial_idx.intersection(point + point, objects=False)
        for edge_id in candidates:
            u, v, data = self.edge_data_list[edge_id]
            pos_u = self.graph.nodes[u]['pos']
            pos_v = self.graph.nodes[v]['pos']
            if point_in_cylinder(point, pos_u, pos_v, data['radius']):
                value[0] = data['radius']
                return
        value[0] = 0.0

    def value_shape(self):
        return ()

def _build_rtree_3d():
    p = rtree_index.Property()
    p.dimension = 3
    return rtree_index.Index(properties=p)
