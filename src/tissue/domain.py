import numpy as np
from dolfin import SubDomain, MeshFunction, Measure, UnitCubeMesh, facets, near, UserExpression, DOLFIN_EPS, Point
from .geometry import BoundaryPoint

def get_cells_along_path(G, path, tolerance = DOLFIN_EPS):
    global_vertices = []
    global_coords = G.mesh.coordinates()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G.has_edge(u, v):
            edge, forward = (u, v), True
        elif G.has_edge(v, u):
            edge, forward = (v, u), False
        else:
            raise ValueError(f"No edge between {u} and {v} in the graph.")
        if not "submesh" in G.edges[edge]:
            raise ValueError("FenicsGraph object submeshes not initialized. Call .make_submeshes()")
        submesh = G.edges[edge]["submesh"]
        coords = submesh.coordinates()
        if hasattr(submesh, 'entity_map'):
            local_to_global = submesh.entity_map(0)
        else:
            local_to_global = []
            for local_pt in coords:
                matches = np.where(np.all(np.isclose(global_coords, local_pt, atol = tolerance), axis=1))[0]
                if len(matches) == 0:
                    raise ValueError(f"No matching global vertex for local coordinate: {local_pt}")
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

class AveragingRadius(UserExpression):
    def __init__(self, tree, G, **kwargs):
        super().__init__(**kwargs)
        self.tree = tree
        self.G = G
    def eval(self, value, x):
        p = Point(x[0], x[1], x[2])
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
        else:
            edge_ix = self.G.mf[cell]
            edge = list(self.G.edges())[edge_ix]
            value[0] = self.G.edges()[edge]['radius']

class SegmentLength(UserExpression):
    def __init__(self, tree, G, **kwargs):
        super().__init__(**kwargs)
        self.tree = tree
        self.G = G
    def eval(self, value, x):
        p = Point(*x)
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
            return
        edge_ix = self.G.mf[cell]
        edge = list(self.G.edges())[edge_ix]
        u, v = edge
        pos_u = np.array(self.G.nodes[u]['pos'])
        pos_v = np.array(self.G.nodes[v]['pos'])
        length = np.linalg.norm(pos_v - pos_u)
        value[0] = float(length)

def get_Omega_rect(G, bounds = None, voxel_dim = (16, 16, 16), padding = 0.008):
    positions = [data['pos'] for node, data in G.nodes(data = True)]
    pos_array = np.array(positions)
    Lambda_min = np.min(pos_array, axis = 0)
    Lambda_max = np.max(pos_array, axis = 0)
    if bounds is None:
        scales = Lambda_max - Lambda_min + 2 * padding
        shifts = Lambda_min - padding
        bounds = [shifts, shifts + scales]
    else:
        lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
        if not (np.all(Lambda_min >= lower) and np.all(Lambda_max <= upper)):
            raise ValueError("Lambda is not contained within the provided bounds.")
        scales = upper - lower
        shifts = lower
    Omega = UnitCubeMesh(*voxel_dim)
    Omega_coords = Omega.coordinates()
    Omega_coords[:] = Omega_coords * scales + shifts
    return Omega, bounds

def get_Omega_rect_from_res(G, bounds=None, voxel_res=0.001, padding=0.008):
    positions    = [data['pos'] for node, data in G.nodes(data=True)]
    pos_array    = np.array(positions)
    Lambda_min   = np.min(pos_array, axis=0)
    Lambda_max   = np.max(pos_array, axis=0)

    if bounds is None:
        scales = Lambda_max - Lambda_min + 2 * padding
        shifts = Lambda_min - padding
        bounds = [shifts, shifts + scales]
    else:
        lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
        if not (np.all(Lambda_min >= lower) and np.all(Lambda_max <= upper)):
            raise ValueError("Lambda is not contained within the provided bounds.")
        scales = upper - lower
        shifts = lower

    voxel_dim = tuple(
        max(1, int(np.ceil(scales[i] / voxel_res)))
        for i in range(3)
    )

    Omega = UnitCubeMesh(*voxel_dim)
    Omega_coords = Omega.coordinates()
    Omega_coords[:] = Omega_coords * scales + shifts

    return Omega, bounds
