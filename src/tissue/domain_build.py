import numpy as np
from dolfin import SubDomain, MeshFunction, Measure, UnitCubeMesh, facets, near, UserExpression, DOLFIN_EPS
from .geometry import BoundaryPoint

class DomainBuild:
    def __init__(self, G, Omega, Lambda_num_nodes_exp = 5, Lambda_inlet_nodes = None, Omega_sink_subdomain = None):
        self.G = G
        G.make_mesh(n = Lambda_num_nodes_exp)
        G.make_submeshes()
        
        self.Lambda, self.Omega = G.mesh, Omega
        self.boundary_Omega = MeshFunction("size_t", self.Omega, Omega.topology().dim() - 1, 0)
        self.boundary_Lambda = MeshFunction("size_t", self.Lambda, G.mesh.topology().dim() - 1, 0)

        if Omega_sink_subdomain is not None:
            Omega_sink_subdomain.mark(self.boundary_Omega, 1)
        self.dsOmega = Measure("ds", domain = self.Omega, subdomain_data = self.boundary_Omega)

        if Lambda_inlet_nodes is not None:
            lambda_coords = self.Lambda.coordinates()
            for node_id in Lambda_inlet_nodes:
                coordinate = lambda_coords[node_id]
                inlet_subdomain = BoundaryPoint(coordinate)
                inlet_subdomain.mark(self.boundary_Lambda, 1)
        
        self.dsLambda = Measure("ds", domain = self.Lambda, subdomain_data = self.boundary_Lambda)
        self.dxOmega = Measure("dx", domain = self.Omega)
        self.dxLambda = Measure("dx", domain = self.Lambda)
        self.dsOmegaNeumann = self.dsOmega(0)
        self.dsOmegaSink = self.dsOmega(1)
        self.dsLambdaRobin = self.dsLambda(0)
        self.dsLambdaInlet = self.dsLambda(1)

    def get_cells_along_path(self, path, tolerance = DOLFIN_EPS):
        global_vertices = []
        global_coords = Lambda.coordinates()
    
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.G.has_edge(u, v):
                edge, forward = (u, v), True
            elif self.G.has_edge(v, u):
                edge, forward = (v, u), False
            else:
                raise ValueError(f"No edge between {u} and {v} in the graph.")
            if not "submesh" in self.G.edges[edge]:
                raise ValueError("FenicsGraph object submeshes not initialized. Call .make_submeshes()")
            submesh = self.G.edges[edge]["submesh"]
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
            tangent = self.G.edges[edge]["tangent"]
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
    def __init__(self, domain, **kwargs):
        self.G = domain.G
        self.tree = domain.Lambda.bounding_box_tree()
        self.tree.build(domain.Lambda)
        super().__init__(**kwargs)

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
    def __init__(self, domain, **kwargs):
        self.G = domain.G
        self.tree = domain.Lambda.bounding_box_tree()
        self.tree.build(domain.Lambda)
        super().__init__(**kwargs)

    def eval(self, value, x):
        p = Point(*x)
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
            return
        edge_ix = self.G.mf[cell]
        edge = list(self.G.edges())[edge_ix]
        edge_data = self.G.edges[edge]
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