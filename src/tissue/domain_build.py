    import numpy as np
from dolfin import SubDomain, MeshFunction, Measure, UnitCubeMesh, facets, near, UserExpression
from .geometry import BoundaryPoint

class DomainBuild:
    def __init__(
        self, 
        fenics_graph,
        Omega_build,
        Lambda_num_nodes_exp = 5,
        Lambda_inlet_nodes = None,
        Omega_sink_subdomain = None
    ):
        if not hasattr(fenics_graph, "mesh"):
            raise ValueError("FenicsGraph object mesh not initialized. Call .make_mesh()")

        self.fenics_graph = fenics_graph
        self.Lambda = fenics_graph.mesh
        self.Omega = Omega_build.Omega
        self.boundary_Omega = MeshFunction("size_t", self.Omega, Omega_build.Omega.topology().dim() - 1, 0)
        self.boundary_Lambda = MeshFunction("size_t", self.Lambda, fenics_graph.mesh.topology().dim() - 1, 0)

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
            if not "submesh" in self.fenics_graph.edges[edge]:
                raise ValueError("FenicsGraph object submeshes not initialized. Call .make_submeshes()")
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

def get_Omega(
    fenics_graph,
    bounds = None,
    voxel_dim = (16, 16, 16),
    padding = 0.008
):
    positions = [data['pos'] for node, data in graph.nodes(data=True)]
    pos_array = np.array(positions)
    Lambda_min = np.min(pos_array, axis=0)
    Lambda_max = np.max(pos_array, axis=0)
    
    Omega = UnitCubeMesh(*voxel_dim)
    Omega_coords = Omega.coordinates()
    
    if bounds is None:
        scales = Lambda_max - Lambda_min + 2 * padding
        shifts = Lambda_min - padding
    else:
        lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
        if not (np.all(Lambda_min >= lower) and np.all(Lambda_max <= upper)):
            raise ValueError("Lambda mesh is not contained within the provided bounds.")
        scales = upper - lower
        shifts = lower
    
    Omega_coords[:] = Omega_coords * scales + shifts
    sizes = np.diff(bounds, axis = 0)[0]
    area_Omega = 2 * (sizes[0] * sizes[1] + sizes[0] * sizes[2] + sizes[1] * sizes[2])
    return Omega, bounds