from graphnics import *
from xii import *
import matplotlib.pyplot as plt
import pandas as pd

def get_cells_along_path(G, path, tolerance = DOLFIN_EPS):
    if not G.mesh:
        raise ValueError("FenicsGraph object meshes not initialized. Call .make_mesh()")
    
    
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

def get_path_pressure(G, uh1d, path, directory = None):
    node_ids = get_cells_along_path(G, path)
    mesh = uh1d.function_space().mesh()
    coords = mesh.coordinates()
    pressure = uh1d.compute_vertex_values(mesh)
    path_coords, path_pressure = coords[node_ids], pressure[node_ids]
    culum_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1))))

    if directory is not None:
        df_data = {"culum_dist": culum_dist, "path_pressure": path_pressure}
        df = pd.DataFrame(df_data)
        df.to_csv(directory, index = False)
    
    return culum_dist, path_pressure