import os
import json
from graphnics import FenicsGraph

def build_fenicsgraph_from_json(directory):
    G = FenicsGraph()
    branch_points = {}

    json_files = sorted([
        f for f in os.listdir(directory)
        if f.startswith("Centerline_") and f.endswith(".mrk.json")
    ])
    num_files = len(json_files)
    print(f"Found {num_files} JSON files in '{directory}'.")

    ind = 0
    for idx, file_name in enumerate(json_files):
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)

        points = data['markups'][0]['controlPoints']
        radius = data['markups'][0]['measurements'][3]['controlPointValues']

        G.add_nodes_from(range(ind - idx, ind + len(points) - idx))

        
        v1 = 0
        for key, val in branch_points.items():
            if points[0]['position'] == val:
                v1 = key
                break

        
        v2 = ind - idx + 1
        pos_v1 = points[0]['position']
        pos_v2 = points[1]['position']
        G.nodes[v1]["pos"] = pos_v1
        G.nodes[v2]["pos"] = pos_v2
        G.nodes[v1]["radius"] = radius[0]
        G.nodes[v2]["radius"] = radius[1]
        G.add_edge(v1, v2)
        
        G.edges[v1, v2]["radius"] = (G.nodes[v1]["radius"] + G.nodes[v2]["radius"]) / 2

        
        for i in range(len(points) - 2):
            v1 = ind - idx + 1 + i
            v2 = v1 + 1
            pos_v1 = points[i + 1]['position']
            pos_v2 = points[i + 2]['position']
            G.nodes[v1]["pos"] = pos_v1
            G.nodes[v2]["pos"] = pos_v2
            G.nodes[v1]["radius"] = radius[i + 1]
            G.nodes[v2]["radius"] = radius[i + 2]
            G.add_edge(v1, v2)
            G.edges[v1, v2]["radius"] = (G.nodes[v1]["radius"] + G.nodes[v2]["radius"]) / 2

        ind += len(points)
        branch_points.update({ind - idx - 1: pos_v2})

    G.make_mesh()
    return G