
import os, json
from graphnics import FenicsGraph

def get_fg_from_json(directory: str) -> FenicsGraph:
    json_files = sorted([f for f in os.listdir(directory)
                         if f.startswith("Centerline_") and f.endswith(".mrk.json")])
    if not json_files:
        raise ValueError(f"No .json files found in {directory}")

    G = FenicsGraph()
    branch_points, ind = {}, 0
    for idx, fn in enumerate(json_files):
        data = json.load(open(os.path.join(directory, fn)))
        pts = data['markups'][0]['controlPoints']
        rad = data['markups'][0]['measurements'][3]['controlPointValues']

        G.add_nodes_from(range(ind - idx, ind + len(pts) - idx))
        v1 = next((k for k, v in branch_points.items() if pts[0]['position'] == v), 0)
        v2 = ind - idx + 1
        G.nodes[v1]["pos"], G.nodes[v2]["pos"] = pts[0]['position'], pts[1]['position']
        G.nodes[v1]["radius"], G.nodes[v2]["radius"] = rad[0], rad[1]
        G.add_edge(v1, v2); G.edges[v1, v2]["radius"] = (G.nodes[v1]["radius"] + G.nodes[v2]["radius"]) / 2

        for i in range(len(pts) - 2):
            a, b = ind - idx + 1 + i, ind - idx + 2 + i
            G.nodes[a]["pos"], G.nodes[b]["pos"] = pts[i+1]['position'], pts[i+2]['position']
            G.nodes[a]["radius"], G.nodes[b]["radius"] = rad[i+1], rad[i+2]
            G.add_edge(a, b); G.edges[a, b]["radius"] = (G.nodes[a]["radius"] + G.nodes[b]["radius"]) / 2

        ind += len(pts)
        branch_points[ind - idx - 1] = pts[-1]['position']
    return G

