


from fenics import *
import networkx as nx
import numpy as np
import sys

sys.path.append("../")
from graphnics import *


def color_graph(G):
    

    G_disconn = G.copy(as_view=False)
    G_disconn = nx.Graph(G_disconn)

    G_undir = nx.Graph(G)

    C = nx.adj_matrix(nx.Graph(G))
    num_vertex_conns = np.asarray(np.sum(C, axis=1)).flatten()
    bifurcation_points = np.where(num_vertex_conns > 2)[0].tolist()

    
    
    
    

    for b in bifurcation_points:
        for i, e in enumerate(G_undir.edges(b)):

            
            vertex_list = list(e)
            vertex_list.remove(b)
            v2 = vertex_list[0]

            
            new_bif_vertex = f"{b} {v2}"
            G_disconn.add_node(new_bif_vertex)
            G_disconn.nodes()[new_bif_vertex]["pos"] = G_undir.nodes()[b]["pos"]

            G_disconn.add_edge(new_bif_vertex, v2)  

            
            try:
                
                
                G_disconn.remove_edge(e[0], e[1])
            except:
                
                for v_e in e:
                    assert (
                        v_e in bifurcation_points
                    ), "Graph coloring algorithm may be malfunctioning"

    
    subG = list(G_disconn.subgraph(c) for c in nx.connected_components(G_disconn))

    
    n_branches = 0

    for sG in subG:
        C = nx.adj_matrix(sG)
        num_vertex_conns = np.asarray(np.sum(C, axis=1)).flatten()

        

        is_disconn_graph = (
            np.min(num_vertex_conns) > 0
        )  
        is_disconn_graph = is_disconn_graph and np.max(num_vertex_conns) < 3

        if is_disconn_graph:
            for e in sG.edges():
                v1 = str(e[0]).split(" ")[0]
                v2 = str(e[1]).split(" ")[0]

                
                orig_e1 = (int(v1), int(v2))
                orig_e2 = (int(v2), int(v1))
                try:
                    G.edges()[orig_e1]["color"] = n_branches
                except:
                    G.edges()[orig_e2]["color"] = n_branches

            n_branches += 1

    
    
    for e in G.edges():
        if "color" not in G.edges()[e]:
            G.edges()[e]["color"] = n_branches
            n_branches += 1


def plot_graph_color(G):
    

    pos = nx.get_node_attributes(G, "pos")

    colors = nx.get_edge_attributes(G, "color")
    nx.draw_networkx_edge_labels(G, pos, colors)

    nx.draw_networkx(G, pos)
    colors = list(nx.get_edge_attributes(G, "color").values())


def assign_radius_using_Murrays_law(G, start_node, start_radius):
    

    
    G_ = nx.bfs_tree(G, source=start_node)
    for i in range(0, len(G.nodes)):
        G_.nodes()[i]["pos"] = G.nodes()[i]["pos"]

    for e in list(G_.edges()):
        v1, v2 = e
        length = np.linalg.norm(
            np.asarray(G_.nodes()[v2]["pos"]) - np.asarray(G_.nodes()[v1]["pos"])
        )
        G_.edges()[e]["length"] = length

    assert (
        len(list(G.edges(start_node))) is 1
    ), "start node has to have a single edge sprouting from it"

    
    for i, e in enumerate(G_.edges()):

        
        if i == 0:
            G_.edges()[e]["radius"] = start_radius

        
        else:
            v1, v2 = e  
            edge_in = list(G_.in_edges(v1))[0]
            radius_p = G_.edges()[edge_in]["radius"]

            
            

            
            sub_graphs = {}
            sub_graph_lengths = {}
            for v in [v2, v1]:
                sub_graph = G_.subgraph(nx.shortest_path(G_, v))

                sub_graph_length = 0
                for d_e in sub_graph.edges():
                    sub_graph_length += sub_graph.edges()[d_e]["length"]

                sub_graphs[str(v)] = sub_graph
                sub_graph_lengths[str(v)] = sub_graph_length

            sub_graph_lengths[str(v2)] += G_.edges()[e]["length"]

            fraction = sub_graph_lengths[str(v2)] / sub_graph_lengths[str(v1)]

            if sub_graph_lengths[str(v2)] is 0:  
                fraction = 1 / len(sub_graphs[str(v1)].edges())

            
            radius_d = (fraction) ** (1 / 3) * radius_p
            G_.edges()[e]["radius"] = radius_d

    return G_


class DistFromSource(UserExpression):
    

    def __init__(self, G, source_node, **kwargs):
        

        self.G = G
        self.source = source_node
        super().__init__(**kwargs)

        
        if len(nx.get_edge_attributes(G, "length")) is 0:
            G.compute_edge_lengths()

        G_bfs = nx.bfs_tree(G, source_node)

        for n in G_bfs.nodes():
            G_bfs.nodes()[n]["pos"] = G.nodes()[n]["pos"]

        G_bfs = copy_from_nx_graph(G_bfs)
        G_bfs.compute_edge_lengths()

        
        dist = nx.shortest_path_length(G_bfs, source_node, weight="length")

        
        
        

        mesh, mf = G_bfs.get_mesh(n=0)

        V = FunctionSpace(mesh, "CG", 1)
        dist_func = Function(V)

        
        dofmap = list(dof_to_vertex_map(V))  

        
        for n in G.nodes():
            dof_ix = dofmap.index(n)
            dist_func.vector()[dof_ix] = dist[n]

        
        dist_func.set_allow_extrapolation(True)
        self.dist_func = dist_func
        

    def eval(self, values, x):
        
        values[0] = self.dist_func(x)

    def value_shape(self):
        return ()
        