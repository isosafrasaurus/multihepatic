from __future__ import annotations

from typing import Any, Callable

import networkx as nx
import numpy as np


def build_graph_from_polydata(
    points: np.ndarray,
    polylines: list[np.ndarray],
    *,
    point_radius: np.ndarray | None,
    cell_radius: np.ndarray | None,
    default_radius: float,
    reverse_edges: bool,
) -> nx.DiGraph:
    """
    Build a DiGraph where nodes are VTK point indices and edges follow polyline connectivity.

    Radius assignment:
      - If point_radius is provided (len == npoints): per-segment radius = mean(endpoint radii)
      - Else if cell_radius is provided (len == npolyline-cells): radius = cell_radius[cell_id]
      - Else: radius = default_radius
    """
    npts = int(points.shape[0])
    G = nx.DiGraph()
    for v in range(npts):
        # networks_fenicsx expects 'pos' and contiguous integer node IDs.
        G.add_node(int(v), pos=points[v].astype(np.float64, copy=False).tolist())

    pr = None
    if point_radius is not None:
        pr = np.asarray(point_radius, dtype=np.float64).reshape((npts,))

    cr = None
    if cell_radius is not None:
        cr = np.asarray(cell_radius, dtype=np.float64).reshape((-1,))

    for cell_id, conn in enumerate(polylines):
        if conn.size < 2:
            continue
        seq = conn[::-1] if reverse_edges else conn
        # Walk consecutive point pairs
        for a, b in zip(seq[:-1], seq[1:]):
            u = int(a)
            v = int(b)
            if u == v:
                continue
            if pr is not None:
                r = 0.5 * (float(pr[u]) + float(pr[v]))
            elif cr is not None and cell_id < cr.size:
                r = float(cr[cell_id])
            else:
                r = float(default_radius)

            if G.has_edge(u, v):
                # Keep something deterministic if duplicates exist: average them.
                old = float(G.edges[u, v].get("radius", r))
                G.edges[u, v]["radius"] = 0.5 * (old + r)
            else:
                G.add_edge(u, v, radius=float(r))
    return G


def _color_edges_like_networks_fenicsx(
    graph: nx.DiGraph,
    strategy: str | Callable[..., Any] | None,
) -> dict[tuple[int, int], int]:
    """
    Replicates networks_fenicsx.mesh.color_graph(...):
      - strategy is None  -> unique color per directed edge in graph.edges order
      - strategy not None -> greedy coloring on the line graph of graph.to_undirected()
    """
    if strategy is not None:
        undirected_edge_graph = nx.line_graph(graph.to_undirected())
        edge_coloring = nx.coloring.greedy_color(undirected_edge_graph, strategy=strategy)
    else:
        edge_coloring = {edge: i for i, edge in enumerate(graph.edges)}
    return edge_coloring


def compute_radius_by_tag(
    graph: nx.DiGraph,
    *,
    color_strategy: str | Callable[..., Any] | None,
    default_radius: float,
    strict_if_grouped: bool = True,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> np.ndarray:
    """
    Build a lookup array radius_by_tag where tag == subdomain marker produced by NetworkMesh.

    IMPORTANT:
      - If color_strategy is None, each edge gets its own tag, so radii can vary per edge.
      - If color_strategy groups multiple edges into the same tag, then a single radius must
        represent the whole group. With strict_if_grouped=True we error if radii differ.
    """
    edge_coloring = _color_edges_like_networks_fenicsx(graph, color_strategy)
    if len(edge_coloring) == 0:
        return np.zeros((0,), dtype=np.float64)

    num_tags = int(max(edge_coloring.values())) + 1
    buckets: list[list[float]] = [[] for _ in range(num_tags)]

    for u, v in graph.edges:
        key = (int(u), int(v))
        tag = edge_coloring.get(key, None)
        if tag is None:
            # Be robust if the coloring dict ended up with reversed undirected keys
            tag = edge_coloring.get((int(v), int(u)), None)
        if tag is None:
            raise RuntimeError(f"Could not find a color/tag for edge {key}.")
        r = float(graph.edges[u, v].get("radius", default_radius))
        buckets[int(tag)].append(r)

    out = np.full((num_tags,), float(default_radius), dtype=np.float64)
    for tag, vals in enumerate(buckets):
        if len(vals) == 0:
            continue
        r0 = float(vals[0])
        if strict_if_grouped and len(vals) > 1:
            # If grouping is enabled, require a single consistent radius per tag.
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if not np.isclose(vmin, vmax, rtol=float(rtol), atol=float(atol)):
                raise ValueError(
                    "color_strategy grouped multiple edges into the same subdomain tag, "
                    "but their radii differ. Use color_strategy=None (unique tag per edge), "
                    "or pass an explicit cell_radius to PressureProblem instead.\n"
                    f"tag={tag}, radius_range=[{vmin}, {vmax}]"
                )
        out[tag] = r0
    return out