# tissue/meshing.py
import os, json
from graphnics import FenicsGraph

# NEW:
from typing import Optional
import numpy as np
from dolfin import Mesh, MeshEditor, MeshFunction

try:
    import meshio  # type: ignore
except ImportError:  # pragma: no cover
    meshio = None


def _require_meshio() -> None:
    if meshio is None:
        raise RuntimeError(
            "meshio is required to read .vtk/.vtp files. "
            "Install it via `pip install meshio`."
        )
# END NEW


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


# NEW: build FenicsGraph from a 1D .vtk/.vtp file (polyline centerlines)
def get_fg_from_vtk(
    filename: str,
    *,
    radius_field: str = "Radius",
) -> FenicsGraph:
    """
    Build a FenicsGraph from a 1D .vtk/.vtp file containing line cells.

    Assumptions:
      - `filename` stores a mesh with `line` cells describing centerlines.
      - Vessel radii are in point_data[radius_field] (scalar), or if no such
        field is present, a default radius 1.0 is used.
    """
    _require_meshio()
    m = meshio.read(filename)

    # Find line cells
    line_cells = None
    for cell_block in m.cells:
        if "line" in cell_block.type:
            line_cells = cell_block.data
            break
    if line_cells is None:
        raise ValueError(
            f"No line cells found in {filename!r}; "
            f"found cell types {[c.type for c in m.cells]}"
        )

    points = m.points[:, :3]
    G = FenicsGraph()

    # Add nodes with coordinates
    for i, xyz in enumerate(points):
        G.add_node(i)
        G.nodes[i]["pos"] = [float(x) for x in xyz]

    # Radii from point _data
    radii = None
    if radius_field in m.point_data:
        radii = m.point_data[radius_field]
    elif m.point_data:
        # fall back to first scalar field if available
        key, arr = next(iter(m.point_data.items()))
        if arr.ndim == 1 or arr.shape[1] == 1:
            radii = arr

    if radii is not None:
        for i, r in enumerate(radii):
            G.nodes[i]["radius"] = float(r)
    else:
        for i in G.nodes:
            G.nodes[i]["radius"] = 1.0

    # Connect nodes according to lines
    for u, v in line_cells:
        u, v = int(u), int(v)
        G.add_edge(u, v)
        G.edges[u, v]["radius"] = 0.5 * (
            G.nodes[u]["radius"] + G.nodes[v]["radius"]
        )

    return G


# NEW: read 3D tetrahedral mesh from .vtk
def mesh_from_vtk(
    filename: str,
    *,
    cell_type_hint: str = "tetra",
) -> Mesh:
    """
    Convert a 3D .vtk unstructured grid to a FEniCS Mesh.

    Expects tetrahedral elements; by default looks for a cell type containing
    'tetra'. If cell_type_hint is more specific (e.g. 'tetra10') it is used first.
    """
    _require_meshio()
    m = meshio.read(filename)

    cells = None
    # Try exact hint first
    for cell_block in m.cells:
        if cell_type_hint in cell_block.type:
            cells = cell_block.data
            break
    # Fallback: any tetra-like cell
    if cells is None:
        for cell_block in m.cells:
            if "tetra" in cell_block.type:
                cells = cell_block.data
                break

    if cells is None:
        raise ValueError(
            f"Could not find tetrahedral cells in {filename!r}; "
            f"available cell types: {[c.type for c in m.cells]}"
        )

    points = m.points[:, :3]

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)
    editor.init_vertices(len(points))
    editor.init_cells(len(cells))

    for i, xyz in enumerate(points):
        editor.add_vertex(i, xyz)

    for i, cell in enumerate(cells):
        editor.add_cell(i, cell)

    editor.close()
    return mesh


# NEW: build sink boundary marker MeshFunction from a surface .vtk
def sink_markers_from_surface_vtk(
    Omega: Mesh,
    surface_filename: str,
    *,
    marker_value: int = 1,
    decimals: int = 12,
) -> MeshFunction:
    """
    Create a facet MeshFunction on `Omega` where facets belonging to the
    surface mesh from `surface_filename` are marked with `marker_value`.

    Assumptions:
      - `surface_filename` stores a surface mesh (triangles) whose points
        coincide (up to rounding) with vertices of `Omega`.
      - The file is typically produced from the same volume mesh.

    This is a geometric matching based on vertex coordinates.
    """
    _require_meshio()
    m = meshio.read(surface_filename)

    tri_cells = None
    for cell_block in m.cells:
        if "triangle" in cell_block.type:
            tri_cells = cell_block.data
            break
    if tri_cells is None:
        raise ValueError(
            f"No triangle cells found in {surface_filename!r}; "
            f"found cell types {[c.type for c in m.cells]}"
        )

    surf_points = m.points[:, :3]

    # Build lookup from rounded coordinates to Omega vertex indices
    coords = Omega.coordinates()
    from collections import defaultdict
    vert_lookup = defaultdict(list)
    for vid, xyz in enumerate(coords):
        key = tuple(np.round(xyz, decimals=decimals))
        vert_lookup[key].append(vid)

    # Map each surface-vertex to (one) Omega vertex index if possible
    surf_vertex_to_omega = []
    for xyz in surf_points:
        key = tuple(np.round(xyz, decimals=decimals))
        candidates = vert_lookup.get(key, [])
        surf_vertex_to_omega.append(candidates[0] if candidates else None)

    desired_facets_vertices = set()
    for tri in tri_cells:
        omega_vids = []
        for local in tri:
            omega_v = surf_vertex_to_omega[int(local)]
            if omega_v is None:
                omega_vids = []
                break
            omega_vids.append(int(omega_v))
        if omega_vids:
            desired_facets_vertices.add(tuple(sorted(omega_vids)))

    facet_markers = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)

    from dolfin import facets as dolfin_facets
    for f in dolfin_facets(Omega):
        vids = tuple(sorted(f.entities(0)))
        if vids in desired_facets_vertices:
            facet_markers[f] = marker_value

    return facet_markers
