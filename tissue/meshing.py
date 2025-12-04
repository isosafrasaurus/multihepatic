# tissue/meshing.py
import os, json
from graphnics import FenicsGraph
import numpy as np
from dolfin import Mesh, MeshEditor, MeshFunction, facets as dolfin_facets
from collections import defaultdict

# existing:
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

try:
    import vtk  # type: ignore
except ImportError:  # pragma: no cover
    vtk = None


def _require_vtk() -> None:
    if vtk is None:
        raise RuntimeError(
            "The 'vtk' Python package is required to read POLYDATA .vtk "
            "centerline files (e.g. VMTK output). Install it via `pip install vtk`."
        )


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
    Build a FenicsGraph from a 1D .vtk/.vtp centerline file.

    This version uses the VTK Python bindings and supports POLYDATA,
    which is what your sortedVesselNetwork.vtk file is.
    """
    _require_vtk()

    # Choose the appropriate reader
    fname = filename.lower()
    if fname.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        # legacy .vtk, which is what you have
        reader = vtk.vtkPolyDataReader()

    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read any points from {filename!r}")

    points_vtk = poly.GetPoints()
    n_points = points_vtk.GetNumberOfPoints()

    G = FenicsGraph()

    # --- Nodes: positions ---
    for i in range(n_points):
        x, y, z = points_vtk.GetPoint(i)
        G.add_node(i)
        G.nodes[i]["pos"] = [float(x), float(y), float(z)]

    # --- Node radii from point data (if available) ---
    radii = None
    pd = poly.GetPointData()
    if pd is not None:
        arr = pd.GetArray(radius_field)

        # Fallback: look for *any* array with "radius" in the name
        if arr is None:
            for j in range(pd.GetNumberOfArrays()):
                name_j = pd.GetArrayName(j)
                if name_j and "radius" in name_j.lower():
                    arr = pd.GetArray(j)
                    radius_field = name_j
                    break

        if arr is not None:
            radii = [float(arr.GetTuple1(i)) for i in range(n_points)]

    # If no radius data, default to 1.0
    if radii is None:
        radii = [1.0] * n_points

    for i, r in enumerate(radii):
        G.nodes[i]["radius"] = float(r)

    # --- Edges: from polyline connectivity ---
    # Centerlines are typically stored in polydata "Lines"
    lines = poly.GetLines()
    id_list = vtk.vtkIdList()
    lines.InitTraversal()

    while lines.GetNextCell(id_list):
        n_ids = id_list.GetNumberOfIds()
        if n_ids < 2:
            continue

        prev = int(id_list.GetId(0))
        for k in range(1, n_ids):
            curr = int(id_list.GetId(k))
            u, v = prev, curr
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                G.edges[u, v]["radius"] = 0.5 * (
                    G.nodes[u]["radius"] + G.nodes[v]["radius"]
                )
            prev = curr

    return G



# NEW: read 3D tetrahedral mesh from .vtk
def mesh_from_vtk(
    filename: str,
    *,
    use_delaunay_if_polydata: bool = True,
) -> Mesh:
    """
    Build a 3D tetrahedral FEniCS Mesh from a VTK file.

    - If the file contains a vtkUnstructuredGrid with tetra cells, we just
      convert that.
    - If the file contains vtkPolyData (e.g. from `nii2mesh`), and
      use_delaunay_if_polydata=True, we run vtkDelaunay3D to generate
      a tetrahedral volume mesh from the surface / point cloud.

    NOTE: Delaunay3D gives a tetrahedralization of the point set (typically
    the convex hull). This is an approximation of your liver volume; if you
    need a high‑quality conforming mesh, you’d want an external tet mesher
    (TetGen, gmsh, etc.).
    """
    _require_vtk()

    fname = filename.lower()
    # Use generic reader that can handle multiple dataset types
    if fname.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fname.endswith(".vtk"):
        reader = vtk.vtkDataSetReader()
    else:
        # Fall back to generic data object reader
        reader = vtk.vtkGenericDataObjectReader()

    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    # Case 1: already an unstructured grid
    if isinstance(data, vtk.vtkUnstructuredGrid):
        ugrid = data

    # Case 2: POLYDATA surface -> tetrahedralize via Delaunay3D
    elif isinstance(data, vtk.vtkPolyData):
        if not use_delaunay_if_polydata:
            raise ValueError(
                f"VTK file {filename!r} is POLYDATA (surface). "
                "You need a volume (unstructured grid) mesh or enable "
                "use_delaunay_if_polydata."
            )

        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(data)
        # Default alpha=0 -> full tetrahedralization of convex hull
        delaunay.Update()
        ugrid = delaunay.GetOutput()

    else:
        raise ValueError(
            f"Unsupported VTK dataset type {type(data).__name__} in {filename!r}"
        )

    if ugrid is None or ugrid.GetNumberOfPoints() == 0:
        raise RuntimeError(f"No points found in VTK dataset {filename!r}")

    # Extract tetrahedral cells
    tetrahedra = []
    id_list = vtk.vtkIdList()
    n_cells = ugrid.GetNumberOfCells()
    for ci in range(n_cells):
        ctype = ugrid.GetCellType(ci)
        if ctype == vtk.VTK_TETRA:
            ugrid.GetCellPoints(ci, id_list)
            if id_list.GetNumberOfIds() != 4:
                continue
            tetrahedra.append([id_list.GetId(j) for j in range(4)])

    if not tetrahedra:
        raise RuntimeError(
            f"No tetrahedral cells (VTK_TETRA) found in {filename!r} "
            f"(dataset type {type(ugrid).__name__})."
        )

    # Build FEniCS mesh via MeshEditor
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)

    n_points = ugrid.GetNumberOfPoints()
    editor.init_vertices(n_points)
    editor.init_cells(len(tetrahedra))

    for i in range(n_points):
        x, y, z = ugrid.GetPoint(i)
        editor.add_vertex(i, (float(x), float(y), float(z)))

    for i, tet in enumerate(tetrahedra):
        editor.add_cell(i, [int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])])

    editor.close()
    return mesh


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
      - `surface_filename` is a surface mesh (triangles) in VTK POLYDATA
        or XML-PolyData (.vtp) format.
      - Its points lie on the boundary of the volume mesh `Omega`.
      - The VTK surface points coincide (up to rounding) with vertices
        of `Omega`. We match by coordinates up to `decimals` digits.
    """
    _require_vtk()

    fname = surface_filename.lower()
    if fname.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()

    reader.SetFileName(surface_filename)
    reader.Update()
    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read polydata from {surface_filename!r}")

    # Surface points
    pts = poly.GetPoints()
    n_pts = pts.GetNumberOfPoints()
    surf_points = np.array([pts.GetPoint(i) for i in range(n_pts)], dtype=float)

    # Triangles connectivity
    polys = poly.GetPolys()
    id_list = vtk.vtkIdList()
    polys.InitTraversal()
    surf_tris = []
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            surf_tris.append(
                [int(id_list.GetId(0)),
                 int(id_list.GetId(1)),
                 int(id_list.GetId(2))]
            )

    if not surf_tris:
        raise RuntimeError(
            f"No triangular cells found in sink surface {surface_filename!r}"
        )

    # Map surface vertices -> Omega vertex indices via coordinate rounding
    coords = Omega.coordinates()
    vert_lookup = defaultdict(list)
    for vid, xyz in enumerate(coords):
        key = tuple(np.round(xyz, decimals=decimals))
        vert_lookup[key].append(vid)

    surf_vertex_to_omega = [None] * n_pts
    for i, xyz in enumerate(surf_points):
        key = tuple(np.round(xyz, decimals=decimals))
        cands = vert_lookup.get(key, [])
        if cands:
            surf_vertex_to_omega[i] = cands[0]

    desired_facets_vertices = set()
    for tri in surf_tris:
        mapped = []
        for local in tri:
            omega_vid = surf_vertex_to_omega[local]
            if omega_vid is None:
                mapped = []
                break
            mapped.append(int(omega_vid))
        if mapped:
            desired_facets_vertices.add(tuple(sorted(mapped)))

    facet_markers = MeshFunction(
        "size_t", Omega, Omega.topology().dim() - 1, 0
    )

    # Mark facets whose vertex set matches any surface triangle's vertex set
    for f in dolfin_facets(Omega):
        vids = tuple(sorted(f.entities(0)))
        if vids in desired_facets_vertices:
            facet_markers[f] = marker_value

    return facet_markers