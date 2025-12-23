from __future__ import annotations

from dolfin import Mesh, MeshEditor
from graphnics import FenicsGraph
import vtk

def vtk_to_graph(filename: str, radius_field: str = "Radius") -> FenicsGraph:
    fname = filename.lower()
    if fname.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()

    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()

    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Failed to read any points from {filename!r}")

    points_vtk = poly.GetPoints()
    n_points = points_vtk.GetNumberOfPoints()

    G = FenicsGraph()

    for i in range(n_points):
        x, y, z = points_vtk.GetPoint(i)
        G.add_node(i)
        G.nodes[i]["pos"] = [float(x), float(y), float(z)]

    radii = None
    pd = poly.GetPointData()
    if pd is not None:
        arr = pd.GetArray(radius_field)
        if arr is None:
            for j in range(pd.GetNumberOfArrays()):
                name_j = pd.GetArrayName(j)
                if name_j and "radius" in name_j.lower():
                    arr = pd.GetArray(j)
                    radius_field = name_j
                    break
        if arr is not None:
            radii = [float(arr.GetTuple1(i)) for i in range(n_points)]

    if radii is None:
        radii = [1.0] * n_points

    for i, r in enumerate(radii):
        G.nodes[i]["radius"] = float(r)

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
                G.edges[u, v]["radius"] = 0.5 * (G.nodes[u]["radius"] + G.nodes[v]["radius"])
            prev = curr

    return G


def vtk_to_mesh(filename: str, use_delaunay_if_polydata: bool = True) -> Mesh:
    fname = filename.lower()
    if fname.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fname.endswith(".vtk"):
        reader = vtk.vtkDataSetReader()
    else:
        reader = vtk.vtkGenericDataObjectReader()

    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    if isinstance(data, vtk.vtkUnstructuredGrid):
        ugrid = data
    elif isinstance(data, vtk.vtkPolyData):
        if not use_delaunay_if_polydata:
            raise ValueError(
                f"VTK file {filename!r} is POLYDATA (surface). "
                "Provide a volume mesh or enable use_delaunay_if_polydata."
            )
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(data)
        delaunay.Update()
        ugrid = delaunay.GetOutput()
    else:
        raise ValueError(f"Unsupported VTK dataset type {type(data).__name__} in {filename!r}")

    if ugrid is None or ugrid.GetNumberOfPoints() == 0:
        raise RuntimeError(f"No points found in VTK dataset {filename!r}")

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


__all__ = [
    "vtk_to_graph",
    "vtk_to_mesh",
]