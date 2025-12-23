from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
from dolfin import Mesh, MeshEditor, MeshFunction, facets as dolfin_facets
from graphnics import FenicsGraph

try:
    import meshio
except ImportError:
    meshio = None

try:
    import vtk
except ImportError:
    vtk = None


def require_vtk() -> None:
    if vtk is None:
        raise RuntimeError(
            "The 'vtk' Python package is required for VTK/VTP inputs. "
            "Install it via `pip install vtk`."
        )


def require_meshio() -> None:
    if meshio is None:
        raise RuntimeError(
            "meshio is required for certain VTK/VTU conversion paths. "
            "Install it via `pip install meshio`."
        )


def get_fg_from_vtk(filename: str, *, radius_field: str = "Radius") -> FenicsGraph:
    require_vtk()

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


def mesh_from_vtk(filename: str, *, use_delaunay_if_polydata: bool = True) -> Mesh:
    require_vtk()

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


def sink_markers_from_surface_vtk(
        Omega: Mesh,
        surface_filename: str,
        *,
        marker_value: int = 1,
        decimals: int = 12,
) -> MeshFunction:
    require_vtk()

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

    pts = poly.GetPoints()
    n_pts = pts.GetNumberOfPoints()
    surf_points = np.array([pts.GetPoint(i) for i in range(n_pts)], dtype=float)

    polys = poly.GetPolys()
    id_list = vtk.vtkIdList()
    polys.InitTraversal()
    surf_tris = []
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            surf_tris.append([int(id_list.GetId(0)), int(id_list.GetId(1)), int(id_list.GetId(2))])

    if not surf_tris:
        raise RuntimeError(f"No triangular cells found in sink surface {surface_filename!r}")

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

    facet_markers = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)

    for f in dolfin_facets(Omega):
        vids = tuple(sorted(f.entities(0)))
        if vids in desired_facets_vertices:
            facet_markers[f] = marker_value

    return facet_markers


__all__ = [
    "require_vtk",
    "require_meshio",
    "get_fg_from_vtk",
    "mesh_from_vtk",
    "sink_markers_from_surface_vtk",
]
