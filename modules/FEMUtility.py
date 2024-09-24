from dolfin import *
from graphnics import *
from xii import *
from rtree import index
import meshio
import networkx as nx
import numpy as np
import os
import scipy.spatial
import vtk

class FEMUtility:
    def fenicsgraph_to_vtk(self, Lambda: Mesh, file_path: str, G_rf: "RadiusFunction", uh1d: Function = None):
        points = Lambda.coordinates()
        cells = {"line": Lambda.cells()}
        radius_values = np.array([G_rf(point) for point in points])

        # Evaluate uh1d function at each node in the mesh
        if uh1d is not None:
            uh1d_values = np.array([uh1d(point) for point in points])
            mesh = meshio.Mesh(points, cells, point_data={"radius": radius_values, "Pressure1D": uh1d_values})
        else:
            mesh = meshio.Mesh(points, cells, point_data={"radius": radius_values})
        mesh.write(file_path)

        # Convert the mesh to Polydata using VTK
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(file_path)
        reader.Update()
        geometryFilter = vtk.vtkGeometryFilter()
        geometryFilter.SetInputData(reader.GetOutput())
        geometryFilter.Update()
        polydata = geometryFilter.GetOutput()
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(polydata)
        writer.Write()
    
    def load_mesh(G):
        G.make_mesh()
        Lambda, edge_marker = G.get_mesh()
        
        node_positions = nx.get_node_attributes(G, "pos")
        node_coords = np.asarray(list(node_positions.values()))

        Omega = UnitCubeMesh(32, 32, 32)
        Omega_coords = Omega.coordinates()
        xl, yl, zl = (np.max(node_coords, axis=0) - np.min(node_coords, axis=0))
        Omega_coords[:, :] *= [xl + 3, yl + 3, zl + 3]
        return Lambda, Omega
        
    def build_spatial_index(G):
        """
        Builds an R-tree spatial index for the cylinders represented by the graph edges.
        """
        p = index.Property()
        p.dimension = 3  # 3D
        idx = index.Index(properties=p)
        
        for edge_id, (u, v, data) in enumerate(G.edges(data=True)):
            pos_u = np.array(G.nodes[u]['pos'])
            pos_v = np.array(G.nodes[v]['pos'])
            radius = data['radius']
            
            min_coords = np.minimum(pos_u, pos_v) - radius
            max_coords = np.maximum(pos_u, pos_v) + radius
            
            # R-tree expects bounding boxes in the form (minx, miny, minz, maxx, maxy, maxz)
            bbox = tuple(min_coords.tolist() + max_coords.tolist())
            idx.insert(edge_id, bbox, obj=(u, v, data))
        
        return idx