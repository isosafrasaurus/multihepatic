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
    @staticmethod
    def load_mesh(G, Omega_box: list[float] = None, robin_endpoints: list[int] = None):
        G.make_mesh()
        Lambda, edge_marker = G.get_mesh()

        node_positions = nx.get_node_attributes(G, "pos")
        node_coords = np.asarray(list(node_positions.values()))

        # Fit Omega around Lambda unless Omega_box is provided
        Omega = UnitCubeMesh(32, 32, 32)
        Omega_coords = Omega.coordinates()
        if Omega_box is None:
            xmax, ymax, zmax = np.max(node_coords, axis=0)
            xmin, ymin, zmin = np.min(node_coords, axis=0)
            Omega_coords[:, :] *= [xmax - xmin + 10, ymax - ymin + 10, zmax - zmin + 10]
            Omega_coords[:, :] += [xmin - 5, ymin - 5, zmin - 5]
        else:
            Omega_coords[:, :] *= [Omega_box[3] - Omega_box[0], Omega_box[4] - Omega_box[1], Omega_box[5] - Omega_box[2]]
            Omega_coords[:, :] += [Omega_box[0], Omega_box[1], Omega_box[2]]

        # Create a MeshFunction for boundary markers on Omega
        boundary_markers = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)

        # Define Face 1 as the top face (z = 1.0)
        class Face1(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], 1.0, DOLFIN_EPS)

        # Instantiate and mark Face 1 with marker '1'
        face1 = Face1()
        face1.mark(boundary_markers, 1)

        # Create a MeshFunction for boundary markers on Lambda
        lambda_boundary_markers = MeshFunction("size_t", Lambda, Lambda.topology().dim() - 1, 0)

        if robin_endpoints is not None:
            # Define a SubDomain for each Robin endpoint
            for endpoint in robin_endpoints:
                class RobinEndpoint(SubDomain):
                    def __init__(self, point):
                        super().__init__()
                        self.point = point

                    def inside(self, x, on_boundary):
                        return on_boundary and near(x[0], self.point[0], DOLFIN_EPS) and \
                               near(x[1], self.point[1], DOLFIN_EPS) and \
                               near(x[2], self.point[2], DOLFIN_EPS)

                pos = G.nodes[endpoint]['pos']
                robin_subdomain = RobinEndpoint(pos)
                robin_subdomain.mark(lambda_boundary_markers, 1)  # Marker '1' for Robin

        return Lambda, Omega, boundary_markers, edge_marker, lambda_boundary_markers

    @staticmethod
    def fenics_to_vtk(Lambda: Mesh, file_path: str, radius_map: "RadiusFunction", uh1d: Function = None):
        points = Lambda.coordinates()
        cells = {"line": Lambda.cells()}
        radius_values = np.array([radius_map(point) for point in points])

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