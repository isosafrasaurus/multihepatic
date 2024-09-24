from dolfin import *
from graphnics import *
from xii import *
import meshio
import networkx as nx
import numpy as np
import os
import scipy.spatial
import vtk

from dolfin import UserExpression, MeshFunction
import numpy as np
from typing import List, Tuple
import networkx as nx
from rtree import index as rtree_index

class FEMEdge:
    def __init__(self, G, kappa, alpha, beta, gamma, del_Omega, P_infty):
        kappa, alpha, beta, gamma, del_Omega, P_infty = map(Constant, [kappa, alpha, beta, gamma, del_Omega, P_infty])
        self.Lambda, self.Omega = FEMUtility.load_mesh(G)
        rtree = RadiusFunction.build_spatial_index(G)
        
        V3 = FunctionSpace(Omega, "CG", 1)
        V1 = FunctionSpace(Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = list(map(TrialFunction, W))
        v3, v1 = list(map(TestFunction, W))

        G_rf = RadiusFunction(G, G_edge_marker, G_kdt)
        self.G_rf = G_rf
        cylinder = Circle(radius=G_rf, degree=5)
        u3_avg = Average(u3, Lambda, cylinder)
        v3_avg = Average(v3, Lambda, cylinder)

        # Dirac measures
        dxOmega = Measure("dx", domain=Omega)
        dxLambda = Measure("dx", domain=Lambda)
        dsLambda = Measure("ds", domain=Lambda)

        D_area = np.pi * G_rf ** 2
        D_perimeter = 2 * np.pi * G_rf

        a00 = alpha * inner(grad(u3), grad(v3)) * dxOmega + kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda
        a01 = -kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
        a10 = -kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
        a11 = beta * inner(grad(u1), grad(v1)) * D_area * dxLambda + kappa * inner(u1, v1) * D_perimeter * dxLambda - gamma * inner(u1, v1) * dsLambda
        L0 = inner(Constant(0), v3_avg) * dxLambda
        L1 = inner(Constant(0), v1) * dxLambda - gamma * inner(P_infty, v1) * dsLambda

        a = [[a00, a01], [a10, a11]]
        L = [L0, L1]
        W_bcs = [[DirichletBC(V3, del_Omega, boundary_Omega)], []]

        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))
        wh = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)
        uh3d, uh1d = wh
        uh3d.rename("3D Pressure", "3D Pressure Distribution")
        uh1d.rename("1D Pressure", "1D Pressure Distribution")
        self.uh3d, self.uh1d = uh3d, uh1d

    def save_vtk(self, directory_path: str):
        """
        Saves the computed solutions (pressure and sensitivity) to VTK files in the specified directory.

        Args:
            directory_path (str): The path to the directory where VTK files will be saved.
        """
        # Create output directory if it doesn't exist and save
        os.makedirs(directory_path, exist_ok=True)
        output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
        output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
        self._FenicsGraph_to_vtk(self.Lambda, output_file_1d, self.G_rf, uh1d=self.uh1d)
        File(output_file_3d) << self.uh3d
        
    def _FenicsGraph_to_vtk(self, Lambda: Mesh, file_path: str, G_rf: "RadiusFunction", uh1d: Function = None):
        """
        Saves a tube mesh as a VTK file with the option to include the 1D solution as a data array.

        Args:
            Lambda (dolfin.Mesh): The mesh to be saved.
            file_path (str): The path where the VTK file will be saved.
            G_rf (dolfin.UserExpression): A function to compute radius values at each point.
            uh1d (dolfin.Function, optional): A function representing 1D pressure data. Defaults to None.
        """
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
        
    def _point_in_cylinder(point, pos_u, pos_v, radius):
        """
        Checks if a point is inside the cylinder defined by pos_u, pos_v, and radius.
        """
        p = np.array(point)
        u = np.array(pos_u)
        v = np.array(pos_v)
        line = v - u
        line_length_sq = np.dot(line, line)
        if line_length_sq == 0:
            # u and v are the same point
            return np.linalg.norm(p - u) <= radius
        t = np.dot(p - u, line) / line_length_sq
        t = max(0, min(1, t))
        projection = u + t * line
        distance = np.linalg.norm(p - projection)
        return distance <= radius

    def _find_encapsulating_radius(G, spatial_idx, point):
        """
        Finds any edge in G whose cylinder encapsulates the given point.
        """
        # Query R-tree for candidate cylinders
        candidates = spatial_idx.intersection((point[0], point[1], point[2],
                                              point[0], point[1], point[2]),
                                             objects=True)
        
        for candidate in candidates:
            u, v, data = candidate.object
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            radius = data['radius']
            if self._point_in_cylinder(point, pos_u, pos_v, radius):
                return radius
        
        return None

class Uh3dAtLambda(UserExpression):
    """
    A user expression to evaluate the 3D solution (uh3d) at points on the 1D network (Lambda).

    Args:
        uh3d (dolfin.Function): The 3D solution function.
        **kwargs: Additional keyword arguments to be passed to the UserExpression constructor.
    """
    def __init__(self, uh3d: Function, **kwargs):
        self.uh3d = uh3d
        super().__init__(**kwargs)
    
    def eval(self, value: np.ndarray, x: np.ndarray):
        value[0] = self.uh3d(x)
    
    def value_shape(self) -> tuple:
        return ()

class RadiusFunction(UserExpression):
    """
    A user expression to compute the radius at a given point based on the encapsulating edge in a graph.

    Args:
        G (networkx.Graph): The graph containing edges with 'radius' attributes.
        G_edge_marker (MeshFunction): A mesh function associated with the graph (if needed).
        spatial_idx (rtree.index.Index): The R-tree spatial index for efficient edge querying.
        edge_data_list (List[Tuple]): A list containing edge data tuples (u, v, data).
        **kwargs: Additional keyword arguments to be passed to the UserExpression constructor.
    """
    def __init__(self, G: nx.Graph, G_edge_marker: MeshFunction,
                 spatial_idx: rtree_index.Index,
                 edge_data_list: List[Tuple], **kwargs):
        self.G = G
        self.G_edge_marker = G_edge_marker
        self.spatial_idx = spatial_idx
        self.edge_data_list = edge_data_list
        super().__init__(**kwargs)

    def eval(self, value: np.ndarray, x: np.ndarray):
        """
        Evaluate the radius at point x by finding if it's encapsulated by any edge's cylinder.

        Args:
            value (np.ndarray): The output value array to store the radius.
            x (np.ndarray): The 3D coordinates of the point where the radius is evaluated.
        """
        point = tuple(x[:3])  # Ensure it's a tuple of (x, y, z)

        # Query the R-tree with the point as a zero-volume box
        candidates = list(self.spatial_idx.intersection(point + point, objects=False))

        for edge_id in candidates:
            u, v, data = self.edge_data_list[edge_id]
            pos_u = np.array(self.G.nodes[u]['pos'])
            pos_v = np.array(self.G.nodes[v]['pos'])
            radius = data['radius']

            if self.point_in_cylinder(point, pos_u, pos_v, radius):
                value[0] = radius
                return  # Exit after finding the first encapsulating cylinder

        # If no encapsulating cylinder is found, assign a default radius (e.g., 0)
        value[0] = 0.0

    def value_shape(self) -> tuple:
        return ()

    @staticmethod
    def point_in_cylinder(point, pos_u, pos_v, radius):
        """
        Checks if a point is inside the cylinder defined by pos_u, pos_v, and radius.

        Args:
            point (tuple): The (x, y, z) coordinates of the point.
            pos_u (np.ndarray): The (x, y, z) coordinates of the first endpoint of the cylinder.
            pos_v (np.ndarray): The (x, y, z) coordinates of the second endpoint of the cylinder.
            radius (float): The radius of the cylinder.

        Returns:
            bool: True if the point is inside the cylinder, False otherwise.
        """
        p = np.array(point)
        u = pos_u
        v = pos_v
        line = v - u
        line_length_sq = np.dot(line, line)
        if line_length_sq == 0:
            # The cylinder is degenerate (both endpoints are the same)
            return np.linalg.norm(p - u) <= radius

        # Project point p onto the line segment uv, clamped between u and v
        t = np.dot(p - u, line) / line_length_sq
        t = np.clip(t, 0.0, 1.0)
        projection = u + t * line
        distance = np.linalg.norm(p - projection)
        return distance <= radius

    @staticmethod
    def build_edge_spatial_index(G):
        """
        Builds an R-tree spatial index for the cylinders represented by the graph edges.

        Args:
            G (networkx.Graph): The graph containing edges with 'radius' attributes.

        Returns:
            rtree.index.Index: The spatial index of edges.
            list: A list of edge data tuples (u, v, data).
        """
        p = rtree_index.Property()
        p.dimension = 3  # 3D indexing
        spatial_idx = rtree_index.Index(properties=p)
        edge_data_list = []

        for edge_id, (u, v, data) in enumerate(G.edges(data=True)):
            pos_u = np.array(G.nodes[u]['pos'])
            pos_v = np.array(G.nodes[v]['pos'])
            radius = data['radius']

            min_coords = np.minimum(pos_u, pos_v) - radius
            max_coords = np.maximum(pos_u, pos_v) + radius

            # R-tree expects bounding boxes in the form (minx, miny, minz, maxx, maxy, maxz)
            bbox = tuple(min_coords.tolist() + max_coords.tolist())
            spatial_idx.insert(edge_id, bbox)
            edge_data_list.append((u, v, data))

        return spatial_idx, edge_data_list
