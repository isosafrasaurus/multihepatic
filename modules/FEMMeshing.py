from dolfin import *
from graphnics import *
from xii import *
import meshio
import networkx as nx
import numpy as np
import os
import scipy.spatial
import vtk
from FEMUtility import FEMUtility
from rtree import index as rtree_index
from typing import List, Tuple

class FEMMeshing:
    def __init__(self, G: "FenicsGraph", kappa: float = 1.0, alpha: float = 9.6e-2, beta: float = 1.45e4, gamma: float = 1.0, del_Omega: float = 3.0, P_infty: float = 1.0e3):
        kappa, alpha, beta, gamma, del_Omega, P_infty = map(Constant, [kappa, alpha, beta, gamma, del_Omega, P_infty])
        Lambda, Omega, boundary_Omega, G_mf, G_kdt = FEMUtility.load_mesh_test(G)
        self.Lambda, self.Omega = Lambda, Omega

        V3 = FunctionSpace(Omega, "CG", 1)
        V1 = FunctionSpace(Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = list(map(TrialFunction, W))
        v3, v1 = list(map(TestFunction, W))

        spatial_idx, edge_data_list = self.build_edge_spatial_index(G)
        self.G_rf = RadiusFunction(G=G,
                              G_edge_marker=G_mf,
                              spatial_idx=spatial_idx,
                              edge_data_list=edge_data_list,
                              degree=0)
        cylinder = Circle(radius=G_rf, degree=5)
        u3_avg = Average(u3, Lambda, cylinder)
        v3_avg = Average(v3, Lambda, cylinder)

        dxOmega = Measure("dx", domain=Omega)
        dxLambda = Measure("dx", domain=Lambda)
        dsLambda = Measure("ds", domain=Lambda)

        D_area = pi * pow(G_rf, 2)
        D_perimeter = 2 * pi * G_rf

        a00 = alpha * inner(grad(u3), grad(v3)) * dxOmega + kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda
        a01 = -kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
        a10 = -kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
        a11 = beta * inner(grad(u1), grad(v1)) * D_area * dxLambda + \
               kappa * inner(u1, v1) * D_perimeter * dxLambda - \
               gamma * inner(u1, v1) * dsLambda
        L0 = inner(Constant(0), v3_avg) * dxLambda
        L1 = inner(Constant(0), v1) * dxLambda - gamma * inner(Constant(P_infty), v1) * dsLambda

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

    def build_edge_spatial_index(self, G: nx.Graph):
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

    def save_vtk(self, directory_path: str):
        os.makedirs(directory_path, exist_ok=True)
        output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
        output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
        FEMUtility.fenics_to_vtk(self.Lambda, output_file_1d, self.G_rf, uh1d=self.uh1d)
        File(output_file_3d) << self.uh3d

class RadiusFunction(UserExpression):
    def __init__(self, 
                 G: nx.Graph, 
                 G_edge_marker: MeshFunction,
                 spatial_idx: rtree_index.Index,
                 edge_data_list: List[Tuple],
                 **kwargs):
        self.G = G
        self.G_edge_marker = G_edge_marker
        self.spatial_idx = spatial_idx
        self.edge_data_list = edge_data_list
        super().__init__(**kwargs)

    def eval(self, value: np.ndarray, x: np.ndarray):
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