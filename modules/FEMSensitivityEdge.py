from dolfin import *
from graphnics import *
from xii import *
import meshio
import networkx as nx
import numpy as np
import os
import scipy.spatial
import vtk

class FEMSensitivity:
    

    
    def __init__(self, 
      G: "FenicsGraph", 
      kappa: float = 1.0, 
      alpha: float = 9.6e-2, 
      beta: float = 1.45e4, 
      gamma: float = 1.0, 
      del_Omega: float = 3.0, 
      P_infty: float = 1.0e3,
      Omega_bbox : tuple = None
      ):

        
        kappa, alpha, beta, gamma, del_Omega, P_infty = map(Constant, [kappa, alpha, beta, gamma, del_Omega, P_infty])

        
        G.make_mesh()
        Lambda, G_mf = G.get_mesh()

        
        edge_positions = []
        edge_list = list(G.edges())
        for u, v in edge_list:
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            midpoint = ((pos_u[0] + pos_v[0])/2, (pos_u[1] + pos_v[1])/2, (pos_u[2] + pos_v[2])/2)
            edge_positions.append(midpoint)
        edge_coords = np.asarray(edge_positions)
        G_kdt = scipy.spatial.cKDTree(edge_coords)

        
        Omega = UnitCubeMesh(32, 32, 32)
        Omega_coords = Omega.coordinates()
        xl, yl, zl = (np.max(edge_coords, axis=0) - np.min(edge_coords, axis=0))
        
        if Omega_bbox is not None:
            Omega_coords[:, :] *= [Omega_bbox[0], Omega_bbox[1], Omega_bbox[2]]
        else:
            Omega_coords[:, :] *= [xl + 3, yl + 3, zl + 3]
        
        
        def boundary_Omega(x, on_boundary):
            return on_boundary and not near(x[2], 0) and not near(x[2], zl)

        self.Lambda, self.Omega = Lambda, Omega

        
        V3 = FunctionSpace(Omega, "CG", 1)
        V1 = FunctionSpace(Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = list(map(TrialFunction, W))
        v3, v1 = list(map(TestFunction, W))

        
        G_rf = RadiusFunction(G, edge_list, G_kdt, degree=5)
        self.G_rf = G_rf
        cylinder = Circle(radius=G_rf, degree=5)
        u3_avg = Average(u3, Lambda, cylinder)
        v3_avg = Average(v3, Lambda, cylinder)

        
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

        
        s3k, s1k = list(map(TrialFunction, W))
        v3k, v1k = list(map(TestFunction, W))
        s3k_avg = Average(s3k, Lambda, cylinder)
        v3k_avg = Average(v3k, Lambda, cylinder)
        u3h_at_Lambda = interpolate(Uh3dAtLambda(uh3d, degree=uh3d.function_space().ufl_element().degree()), V1)

        
        a00_sens = alpha * inner(grad(s3k), grad(v3k)) * dxOmega + kappa * inner(s3k_avg, v3k_avg) * D_perimeter * dxLambda
        a01_sens = -kappa * inner(s1k, v3k_avg) * D_perimeter * dxLambda
        a10_sens = -kappa * inner(s3k_avg, v1k) * D_perimeter * dxLambda
        a11_sens = beta * inner(grad(s1k), grad(v1k)) * D_area * dxLambda + kappa * inner(s1k, v1k) * D_perimeter * dxLambda

        
        L0_sens = inner(uh1d - u3h_at_Lambda, v3k_avg) * D_perimeter * dxLambda
        L1_sens = inner(u3h_at_Lambda - uh1d, v1k) * D_perimeter * dxLambda

        
        a_sens = [[a00_sens, a01_sens], [a10_sens, a11_sens]]
        L_sens = [L0_sens, L1_sens]

        A_sens, b_sens = map(ii_assemble, (a_sens, L_sens))
        A_sens, b_sens = apply_bc(A_sens, b_sens, W_bcs)
        A_sens, b_sens = map(ii_convert, (A_sens, b_sens))

        
        wh_sens = ii_Function(W)
        solver_sens = LUSolver(A_sens, "mumps")
        solver_sens.solve(wh_sens.vector(), b_sens)
        sh3d, sh1d = wh_sens
        sh3d.rename("Sensitivity 3D", "Sensitivity 3D Distribution")
        sh1d.rename("Sensitivity 1D", "Sensitivity 1D Distribution")
        self.sh3d, self.sh1d = sh3d, sh1d
    
    def save_vtk(self, directory_path: str):
        
        
        os.makedirs(directory_path, exist_ok=True)
        output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
        output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
        output_file_sens_1d = os.path.join(directory_path, "sensitivity1d.vtk")
        output_file_sens_3d = os.path.join(directory_path, "sensitivity3d.pvd")
        self._FenicsGraph_to_vtk(self.Lambda, output_file_1d, self.G_rf, uh1d=self.uh1d)
        self._FenicsGraph_to_vtk(self.Lambda, output_file_sens_1d, self.G_rf, uh1d=self.sh1d)
        File(output_file_3d) << self.uh3d
        File(output_file_sens_3d) << self.sh3d
        
    def _FenicsGraph_to_vtk(self, Lambda: Mesh, file_path: str, G_rf: "RadiusFunction", uh1d: Function = None):
        
        points = Lambda.coordinates()
        cells = {"line": Lambda.cells()}
        radius_values = np.array([G_rf(point) for point in points])

        
        if uh1d is not None:
            uh1d_values = np.array([uh1d(point) for point in points])
            mesh = meshio.Mesh(points, cells, point_data={"radius": radius_values, "Pressure1D": uh1d_values})
        else:
            mesh = meshio.Mesh(points, cells, point_data={"radius": radius_values})
        mesh.write(file_path)

        
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

class Uh3dAtLambda(UserExpression):
    
    def __init__(self, uh3d: Function, **kwargs):
        self.uh3d = uh3d
        super().__init__(**kwargs)
    
    def eval(self, value: np.ndarray, x: np.ndarray):
        value[0] = self.uh3d(x)
    
    def value_shape(self) -> tuple:
        return ()

class RadiusFunction(UserExpression):
    
    def __init__(self, G: "FenicsGraph", edge_list: list, G_kdt: scipy.spatial.cKDTree, **kwargs):
        self.G = G
        self.edge_list = edge_list
        self.G_kdt = G_kdt
        super().__init__(**kwargs)

    def eval(self, value: np.ndarray, x: np.ndarray):
        p = (x[0], x[1], x[2])
        _, nearest_edge_index = self.G_kdt.query(p)
        u, v = self.edge_list[nearest_edge_index]
        radius = self.G.edges[u, v]['radius']
        value[0] = radius

    def value_shape(self) -> tuple:
        return ()
