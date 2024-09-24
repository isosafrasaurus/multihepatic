from dolfin import *
from graphnics import *
from xii import *
import meshio
import networkx as nx
import numpy as np
import os
import scipy.spatial
import vtk

class FEMSink:
    

    
    def __init__(self, 
                 G: "FenicsGraph", 
                 kappa: float = 1.0, 
                 alpha: float = 9.6e-2, 
                 beta: float = 1.45e4, 
                 gamma: float = 1.0, 
                 P_infty: float = 1.0e3,
                 theta: float = 1.0,
                 P_sink: float = 1.0e3,
                 Omega_bbox: tuple = None):
        
        
        
        self.kappa = Constant(kappa)
        self.alpha = Constant(alpha)
        self.beta = Constant(beta)
        self.gamma = Constant(gamma)
        self.P_infty = Constant(P_infty)
        self.theta = Constant(theta)
        self.P_sink = Constant(P_sink)

        
        G.make_mesh()
        Lambda, G_mf = G.get_mesh()

        
        node_positions = nx.get_node_attributes(G, "pos")
        node_coords = np.asarray(list(node_positions.values()))
        G_kdt = scipy.spatial.cKDTree(node_coords)

        
        Omega = UnitCubeMesh(32, 32, 32)
        Omega_coords = Omega.coordinates()
        xl, yl, zl = (np.max(node_coords, axis=0) - np.min(node_coords, axis=0))
        
        if Omega_bbox is not None:
            Omega_coords[:, :] *= [Omega_bbox[0], Omega_bbox[1], Omega_bbox[2]]
        else:
            Omega_coords[:, :] *= [xl + 3, yl + 3, zl + 3]
        
        
        Omega.bounding_box_tree()

        self.Lambda, self.Omega = Lambda, Omega

        
        class BoundaryFace1(SubDomain):
            def inside(self, x, on_boundary):
                
                return on_boundary and near(x[0], 0.0)

        boundary_markers = MeshFunction("size_t", Omega, Omega.topology().dim()-1, 0)
        boundary_face1 = BoundaryFace1()
        boundary_face1.mark(boundary_markers, 1)
        ds = Measure("ds", domain=Omega, subdomain_data=boundary_markers)

        
        V3 = FunctionSpace(Omega, "CG", 1)
        V1 = FunctionSpace(Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = list(map(TrialFunction, W))
        v3, v1 = list(map(TestFunction, W))

        
        G_rf = RadiusFunction(G, G_mf, G_kdt, degree=1)
        self.G_rf = G_rf

        
        cylinder = Circle(radius=G_rf, degree=5)
        u3_avg = Average(u3, Lambda, cylinder)
        v3_avg = Average(v3, Lambda, cylinder)

        
        dxOmega = Measure("dx", domain=Omega)
        dxLambda = Measure("dx", domain=Lambda)
        dsLambda = Measure("ds", domain=Lambda)

        
        D_area = np.pi * G_rf**2
        D_perimeter = 2 * np.pi * G_rf

        
        a00 = (self.alpha * inner(grad(u3), grad(v3)) * dxOmega + 
               self.kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda + 
               self.theta * u3 * v3 * ds(1))

        a01 = -self.kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
        a10 = -self.kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
        a11 = (self.beta * inner(grad(u1), grad(v1)) * D_area * dxLambda + 
               self.kappa * inner(u1, v1) * D_perimeter * dxLambda - 
               self.gamma * inner(u1, v1) * dsLambda)

        a = [[a00, a01], [a10, a11]]

        L0 = (inner(Constant(0), v3_avg) * dxLambda + 
              self.theta * self.P_sink * v3 * ds(1))

        L1 = (inner(Constant(0), v1) * dxLambda - 
              self.gamma * inner(self.P_infty, v1) * dsLambda)

        L = [L0, L1]

        W_bcs = []

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
        
        
        os.makedirs(directory_path, exist_ok=True)
        output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
        output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
        self._FenicsGraph_to_vtk(self.Lambda, output_file_1d, self.G_rf, uh1d=self.uh1d)
        File(output_file_3d) << self.uh3d

    def _FenicsGraph_to_vtk(self, Lambda: Mesh, file_path: str, G_rf: "RadiusFunction", uh1d: Function = None):
        
        points = Lambda.coordinates()
        cells = {"line": Lambda.cells()}
        radius_values = np.array([G_rf(point) for point in points])

        
        if uh1d is not None:
            uh1d_values = uh1d.compute_vertex_values(Lambda)
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
    
    def __init__(self, G: "FenicsGraph", G_mf: MeshFunction, G_kdt: scipy.spatial.cKDTree, **kwargs):
        self.G = G
        self.G_mf = G_mf
        self.G_kdt = G_kdt
        super().__init__(**kwargs)

    def eval(self, value: np.ndarray, x: np.ndarray):
        p = (x[0], x[1], x[2])
        _, nearest_control_point_index = self.G_kdt.query(p)
        nearest_control_point = list(self.G.nodes)[nearest_control_point_index]
        value[0] = self.G.nodes[nearest_control_point]['radius']
    
    def value_shape(self) -> tuple:
        return ()
