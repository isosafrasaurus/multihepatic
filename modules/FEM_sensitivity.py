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
    """
    A class for performing finite element method (FEM) sensitivity analysis on a 3D domain containing a 1D network.

    This class sets up and solves a steady-state diffusion-reaction problem with different properties in the 3D domain
    and the 1D network. It then computes the sensitivity of the solution with respect to perturbations in the 1D network.

    Attributes:
        G (FenicsGraph): The FenicsGraph object representing the 1D network.
        kappa (float, optional): Coupling coefficient between 3D and 1D domains.
        alpha (float, optional): Diffusion coefficient in the 3D domain.
        beta (float, optional): Diffusion coefficient in the 1D network.
        gamma (float, optional): Reaction coefficient in the 1D network.
        del_Omega (float, optional): Boundary condition value on the 3D domain.
        P_infty (float, optional): Source term value in the 1D network.
        Lambda (dolfin.Mesh): The mesh representing the 1D network.
        Omega (dolfin.Mesh): The mesh representing the 3D domain.
        G_rf (dolfin.UserExpression): A function to compute radius values at each point in the 1D network.
        uh3d (dolfin.Function): The solution for 3D pressure.
        uh1d (dolfin.Function): The solution for 1D pressure.
        sh3d (dolfin.Function): The sensitivity of 3D pressure.
        sh1d (dolfin.Function): The sensitivity of 1D pressure.
    """
    def __init__(self, G: "FenicsGraph", kappa: float = 1.0, alpha: float = 9.6e-2, beta: float = 1.45e4, gamma: float = 1.0, del_Omega: float = 3.0, P_infty: float = 1.0e3):
        """
        Initializes the FEMSensitivity class with given parameters and solves the steady-state and sensitivity problems.

        Args:
            G (FenicsGraph): The FenicsGraph object representing the 1D network.
            kappa (float, optional): Coupling coefficient between 3D and 1D domains.
            alpha (float, optional): Diffusion coefficient in the 3D domain.
            beta (float, optional): Diffusion coefficient in the 1D network.
            gamma (float, optional): Reaction coefficient in the 1D network.
            del_Omega (float, optional): Boundary condition value on the 3D domain.
            P_infty (float, optional): Source term value in the 1D network.
        """
        kappa, alpha, beta, gamma, del_Omega, P_infty = map(Constant, [kappa, alpha, beta, gamma, del_Omega, P_infty])

        # Create meshes
        G.make_mesh()
        Lambda, G_mf = G.get_mesh()
        Omega = UnitCubeMesh(16, 16, 16)

        # Translate all Lambda points to positive
        node_positions = nx.get_node_attributes(G, "pos")
        node_coords = np.asarray(list(node_positions.values()))
        xmin, ymin, zmin = np.min(node_coords, axis=0)
        G_kdt = scipy.spatial.cKDTree(node_coords[:,:] + [-xmin, -ymin, -zmin])
        Lambda_coords = Lambda.coordinates()
        Lambda_coords[:, :] += [-xmin, -ymin, -zmin]

        # Fit Omega around Lambda
        Omega_coords = Omega.coordinates()
        xl, yl, zl = (np.max(node_coords, axis=0) - np.min(node_coords, axis=0))
        Omega_coords[:, :] *= [xl + 3, yl + 3, zl]
        self.Lambda, self.Omega = Lambda, Omega
        
        # Omega boundary function
        def boundary_Omega(x, on_boundary):
            return on_boundary and not near(x[2], 0) and not near(x[2], zl)

        # Function spaces
        V3 = FunctionSpace(Omega, "CG", 1)
        V1 = FunctionSpace(Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = list(map(TrialFunction, W))
        v3, v1 = list(map(TestFunction, W))

        G_rf = RadiusFunction(G, G_mf, G_kdt)
        self.G_rf = G_rf
        cylinder = Circle(radius=G_rf, degree=5)
        u3_avg = Average(u3, Lambda, cylinder)
        v3_avg = Average(v3, Lambda, cylinder)

        # Dirac measures
        dxOmega = Measure("dx", domain=Omega)
        dxLambda = Measure("dx", domain=Lambda)
        dsLambda = Measure("ds", domain=Lambda)

        # Define D_area and D_perimeter
        D_area = np.pi * G_rf ** 2
        D_perimeter = 2 * np.pi * G_rf

        # Blocks for the steady-state problem
        a00 = alpha * inner(grad(u3), grad(v3)) * dxOmega + kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda
        a01 = -kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
        a10 = -kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
        a11 = beta * inner(grad(u1), grad(v1)) * D_area * dxLambda + kappa * inner(u1, v1) * D_perimeter * dxLambda - gamma * inner(u1, v1) * dsLambda

        # Right-hand side for the steady-state problem
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

        # Define trial and test functions for sensitivity variables
        s3k, s1k = list(map(TrialFunction, W))
        v3k, v1k = list(map(TestFunction, W))
        s3k_avg = Average(s3k, Lambda, cylinder)
        v3k_avg = Average(v3k, Lambda, cylinder)
        u3h_at_Lambda = interpolate(Uh3dAtLambda(uh3d, degree=uh3d.function_space().ufl_element().degree()), V1)

        # Left-hand side for the sensitivity problem
        a00_sens = alpha * inner(grad(s3k), grad(v3k)) * dxOmega + kappa * inner(s3k_avg, v3k_avg) * D_perimeter * dxLambda
        a01_sens = -kappa * inner(s1k, v3k_avg) * D_perimeter * dxLambda
        a10_sens = -kappa * inner(s3k_avg, v1k) * D_perimeter * dxLambda
        a11_sens = beta * inner(grad(s1k), grad(v1k)) * D_area * dxLambda + kappa * inner(s1k, v1k) * D_perimeter * dxLambda

        # Right-hand side for the sensitivity problem
        L0_sens = inner(uh1d - u3h_at_Lambda, v3k_avg) * D_perimeter * dxLambda
        L1_sens = inner(u3h_at_Lambda - uh1d, v1k) * D_perimeter * dxLambda

        # Assemble the sensitivity system
        a_sens = [[a00_sens, a01_sens], [a10_sens, a11_sens]]
        L_sens = [L0_sens, L1_sens]

        A_sens, b_sens = map(ii_assemble, (a_sens, L_sens))
        A_sens, b_sens = apply_bc(A_sens, b_sens, W_bcs)
        A_sens, b_sens = map(ii_convert, (A_sens, b_sens))

        # Solve the sensitivity system
        wh_sens = ii_Function(W)
        solver_sens = LUSolver(A_sens, "mumps")
        solver_sens.solve(wh_sens.vector(), b_sens)
        sh3d, sh1d = wh_sens
        sh3d.rename("Sensitivity 3D", "Sensitivity 3D Distribution")
        sh1d.rename("Sensitivity 1D", "Sensitivity 1D Distribution")
        self.sh3d, self.sh1d = sh3d, sh1d
    
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
        output_file_sens_1d = os.path.join(directory_path, "sensitivity1d.vtk")
        output_file_sens_3d = os.path.join(directory_path, "sensitivity3d.pvd")
        self._FenicsGraph_to_vtk(self.Lambda, output_file_1d, self.G_rf, uh1d=self.uh1d)
        self._FenicsGraph_to_vtk(self.Lambda, output_file_sens_1d, self.G_rf, uh1d=self.sh1d)
        File(output_file_3d) << self.uh3d
        File(output_file_sens_3d) << self.sh3d
        
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

class Uh3dAtLambda(UserExpression):
    """
    A user expression to evaluate the 3D solution (uh3d) at points on the 1D network (Lambda).

    Args:
        uh3d (dolfin.Function): The 3D solution function.
        **kwargs: Additional keyword arguments to be passed to the UserExpression constructor.
    """
    def __init__(self, uh3d: Function, **kwargs):
        """
        Initializes the Uh3dAtLambda class.

        Args:
            uh3d (dolfin.Function): The 3D solution function.
            **kwargs: Additional keyword arguments.
        """
        self.uh3d = uh3d
        super().__init__(**kwargs)
    
    def eval(self, value: np.ndarray, x: np.ndarray):
        """
        Evaluates the 3D solution at a given point.

        Args:
            value (np.ndarray): The array to store the evaluated value.
            x (np.ndarray): The coordinates of the point.
        """
        value[0] = self.uh3d(x)
    
    def value_shape(self) -> tuple:
        """
        Returns the shape of the evaluated value.

        Returns:
            tuple: The shape of the evaluated value.
        """
        return ()

class RadiusFunction(UserExpression):
    """
    A user expression to compute the radius at a given point based on the nearest control point in a graph.

    Args:
        G (graphnics.FenicsGraph): The graph containing control points with radius information.
        G_mf (dolfin.MeshFunction): A mesh function associated with the graph.
        G_kdt (scipy.spatial.cKDTree): A k-d tree for efficient nearest neighbor search in the graph.
        **kwargs: Additional keyword arguments to be passed to the UserExpression constructor.
    """
    def __init__(self, G: "FenicsGraph", G_mf: MeshFunction, G_kdt: scipy.spatial.cKDTree, **kwargs):
        """
        Initializes the RadiusFunction class.

        Args:
            G (graphnics.FenicsGraph): The graph containing control points with radius information.
            G_mf (dolfin.MeshFunction): A mesh function associated with the graph.
            G_kdt (scipy.spatial.cKDTree): A k-d tree for efficient nearest neighbor search in the graph.
            **kwargs: Additional keyword arguments.
        """
        self.G = G
        self.G_mf = G_mf
        self.G_kdt = G_kdt
        super().__init__(**kwargs)

    def eval(self, value: np.ndarray, x: np.ndarray):
        """
        Computes the radius at a given point.

        Args:
            value (np.ndarray): The array to store the computed radius.
            x (np.ndarray): The coordinates of the point.
        """
        p = (x[0], x[1], x[2])
        _, nearest_control_point_index = self.G_kdt.query(p)
        nearest_control_point = list(self.G.nodes)[nearest_control_point_index]
        value[0] = self.G.nodes[nearest_control_point]['radius']

    def value_shape(self) -> tuple:
        """
        Returns the shape of the computed radius.

        Returns:
            tuple: The shape of the computed radius.
        """
        return ()