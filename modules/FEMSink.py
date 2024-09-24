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
    """
    A class for solving the primal finite element method (FEM) system on a 3D domain containing a 1D network,
    incorporating source/sink terms via Robin boundary conditions.

    This class sets up and solves a steady-state diffusion-reaction problem with different properties in the 3D domain
    and the 1D network, including a pressure source/sink on one face of the 3D domain.

    Attributes:
        G (FenicsGraph): The FenicsGraph object representing the 1D network.
        kappa (float): Coupling coefficient between 3D and 1D domains.
        alpha (float): Diffusion coefficient in the 3D domain.
        beta (float): Diffusion coefficient in the 1D network.
        gamma (float): Reaction coefficient in the 1D network.
        P_infty (float): Source term value in the 1D network.
        theta (float): Coupling coefficient for the Robin boundary condition on Omega.
        P_sink (float): Pressure source value on the Robin boundary.
        Lambda (dolfin.Mesh): The mesh representing the 1D network.
        Omega (dolfin.Mesh): The mesh representing the 3D domain.
        G_rf (dolfin.UserExpression): A function to compute radius values at each point in the 1D network.
        uh3d (dolfin.Function): The solution for 3D pressure.
        uh1d (dolfin.Function): The solution for 1D pressure.
    """

    # Constructor
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
        """
        Initializes the FEMPrimalWithSource class with given parameters and solves the primal system.

        Args:
            G (FenicsGraph): The FenicsGraph object representing the 1D network.
            kappa (float, optional): Coupling coefficient between 3D and 1D domains.
            alpha (float, optional): Diffusion coefficient in the 3D domain.
            beta (float, optional): Diffusion coefficient in the 1D network.
            gamma (float, optional): Reaction coefficient in the 1D network.
            P_infty (float, optional): Source term value in the 1D network.
            theta (float, optional): Coupling coefficient for the Robin boundary condition on Omega.
            P_sink (float, optional): Pressure source value on the Robin boundary.
            Omega_bbox (tuple, optional): Bounding box scaling factors for the Omega mesh.
        """
        
        # Convert parameters to FEniCS Constants
        self.kappa = Constant(kappa)
        self.alpha = Constant(alpha)
        self.beta = Constant(beta)
        self.gamma = Constant(gamma)
        self.P_infty = Constant(P_infty)
        self.theta = Constant(theta)
        self.P_sink = Constant(P_sink)

        # Generate meshes
        G.make_mesh()
        Lambda, G_mf = G.get_mesh()

        # Extract node positions and build KD-Tree for nearest neighbor search
        node_positions = nx.get_node_attributes(G, "pos")
        node_coords = np.asarray(list(node_positions.values()))
        G_kdt = scipy.spatial.cKDTree(node_coords)

        # Fit Omega around Lambda
        Omega = UnitCubeMesh(32, 32, 32)
        Omega_coords = Omega.coordinates()
        xl, yl, zl = (np.max(node_coords, axis=0) - np.min(node_coords, axis=0))
        
        if Omega_bbox is not None:
            Omega_coords[:, :] *= [Omega_bbox[0], Omega_bbox[1], Omega_bbox[2]]
        else:
            Omega_coords[:, :] *= [xl + 3, yl + 3, zl + 3]
        
        # Update the mesh with scaled coordinates
        Omega.bounding_box_tree()

        self.Lambda, self.Omega = Lambda, Omega

        # Define Boundary Marker for Robin Condition (Face 1)
        class BoundaryFace1(SubDomain):
            def inside(self, x, on_boundary):
                # Define Face 1 as the face where x[0] is approximately 0
                return on_boundary and near(x[0], 0.0)

        boundary_markers = MeshFunction("size_t", Omega, Omega.topology().dim()-1, 0)
        boundary_face1 = BoundaryFace1()
        boundary_face1.mark(boundary_markers, 1)
        ds = Measure("ds", domain=Omega, subdomain_data=boundary_markers)

        # Function spaces
        V3 = FunctionSpace(Omega, "CG", 1)
        V1 = FunctionSpace(Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = list(map(TrialFunction, W))
        v3, v1 = list(map(TestFunction, W))

        # Radius function for the 1D network
        G_rf = RadiusFunction(G, G_mf, G_kdt, degree=1)
        self.G_rf = G_rf

        # Define average operators
        cylinder = Circle(radius=G_rf, degree=5)
        u3_avg = Average(u3, Lambda, cylinder)
        v3_avg = Average(v3, Lambda, cylinder)

        # Measures for integrals
        dxOmega = Measure("dx", domain=Omega)
        dxLambda = Measure("dx", domain=Lambda)
        dsLambda = Measure("ds", domain=Lambda)

        # Define D_area and D_perimeter based on circular cross-section
        D_area = np.pi * G_rf**2
        D_perimeter = 2 * np.pi * G_rf

        # Define the bilinear and linear forms for the primal system
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
        """
        Saves the computed solutions (pressure) to VTK files in the specified directory.

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
            uh1d_values = uh1d.compute_vertex_values(Lambda)
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
        self.uh3d = uh3d
        super().__init__(**kwargs)
    
    def eval(self, value: np.ndarray, x: np.ndarray):
        value[0] = self.uh3d(x)
    
    def value_shape(self) -> tuple:
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
