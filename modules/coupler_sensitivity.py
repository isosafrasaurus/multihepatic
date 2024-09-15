import vtk
import meshio
import os
import numpy as np
import networkx as nx
from dolfin import *
from xii import *
from graphnics import *
from scipy.spatial import cKDTree

def save_mesh_vtk(Lambda, file_path, radius_map_G, uh1d=None):
    """
    Saves a tube mesh as a VTK file with the option to include the 1D solution as a data array.

    Args:
        Lambda (dolfin.Mesh): The mesh to be saved.
        file_path (str): The path where the VTK file will be saved.
        radius_map_G (dolfin.UserExpression): A function to compute radius values at each point.
        uh1d (dolfin.Function, optional): A function representing 1D pressure data.

    Returns:
        None
    """
    points = Lambda.coordinates()
    cells = {"line": Lambda.cells()}

    # Evaluate radius function at each node in the mesh
    radius_values = np.array([radius_map_G(point) for point in points])

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

    # Convert unstructured grid to polydata
    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(reader.GetOutput())
    geometryFilter.Update()

    polydata = geometryFilter.GetOutput()

    # Write polydata to a new VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()

class radius_function(UserExpression):
    """
    A user expression to compute the radius at a given point based on the nearest control point in a graph.

    Args:
        G (graphnics.FenicsGraph): The graph containing control points with radius information.
        mf (dolfin.MeshFunction): A mesh function associated with the graph.
        kdtree (scipy.spatial.cKDTree): A k-d tree for efficient nearest neighbor search in the graph.
        **kwargs: Additional keyword arguments to be passed to the UserExpression constructor.
    """
    def __init__(self, G, mf, kdtree, **kwargs):
        self.G = G
        self.mf = mf
        self.kdtree = kdtree
        super().__init__(**kwargs)

    def eval(self, value, x):
        p = (x[0], x[1], x[2])
        _, nearest_control_point_index = self.kdtree.query(p)
        nearest_control_point = list(self.G.nodes)[nearest_control_point_index]
        value[0] = self.G.nodes[nearest_control_point]['radius']

    def value_shape(self):
        return ()

def run_perfusion_univ(G, directory_path, del_Omega=3.0, perf3=9.6e-2, perf1=1.45e4, kappa_value=3.09e-5, gamma=1.0, P_infty=1.0e3):
    """
    Runs a perfusion simulation with Robin far field effects everywhere on a given graph, and solves the sensitivity system.

    Args:
        G (graphnics.FenicsGraph): The graph representing the network.
        directory_path (str): The directory where the results will be saved
        del_Omega (float, optional): Boundary condition value for 3D pressure. Defaults to 3.0
        perf3 (float, optional): 3D perfusion coefficient
        perf1 (float, optional): 1D perfusion coefficient
        kappa_value (float, optional): Coupling coefficient
        gamma (float, optional): Boundary condition coefficient for 1D pressure
        P_infty (float, optional): Boundary condition value for 1D pressure

    Returns:
        tuple: A tuple containing:
            - output_file_1d (str): The path to the saved 1D pressure VTK file
            - output_file_3d (str): The path to the saved 3D pressure PVD file
            - uh1d (dolfin.Function): The computed 1D pressure function
            - uh3d (dolfin.Function): The computed 3D pressure function
            - sh1d (dolfin.Function): The computed 1D sensitivity function
            - sh3d (dolfin.Function): The computed 3D sensitivity function
    """
    # Create \Lambda
    G.make_mesh()
    Lambda, mf = G.get_mesh()

    # Reference copy of \Lambda
    H = G.copy()

    # Create \Omega
    Omega = UnitCubeMesh(16, 16, 16)

    # Translate all \Lambda points to positive, same in H
    pos = nx.get_node_attributes(G, "pos")
    node_coords = np.asarray(list(pos.values()))
    xmin, ymin, zmin = np.min(node_coords, axis=0)
    d = Lambda.coordinates()
    d[:, :] += [-xmin, -ymin, -zmin]
    for node in H.nodes:
        H.nodes[node]['pos'] = np.array(H.nodes[node]['pos']) + [-xmin, -ymin, -zmin]

    # \Lambda k-d tree
    kdtree = cKDTree(np.array(list(nx.get_node_attributes(H, 'pos').values())))

    # Fit \Omega around \Lambda
    c = Omega.coordinates()
    xl, yl, zl = (np.max(node_coords, axis=0) - np.min(node_coords, axis=0))
    c[:, :] *= [xl + 3, yl + 3, zl]

    def boundary_Omega(x, on_boundary):
        return on_boundary and not near(x[2], 0) and not near(x[2], zl)

    # Constants
    kappa = Constant(kappa_value)
    gamma = Constant(gamma)
    P_infty = Constant(P_infty)
    del_Omega = Constant(del_Omega)

    # Function spaces
    V3 = FunctionSpace(Omega, "CG", 1)
    V1 = FunctionSpace(Lambda, "CG", 1)
    W = [V3, V1]
    u3, u1 = list(map(TrialFunction, W))
    v3, v1 = list(map(TestFunction, W))

    radius_map_G = radius_function(G, mf, kdtree)
    cylinder = Circle(radius=radius_map_G, degree=5)
    u3_avg = Average(u3, Lambda, cylinder)
    v3_avg = Average(v3, Lambda, cylinder)

    # Dirac measures
    dxOmega = Measure("dx", domain=Omega)
    dxLambda = Measure("dx", domain=Lambda)
    dsLambda = Measure("ds", domain=Lambda)

    # Define D_area and D_perimeter
    D_area = np.pi * radius_map_G ** 2
    D_perimeter = 2 * np.pi * radius_map_G

    # Blocks for the steady-state problem
    a00 = perf3 * inner(grad(u3), grad(v3)) * dxOmega + kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda
    a01 = -kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
    a10 = -kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
    a11 = perf1 * inner(grad(u1), grad(v1)) * D_area * dxLambda + kappa * inner(u1, v1) * D_perimeter * dxLambda - gamma * inner(u1, v1) * dsLambda

    # Right-hand side for the steady-state problem
    L0 = inner(Constant(0), v3_avg) * dxLambda
    L1 = -gamma * inner(P_infty, v1) * dsLambda

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

    # Now, set up the sensitivity problem
    # Define trial and test functions for sensitivity variables
    s3k = TrialFunction(V3)
    s1k = TrialFunction(V1)
    v3k = TestFunction(V3)
    v1k = TestFunction(V1)

    s3k_avg = Average(s3k, Lambda, cylinder)
    v3k_avg = Average(v3k, Lambda, cylinder)

    # Create an Expression to evaluate uh3d at points on Lambda
    class Uh3dAtLambda(UserExpression):
        def __init__(self, uh3d, **kwargs):
            self.uh3d = uh3d
            super().__init__(**kwargs)
        def eval(self, value, x):
            value[0] = self.uh3d(x)
        def value_shape(self):
            return ()

    u3h_at_Lambda = interpolate(Uh3dAtLambda(uh3d, degree=uh3d.function_space().ufl_element().degree()), V1)

    # Left-hand side for the sensitivity problem
    a00_sens = perf3 * inner(grad(s3k), grad(v3k)) * dxOmega + kappa * inner(s3k_avg, v3k_avg) * D_perimeter * dxLambda
    a01_sens = -kappa * inner(s1k, v3k_avg) * D_perimeter * dxLambda
    a10_sens = -kappa * inner(s3k_avg, v1k) * D_perimeter * dxLambda
    a11_sens = perf1 * inner(grad(s1k), grad(v1k)) * D_area * dxLambda + kappa * inner(s1k, v1k) * D_perimeter * dxLambda

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

    # Create output directory if it doesn't exist and save
    os.makedirs(directory_path, exist_ok=True)
    output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
    output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
    output_file_sens_1d = os.path.join(directory_path, "sensitivity1d.vtk")
    output_file_sens_3d = os.path.join(directory_path, "sensitivity3d.pvd")
    save_mesh_vtk(Lambda, output_file_1d, radius_map_G, uh1d=uh1d)
    save_mesh_vtk(Lambda, output_file_sens_1d, radius_map_G, uh1d=sh1d)
    File(output_file_3d) << uh3d
    File(output_file_sens_3d) << sh3d

    return output_file_1d, output_file_3d, uh1d, uh3d, sh1d, sh3d, Lambda, Omega
