from dolfin import *
from vtk.util.numpy_support import vtk_to_numpy
from xii import *
from graphnics import *
from scipy.spatial import cKDTree

import vtk
import pyvista as pv
import meshio

def run_perfusion(G, directory_path, del_Omega=3.0, perf3=9.6e-2, perf1=1.45e4, kappa=3.09e-5, gamma=1.0, P_infty=1.0e3, E=[]):
    def boundary_3d(x, on_boundary):
        return on_boundary and not near(x[2], 0) and not near(x[2], zl)

    class RadiusFunction(UserExpression):
        def __init__(self, G, mf, **kwargs):
            self.G = G
            self.mf = mf
            super().__init__(**kwargs)

        def eval(self, value, x):
            p = Point(x[0], x[1], x[2])
            tree = BoundingBoxTree()
            tree.build(Lambda)
            cell = tree.compute_first_entity_collision(p)

            edge_ix = self.mf[cell]
            edge = list(G.edges())[edge_ix]
            value[0] = self.G.nodes()[edge[0]]['radius']

        def value_shape(self):
            return ()

    # Create \Lambda
    G.make_mesh()
    Lambda, mf = G.get_mesh()

    # Create \Omega
    Omega = UnitCubeMesh(16, 16, 16)

    pos = nx.get_node_attributes(G, "pos")
    node_coords = np.asarray(list(pos.values()))
    xmin, ymin, zmin = np.min(node_coords, axis = 0)

    d = Lambda.coordinates()
    d[:, :] += [-xmin, -ymin, -zmin]

    c = Omega.coordinates()
    xl, yl, zl = (np.max(node_coords, axis=0)-np.min(node_coords, axis=0))
    c[:,:] *= [xl+3, yl+3, zl]
    
    # Constants
    kappa = Constant(kappa)
    gamma = Constant(gamma)
    P_infty = Constant(P_infty)
    del_Omega = Constant(del_Omega)

    # Partitions E and B \subset \Lambda
    subdomains_lambda = MeshFunction("size_t", Lambda, Lambda.topology().dim(), 0)
    for index in E:
        subdomains_lambda[index] = 1
    B = [i for i in range(Lambda.num_entities(0)) if i not in E]
    for index in B:
        subdomains_lambda[index] = 2

    # Function spaces
    V3 = FunctionSpace(Omega, "CG", 1)
    V1 = FunctionSpace(Lambda, "CG", 1)
    W = [V3, V1]
    u3, u1 = list(map(TrialFunction, W))
    v3, v1 = list(map(TestFunction, W))
    
    # Precompute radius values at all mesh nodes
    radius_function = RadiusFunction(G, mf)
    
    total_points = len(Lambda.coordinates())
    radius_values = []
    update_interval = max(1, total_points // 10)  # Adjust interval as needed

    for i, point in enumerate(Lambda.coordinates()):
        radius_values.append(radius_function(point))

        if (i + 1) % update_interval == 0 or (i + 1) == total_points:
            progress = (i + 1) / total_points * 100
            print(f"Computed radius for {i + 1} out of {total_points} points ({progress:.1f}%)")

    radius_values = np.array(radius_values)

    # radius_function = RadiusFunction(G, mf)
    cylinder = Circle(radius=radius_function, degree=5)
    u3_avg = Average(u3, Lambda, cylinder)
    v3_avg = Average(v3, Lambda, cylinder)

    # Dirac measures
    dxOmega = Measure("dx", domain=Omega)
    dxLambda = Measure("dx", domain=Lambda)
    dsLambda = Measure("ds", domain=Lambda, subdomain_data=subdomains_lambda)
    
    class DAreaFunction(UserExpression):
        def __init__(self, radius_values, mf, **kwargs):
            self.radius_values = radius_values
            self.mf = mf
            super().__init__(**kwargs)

        def eval(self, value, x):
            p = Point(x[0], x[1], x[2])
            tree = BoundingBoxTree()
            tree.build(Lambda)
            cell = tree.compute_first_entity_collision(p)
            value[0] = np.pi * self.radius_values[self.mf[cell]] ** 2

        def value_shape(self):
            return ()

    class DPerimeterFunction(UserExpression):
        def __init__(self, radius_values, mf, **kwargs):
            self.radius_values = radius_values
            self.mf = mf
            super().__init__(**kwargs)

        def eval(self, value, x):
            p = Point(x[0], x[1], x[2])
            tree = BoundingBoxTree()
            tree.build(Lambda)
            cell = tree.compute_first_entity_collision(p)
            value[0] = 2 * np.pi * self.radius_values[self.mf[cell]]

        def value_shape(self):
            return ()

    # Instantiate D_area and D_perimeter objects
    D_area = DAreaFunction(radius_values, mf)
    D_perimeter = DPerimeterFunction(radius_values, mf)
    
    # Blocks
    a00 = perf3 * inner(grad(u3), grad(v3)) * dx + kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda
    a01 = -kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
    a10 = -kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
    a11 = perf1 * inner(grad(u1), grad(v1)) * D_area * dxLambda + kappa * inner(u1, v1) * D_perimeter * dxLambda - gamma * inner(u1, v1) * dsLambda(1)
    
    # Right-hand side
    L0 = inner(Constant(0), v3_avg) * dxLambda
    L1 = -gamma * inner(P_infty, v1) * dsLambda(1)

    a = [[a00, a01], [a10, a11]]
    L = [L0, L1]

    W_bcs = [[DirichletBC(V3, del_Omega, boundary_3d)], []]

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, W_bcs)
    A, b = map(ii_convert, (A, b))

    wh = ii_Function(W)
    solver = LUSolver(A, "mumps")
    solver.solve(wh.vector(), b)
    uh3d, uh1d = wh
    uh3d.rename("3D Pressure", "3D Pressure Distribution")
    uh1d.rename("1D Pressure", "1D Pressure Distribution")

    # Create output directory if it doesn't exist and save
    os.makedirs(directory_path, exist_ok=True)
    output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
    output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
    save_mesh_as_vtk(Lambda, output_file_1d, radius_function, uh1d=uh1d)
    File(output_file_3d) << uh3d
    
    return output_file_1d, output_file_3d, uh1d, uh3d

def save_mesh_as_vtk(Lambda, file_path, radius_function, uh1d=None):
    points = Lambda.coordinates()
    cells = {"line": Lambda.cells()}
    
    # Evaluate radius function at each node in the mesh
    radius_values = np.array([radius_function(point) for point in points])

    # Evaluate uh1d function at each node in the mesh
    uh1d_values = np.array([uh1d(point) for point in points])
    
    if uh1d != None:
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