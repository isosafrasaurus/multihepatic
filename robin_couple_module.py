from dolfin import *
from vtk.util.numpy_support import vtk_to_numpy
from xii import *
from graphnics import *
import vtk
import pyvista as pv

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
    
    # Retroactively set 'pos' attributes in G
    for node in G.nodes():
        G.nodes[node]['pos'] = d[mf[node]].tolist()

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

    radius_function = RadiusFunction(G, mf)
    cylinder = Circle(radius=radius_function, degree=10)
    u3_avg = Average(u3, Lambda, cylinder)
    v3_avg = Average(v3, Lambda, cylinder)

    # Dirac measures
    dxOmega = Measure("dx", domain=Omega)
    dxLambda = Measure("dx", domain=Lambda)
    dsLambda = Measure("ds", domain=Lambda, subdomain_data=subdomains_lambda)
    
    D_area = np.pi * radius_function ** 2
    D_perimeter = np.pi * radius_function * 2
    
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
    output_file_1d = os.path.join(directory_path, "pressure1d.pvd")
    output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
    File(output_file_1d) << uh1d
    File(output_file_3d) << uh3d
    
    test_path = os.path.join(directory_path, "mygraph.vtk")
    save_graph_as_vtk(G, test_path)
        
    return output_file_1d, output_file_3d, uh1d, uh3d, test_path

def save_graph_as_vtk(G, filename):
    # Create a vtkPoints object to hold the node positions
    points = vtk.vtkPoints()
    
    # Create a vtkFloatArray to hold the radii of the nodes
    radii = vtk.vtkFloatArray()
    radii.SetName("radius")
    
    # Create a vtkCellArray to hold the line segments (edges)
    lines = vtk.vtkCellArray()
    
    # Add the positions and radii of each node
    node_to_id = {}
    for i, (node, data) in enumerate(G.nodes(data=True)):
        pos = data["pos"]
        radius = data["radius"]
        
        point_id = points.InsertNextPoint(pos)
        node_to_id[node] = point_id
        
        radii.InsertNextValue(radius)
    
    # Add the edges to the cell array
    for edge in G.edges():
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, node_to_id[edge[0]])
        line.GetPointIds().SetId(1, node_to_id[edge[1]])
        lines.InsertNextCell(line)
    
    # Create a vtkPolyData object to hold the geometry and topology
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    
    # Add the radius data to the polydata
    polydata.GetPointData().AddArray(radii)
    
    # Write the polydata to a .vtk file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()