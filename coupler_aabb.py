import vtk
import pyvista as pv
import meshio
import numpy as np
from dolfin import *
from vtk.util.numpy_support import vtk_to_numpy
from xii import *
from graphnics import *
from scipy.spatial import cKDTree

class AABB:
    def __init__(self, min_corner, max_corner, edge=None):
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)
        self.edge = edge 

    def intersects(self, point):
        return np.all(point >= self.min_corner) and np.all(point <= self.max_corner)

    def get_center(self):
        return (self.min_corner + self.max_corner) / 2

    def get_extents(self):
        return self.max_corner - self.min_corner

class AABBTree:
    def __init__(self):
        self.root = None

    def build(self, aabbs):
        if len(aabbs) == 1:
            return aabbs[0]
        elif len(aabbs) == 0:
            return None

        
        min_corner = np.min([aabb.min_corner for aabb in aabbs], axis=0)
        max_corner = np.max([aabb.max_corner for aabb in aabbs], axis=0)
        parent_aabb = AABB(min_corner, max_corner)

        
        extents = parent_aabb.get_extents()
        longest_axis = np.argmax(extents)
        sorted_aabbs = sorted(aabbs, key=lambda aabb: aabb.get_center()[longest_axis])

        
        mid = len(sorted_aabbs) // 2
        parent_aabb.left = self.build(sorted_aabbs[:mid])
        parent_aabb.right = self.build(sorted_aabbs[mid:])

        return parent_aabb

    def compute_first_entity_collision(self, point):
        return self._compute_first_entity_collision(self.root, np.array(point))

    def _compute_first_entity_collision(self, node, point):
        if node is None:
            return None, float('inf')

        if not node.intersects(point):
            return None, float('inf')

        if node.edge is not None:
            
            u, v = node.edge
            dist = self._distance_point_to_segment(point, u['pos'], v['pos'])
            return node.edge, dist

        left_collision, left_dist = self._compute_first_entity_collision(node.left, point)
        right_collision, right_dist = self._compute_first_entity_collision(node.right, point)

        if left_dist < right_dist:
            return left_collision, left_dist
        else:
            return right_collision, right_dist

    def _distance_point_to_segment(self, p, a, b):
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest_point = a + t * ab
        return np.linalg.norm(closest_point - p)

class RadiusFunction(UserExpression):
    
    def __init__(self, G, mf, tree, **kwargs):
        self.G = G
        self.mf = mf
        self.tree = tree
        super().__init__(**kwargs)
    
    def eval(self, value, x):
        p = (x[0], x[1], x[2])
        nearest_edge, _ = self.tree.compute_first_entity_collision(p)
        value[0] = nearest_edge['radius']
    
    def value_shape(self):
        return ()
        
def save_mesh_as_vtk(Lambda, file_path, radius_function, uh1d=None):
    
    points = Lambda.coordinates()
    cells = {"line": Lambda.cells()}
    
    
    radius_values = np.array([radius_function(point) for point in points])

    
    uh1d_values = np.array([uh1d(point) for point in points])
    
    if uh1d != None:
        mesh = meshio.Mesh(points, cells, point_data={"radius": radius_values, "1D_Pressure": uh1d_values})
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
        
def run_perfusion_univ(G, directory_path, del_Omega=3.0, perf3=9.6e-2, perf1=1.45e4, kappa=3.09e-5, gamma=1.0, P_infty=1.0e3):        
        
    
    G.make_mesh()
    Lambda, mf = G.get_mesh()
    
    
    H = G.copy()
    
    
    Omega = UnitCubeMesh(16, 16, 16)

    
    pos = nx.get_node_attributes(G, "pos")
    node_coords = np.asarray(list(pos.values()))
    xmin, ymin, zmin = np.min(node_coords, axis = 0)
    d = Lambda.coordinates()
    d[:, :] += [-xmin, -ymin, -zmin]
    for node in H.nodes:
        H.nodes[node]['pos'] = np.array(H.nodes[node]['pos']) + [-xmin, -ymin, -zmin]
    
    
    c = Omega.coordinates()
    xl, yl, zl = (np.max(node_coords, axis=0)-np.min(node_coords, axis=0))
    c[:,:] *= [xl+3, yl+3, zl]
    
    def boundary_Omega(x, on_boundary):
        return on_boundary and not near(x[2], 0) and not near(x[2], zl)
        
    
    aabbs = []
    for u, v, data in G.edges(data=True):
        u_pos = G.nodes[u]['pos']
        v_pos = G.nodes[v]['pos']
        min_corner = np.minimum(u_pos, v_pos)
        max_corner = np.maximum(u_pos, v_pos)
        aabbs.append(AABB(min_corner, max_corner, edge=(G.nodes[u], G.nodes[v])))

    
    tree = AABBTree()
    tree.root = tree.build(aabbs)
    print(tree.root)
        
    
    kappa = Constant(kappa)
    gamma = Constant(gamma)
    P_infty = Constant(P_infty)
    del_Omega = Constant(del_Omega)

    
    V3 = FunctionSpace(Omega, "CG", 1)
    V1 = FunctionSpace(Lambda, "CG", 1)
    W = [V3, V1]
    u3, u1 = list(map(TrialFunction, W))
    v3, v1 = list(map(TestFunction, W))
    
    radius_function = RadiusFunction(G, mf, tree)
    cylinder = Circle(radius=radius_function, degree=5)
    u3_avg = Average(u3, Lambda, cylinder)
    v3_avg = Average(v3, Lambda, cylinder)

    
    dxOmega = Measure("dx", domain=Omega)
    dxLambda = Measure("dx", domain=Lambda)
    dsLambda = Measure("ds", domain=Lambda)
    
    
    D_area = np.pi * radius_function ** 2
    D_perimeter = 2 * np.pi * radius_function
    
    
    a00 = perf3 * inner(grad(u3), grad(v3)) * dx + kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda
    a01 = -kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
    a10 = -kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
    a11 = perf1 * inner(grad(u1), grad(v1)) * D_area * dxLambda + kappa * inner(u1, v1) * D_perimeter * dxLambda - gamma * inner(u1, v1) * dsLambda
    
    
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

    
    os.makedirs(directory_path, exist_ok=True)
    output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
    output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
    save_mesh_as_vtk(Lambda, output_file_1d, radius_function, uh1d=uh1d)
    File(output_file_3d) << uh3d
    
    return output_file_1d, output_file_3d, uh1d, uh3d

def run_perfusion(G, directory_path, del_Omega=3.0, perf3=9.6e-2, perf1=1.45e4, kappa=3.09e-5, gamma=1.0, P_infty=1.0e3, E=[]):
        
    
    G.make_mesh()
    Lambda, mf = G.get_mesh()
    
    
    H = G.copy()
    
    
    Omega = UnitCubeMesh(16, 16, 16)

    
    pos = nx.get_node_attributes(G, "pos")
    node_coords = np.asarray(list(pos.values()))
    xmin, ymin, zmin = np.min(node_coords, axis = 0)
    d = Lambda.coordinates()
    d[:, :] += [-xmin, -ymin, -zmin]
    for node in H.nodes:
        H.nodes[node]['pos'] = np.array(H.nodes[node]['pos']) + [-xmin, -ymin, -zmin]
    
    
    kdtree = cKDTree(np.array(list(nx.get_node_attributes(H, 'pos').values())))

    
    c = Omega.coordinates()
    xl, yl, zl = (np.max(node_coords, axis=0)-np.min(node_coords, axis=0))
    c[:,:] *= [xl+3, yl+3, zl]
    
    
    subdomains_lambda = MeshFunction("size_t", Lambda, Lambda.topology().dim(), 0)
    for index in E:
        subdomains_lambda[index] = 1
    B = [i for i in range(Lambda.num_entities(0)) if i not in E]
    for index in B:
        subdomains_lambda[index] = 2
        
    
    kappa = Constant(kappa)
    gamma = Constant(gamma)
    P_infty = Constant(P_infty)
    del_Omega = Constant(del_Omega)

    
    V3 = FunctionSpace(Omega, "CG", 1)
    V1 = FunctionSpace(Lambda, "CG", 1)
    W = [V3, V1]
    u3, u1 = list(map(TrialFunction, W))
    v3, v1 = list(map(TestFunction, W))
    
    radius_function = RadiusFunction(G, mf)
    cylinder = Circle(radius=radius_function, degree=5)
    u3_avg = Average(u3, Lambda, cylinder)
    v3_avg = Average(v3, Lambda, cylinder)

    
    dxOmega = Measure("dx", domain=Omega)
    dxLambda = Measure("dx", domain=Lambda)
    dsLambda = Measure("ds", domain=Lambda)
    
    
    D_area = np.pi * radius_function ** 2
    D_perimeter = 2 * np.pi * radius_function
    
    
    a00 = perf3 * inner(grad(u3), grad(v3)) * dx + kappa * inner(u3_avg, v3_avg) * D_perimeter * dxLambda
    a01 = -kappa * inner(u1, v3_avg) * D_perimeter * dxLambda
    a10 = -kappa * inner(u3_avg, v1) * D_perimeter * dxLambda
    a11 = perf1 * inner(grad(u1), grad(v1)) * D_area * dxLambda + kappa * inner(u1, v1) * D_perimeter * dxLambda - gamma * inner(u1, v1) * dsLambda(1)
    
    
    L0 = inner(Constant(0), v3_avg) * dxLambda
    L1 = -gamma * inner(P_infty, v1) * dsLambda(1)

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

    
    os.makedirs(directory_path, exist_ok=True)
    output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
    output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
    save_mesh_as_vtk(Lambda, output_file_1d, radius_function, uh1d=uh1d)
    File(output_file_3d) << uh3d
    
    return output_file_1d, output_file_3d, uh1d, uh3d