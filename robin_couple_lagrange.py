


mesh3d = UnitCubeMesh(32, 32, 32)


c = mesh3d.coordinates()
xl, yl, zl = (np.max(node_coords, axis=0) - np.min(node_coords, axis=0))  


scaling_factors = np.maximum([xl, yl, zl], 10) + 10 


c[:, :] *= scaling_factors
c[:, :] += offset  


Alpha1 = Constant(1)
alpha1 = Constant(1)
beta = Constant(1.0e3)
gamma = Constant(1.0)  
p_infty = Constant(1.3e3)  


bc_3d = Constant(3)


def boundary_3d(x, on_boundary):
    return on_boundary and not near(x[2], 0) and not near(x[2], zl)

class FirstPointBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

boundary_markers = MeshFunction("size_t", mesh1d, mesh1d.topology().dim())
boundary_markers.set_all(0)
FirstPointBoundary().mark(boundary_markers, 1)


V3 = FunctionSpace(mesh3d, "CG", 1)
V1 = FunctionSpace(mesh1d, "CG", 1)
W = [V3, V1]

u3, u1 = list(map(TrialFunction, W))
v3, v1 = list(map(TestFunction, W))


class RadiusFunction(UserExpression):
    def __init__(self, radii_map, pos_map, **kwargs):
        self.radii_map = radii_map
        self.pos_map = pos_map
        super().__init__(**kwargs)

    def eval(self, value, x):
        min_dist = float('inf')
        closest_radius = 0
        for node, position in self.pos_map.items():
            posi = np.array(list(position.values()))
            dist = np.linalg.norm(x - posi)
            if dist < min_dist:
                min_dist = dist
                closest_radius = self.radii_map[node]
        value[0] = closest_radius

    def value_shape(self):
        return ()


radii = df_points['Radius'].to_dict()
pos = df_points[['x', 'y', 'z']].to_dict(orient='index')

radius_function = RadiusFunction(radii, pos)
cylinder = Circle(radius=radius_function, degree=10)

Pi_u = Average(u3, mesh1d, cylinder)
Pi_v = Average(v3, mesh1d, cylinder)

dxGamma = Measure("dx", domain=mesh1d)
ds = Measure("ds", domain=mesh1d, subdomain_data=boundary_markers)


a00 = Alpha1 * inner(grad(u3), grad(v3)) * dx + beta * inner(Pi_u, Pi_v) * dxGamma
a01 = -beta * inner(u1, Pi_v) * dxGamma
a10 = -beta * inner(Pi_u, v1) * dxGamma
a11 = alpha1 * inner(grad(u1), grad(v1)) * dx + beta * inner(u1, v1) * dx - gamma * inner(u1, v1) * ds(1)


L0 = inner(Constant(0), Pi_v) * dxGamma
L1 = -gamma * inner(p_infty, v1) * ds(1)


a = [[a00, a01], [a10, a11]]
L = [L0, L1]

W_bcs = [[DirichletBC(V3, bc_3d, boundary_3d)], []]

A, b = map(ii_assemble, (a, L))
A, b = apply_bc(A, b, W_bcs)
A, b = map(ii_convert, (A, b))

wh = ii_Function(W)
solver = LUSolver(A, "mumps")
solver.solve(wh.vector(), b)

uh3d, uh1d = wh
File(WD_PATH + '/plots/pv_lagrangetest/pressure1d.pvd') << uh1d
File(WD_PATH + '/plots/pv_lagrangetest/pressure3d.pvd') << uh3d
visualize_scatter(mesh1d, uh1d, z_level=20)
