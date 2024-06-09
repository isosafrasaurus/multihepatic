def boundary_3d(x, on_boundary):
    return on_boundary and not near(x[2], 0) and not near(x[2], zl)

class RadiusFunction(UserExpression):
    def __init__(self, G, **kwargs):
        self.G = G
        _, self.mf = self.G.get_mesh()
        super().__init__(**kwargs)

    def eval(self, value, x):
        p = Point(x[0], x[1], x[2])
        tree = BoundingBoxTree()
        tree.build(mesh1d)
        cell = tree.compute_first_entity_collision(p)

        edge_ix = self.mf[cell]
        edge = list(G.edges())[edge_ix]
        value[0] = self.G.nodes()[edge[0]]['Radius']

    def value_shape(self):
        return ()

# Constants
kappa = Constant(3.09e-5)
gamma = Constant(1.0)
P_infty = Constant(1.3e3)

# Boundary on \Omega_{\oplus}
bc_3d = Constant(3)

# Boundary partitions E and B on \Lambda
E = [0]
subdomains_lambda = MeshFunction("size_t", mesh1d, mesh1d.topology().dim(), 0)
for index in E:
    subdomains_lambda[index] = 1
B = [i for i in range(mesh1d.num_entities(0)) if i not in E]
for index in B:
    subdomains_lambda[index] = 2

V3 = FunctionSpace(mesh3d, "CG", 1)
V1 = FunctionSpace(mesh1d, "CG", 1)
W = [V3, V1]
u3, u1 = list(map(TrialFunction, W))
v3, v1 = list(map(TestFunction, W))

radius_function = RadiusFunction(G)
cylinder = Circle(radius=radius_function, degree=10)

u3_avg = Average(u3, mesh1d, cylinder)
v3_avg = Average(v3, mesh1d, cylinder)

# Dirac measure
dxLambda = Measure("dx", domain=mesh1d)
dsLambda = Measure("ds", domain=mesh1d, subdomain_data=subdomains_lambda)

# Blocks
a00 =  inner(grad(u3), grad(v3)) * dx + kappa * inner(u3_avg, v3_avg) * dxLambda
a01 = -kappa * inner(u1, v3_avg) * dxLambda
a10 = -kappa * inner(u3_avg, v1) * dxLambda
a11 =  inner(grad(u1), grad(v1)) * dx + kappa * inner(u1, v1) * dx - gamma * inner(u1, v1) * dsLambda(1)

# Right-hand side
L0 = inner(Constant(0), v3_avg) * dxLambda
L1 = -gamma * inner(P_infty, v1) * dsLambda(1)

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
uh3d.rename("3D Pressure", "3D Pressure Distribution")
uh1d.rename("1D Pressure", "1D Pressure Distribution")