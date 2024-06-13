# @title Robin coupling with 3D, universal boundaries

# Create \Omega
mesh3d = UnitCubeMesh(32, 32, 32)
c = mesh3d.coordinates()
c[:, :] *= [xl + 3, yl + 3, zl]

# Create \Lambda
G.make_mesh()
mesh1d, mf = G.get_mesh()

# create a radius function for the averaging surface
class RadiusFunction(UserExpression):
    def __init__(self, G, mf, **kwargs):
        self.G = G
        self.mf = mf
        super().__init__(**kwargs)

    def eval(self, value, x):
        p = Point(x[0], x[1], x[2])
        tree = BoundingBoxTree()
        tree.build(mesh1d)
        cell = tree.compute_first_entity_collision(p)

        edge_ix = self.mf[cell]
        edge = list(G.edges())[edge_ix]
        value[0] = self.G.nodes()[edge[0]]['radius']

    def value_shape(self):
        return ()

# Create \Lambda
G.make_mesh()
mesh1d, mf = G.get_mesh()

Alpha1 = Constant(9.6e-2)
alpha1 = Constant(1.45e4)
beta = Constant(3.09e-5)
gamma = Constant(1.0)  # Adjust gamma
p_infty = Constant(1.3e3)  # Far-field pressure, medical literature suggests PV pressure is 5-10 mmHg

# set boundary conditions for simulation
bc_3d = Constant(3)

# define function that returns lateral faces of 3D boundary
def boundary_3d(x, on_boundary):
    return on_boundary and not near(x[2], 0) and not near(x[2], zl)

# pressure space on global mesh
V3 = FunctionSpace(mesh3d, "CG", 1)
V1 = FunctionSpace(mesh1d, "CG", 1)
W = [V3, V1]

u3, u1 = list(map(TrialFunction, W))
v3, v1 = list(map(TestFunction, W))

radius_function = RadiusFunction(G, mf)
cylinder = Circle(radius=radius_function, degree=10)

Pi_u = Average(u3, mesh1d, cylinder)
Pi_v = Average(v3, mesh1d, cylinder)

# Dirac measure
dxGamma = Measure("dx", domain=mesh1d)
ds1 = Measure("ds", domain=mesh1d)

# blocks
a00 = Alpha1 * inner(grad(u3), grad(v3)) * dx + beta * inner(Pi_u, Pi_v) * dxGamma
a01 = -beta * inner(u1, Pi_v) * dxGamma
a10 = -beta * inner(Pi_u, v1) * dxGamma
a11 = alpha1 * inner(grad(u1), grad(v1)) * dx + beta * inner(u1, v1) * dx - gamma * inner(u1, v1) * ds1

# right-hand side
L0 = inner(Constant(0), Pi_v) * dxGamma
L1 = -gamma * inner(p_infty, v1) * ds1

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
File(WD_PATH + '/plots/pv_epsilongaptest/pressure1d.pvd') << uh1d
File(WD_PATH + '/plots/pv_epsilongaptest/pressure3d.pvd') << uh3d