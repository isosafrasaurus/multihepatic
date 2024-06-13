# @title pv_1 robin coupling in 3D with lagrange

# Create 3D cube mesh
mesh3d = UnitCubeMesh(32, 32, 32)

# Fit 3D mesh around 1D mesh, guaranteeing minimum size and positive coordinates
c = mesh3d.coordinates()
xl, yl, zl = (np.max(node_coords, axis=0) - np.min(node_coords, axis=0))  # Graph length scales

# Calculate scaling factors, ensuring none go below 10
scaling_factors = np.maximum([xl, yl, zl], 10) + 10 # Add 10 for extra space

# Scale and position the cube mesh
c[:, :] *= scaling_factors
c[:, :] += offset  # Align with the adjusted 1D mesh coordinates

# set constants for simulation
Alpha1 = Constant(1)
alpha1 = Constant(1)
beta = Constant(1.0e3)
gamma = Constant(1.0)  # Adjust gamma
p_infty = Constant(1.3e3)  # Far-field pressure, medical literature suggests PV pressure is 5-10 mmHg

# set boundary conditions for simulation
bc_3d = Constant(3)

# define function that returns lateral faces of 3D boundary
def boundary_3d(x, on_boundary):
    return on_boundary and not near(x[2], 0) and not near(x[2], zl)

class FirstPointBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

boundary_markers = MeshFunction("size_t", mesh1d, mesh1d.topology().dim())
boundary_markers.set_all(0)
FirstPointBoundary().mark(boundary_markers, 1)

# pressure space on global mesh
V3 = FunctionSpace(mesh3d, "CG", 1)
V1 = FunctionSpace(mesh1d, "CG", 1)
W = [V3, V1]

u3, u1 = list(map(TrialFunction, W))
v3, v1 = list(map(TestFunction, W))

# create a radius function for the averaging surface
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

# Prepare radius and position maps
radii = df_points['Radius'].to_dict()
pos = df_points[['x', 'y', 'z']].to_dict(orient='index')

radius_function = RadiusFunction(radii, pos)
cylinder = Circle(radius=radius_function, degree=10)

Pi_u = Average(u3, mesh1d, cylinder)
Pi_v = Average(v3, mesh1d, cylinder)

dxGamma = Measure("dx", domain=mesh1d)
ds = Measure("ds", domain=mesh1d, subdomain_data=boundary_markers)

# Define the variational problem
a00 = Alpha1 * inner(grad(u3), grad(v3)) * dx + beta * inner(Pi_u, Pi_v) * dxGamma
a01 = -beta * inner(u1, Pi_v) * dxGamma
a10 = -beta * inner(Pi_u, v1) * dxGamma
a11 = alpha1 * inner(grad(u1), grad(v1)) * dx + beta * inner(u1, v1) * dx - gamma * inner(u1, v1) * ds(1)

# Right-hand side
L0 = inner(Constant(0), Pi_v) * dxGamma
L1 = -gamma * inner(p_infty, v1) * ds(1)

# Assemble system
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
