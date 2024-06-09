# Recenter 1D coordinates to all be >= 0
pos = nx.get_node_attributes(G, "pos")
node_coords = np.asarray(list(pos.values()))
xmin, ymin, zmin = np.min(node_coords, axis=0)
d = mesh1d.coordinates()
d[:, :] += [-xmin, -ymin, -zmin]

# Function space
V1 = FunctionSpace(mesh1d, "CG", 1)
u1 = TrialFunction(V1)
v1 = TestFunction(V1)

# Constants for simulation
alpha1 = Constant(1.45e4)
beta = Constant(3.09e-5)
gamma = Constant(0.7)  # Adjust this value as needed
p_inf = Constant(5.0)  # Far-field pressure

# Measure for 1D mesh
dxGamma = Measure("dx", domain=mesh1d)

# Define Robin boundary conditions
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0)

def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 1)  # Assuming normalized length

bc_robin_left = gamma * p_inf * v1 * ds(subdomain_id=1)
bc_robin_right = gamma * p_inf * v1 * ds(subdomain_id=2)

# Assemble system with Robin boundary conditions
a11 = alpha1 * inner(grad(u1), grad(v1)) * dx + beta * inner(u1, v1) * dxGamma + gamma * v1 * u1 * (ds(subdomain_id=1) + ds(subdomain_id=2))
L1 = beta * inner(Constant(0), v1) * dxGamma + bc_robin_left + bc_robin_right

# Assemble system
A, b = assemble_system(a11, L1)

# Solve system
uh1d = Function(V1)
solve(A, uh1d.vector(), b)

# Save solution
File(WD_PATH + 'plots/pv1_mod/pressure1d.pvd') << uh1d