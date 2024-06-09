from fenics import *
import matplotlib.pyplot as plt

# Define parameters
L = 1.0         # Domain length
beta = 1.0      # Coefficient
p_bar_3D = 1.0  # Mean 3D pressure
p_inf = 0.5     # Far-field pressure
gamma = 1.0     # Boundary condition coefficient
f_1D = Constant(0.0)  # Source term

# Create mesh and define function space
mesh = IntervalMesh(50, 0, L)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary nodes
boundary_nodes = [0,L]

# Define custom Robin boundary condition
class RobinBC(UserExpression):
    def __init__(self, gamma, p_infty, boundary_markers, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.p_infty = p_infty
        self.boundary_markers = boundary_markers

    def eval(self, values, x):
        print(f"eval called! self: {self} values: {values} x: {x}")
        if near(x[0], self.boundary_markers[0]) or near(x[0], self.boundary_markers[1]):
            values[0] = self.gamma * self.p_infty
        else:
            values[0] = 0.0

    def value_shape(self):
        return ()

# Instantiate RobinBC
robin_bc = RobinBC(gamma, p_inf, boundary_nodes, degree=0)

# Define variational problem
p = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(p), grad(v))*dx + beta*p*v*dx
L = beta*p_bar_3D*v*dx + f_1D*v*dx

# Apply Robin boundary conditions
n = FacetNormal(mesh)
a += gamma*p*v*ds
L += robin_bc*v*ds

# Solve the problem
p_1D = Function(V)
solve(a == L, p_1D)

# Plot the solution
plot(p_1D)
plt.xlabel('s')
plt.ylabel('p_{1D}')
plt.title('Solution of the 1D PDE with Robin boundary conditions')
plt.grid(True)
plt.show()