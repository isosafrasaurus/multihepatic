# @title Solve 1D PDE by FEniCS
%%capture

L = 1.0         # Domain length
beta = 1.0      # Coefficient
p_bar_3D = 1.0  # Mean 3D pressure
p_inf = 0.5     # Far-field pressure
gamma = 1.0     # Boundary condition coefficient

# mesh and function space
mesh = IntervalMesh(50, 0, L)
V = FunctionSpace(mesh, 'P', 1)

# variational problem
p = TrialFunction(V)
v = TestFunction(V)

a = dot(grad(p), grad(v))*dx + beta*p*v*dx
L = beta*p_bar_3D*v*dx

# Robin boundary conditions
a += gamma*p*v*ds
L += gamma*p_inf*v*ds

p_1D = Function(V)
solve(a == L, p_1D)

plot(p_1D)
plt.xlabel('s')
plt.ylabel('p_{1D}')
plt.title('1D FEniCS solution')
plt.grid(True)
plt.show()