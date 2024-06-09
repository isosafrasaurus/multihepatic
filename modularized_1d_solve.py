from fenics import *
import matplotlib.pyplot as plt


L = 1.0         
beta = 1.0      
p_bar_3D = 1.0  
p_inf = 0.5     
gamma = 1.0     
f_1D = Constant(0.0)  


mesh = IntervalMesh(50, 0, L)
V = FunctionSpace(mesh, 'P', 1)


boundary_nodes = [0,L]


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


robin_bc = RobinBC(gamma, p_inf, boundary_nodes, degree=0)


p = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(p), grad(v))*dx + beta*p*v*dx
L = beta*p_bar_3D*v*dx + f_1D*v*dx


n = FacetNormal(mesh)
a += gamma*p*v*ds
L += robin_bc*v*ds


p_1D = Function(V)
solve(a == L, p_1D)


plot(p_1D)
plt.xlabel('s')
plt.ylabel('p_{1D}')
plt.title('Solution of the 1D PDE with Robin boundary conditions')
plt.grid(True)
plt.show()