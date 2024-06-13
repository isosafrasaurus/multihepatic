

V1 = FunctionSpace(mesh1d, "CG", 1)
u1 = TrialFunction(V1)
v1 = TestFunction(V1)


alpha1 = Constant(1.45e4)
beta = Constant(3.09e-5)


bc_1d = Expression("0.02*x[2]+6", degree=0)


dxGamma = Measure("dx", domain=mesh1d)


a11 = alpha1 * inner(grad(u1), grad(v1)) * dx + beta * inner(u1, v1) * dxGamma


L1 = inner(Constant(0), v1) * dxGamma


A, b = assemble_system(a11, L1, DirichletBC(V1, bc_1d, "on_boundary"))


uh1d = Function(V1)
solve(A, uh1d.vector(), b)


File(WD_PATH + 'plots/pv1_1dlim/pressure1d.pvd') << uh1d

visualize(mesh1d, uh1d)