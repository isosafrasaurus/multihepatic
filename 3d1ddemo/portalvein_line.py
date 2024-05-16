








from dolfin import *
from xii import *
import numpy as np


def setup_problem(i, f, eps=None):
    
    

    
    Alpha1, Alpha0 = Constant(9.6e-6), Constant(0)
    
    alpha1, alpha0 = Constant(1.45), Constant(0)
    
    beta = Constant(3.09e-5)

    n = 2 ** i

    
    mesh_3d = UnitCubeMesh(n, n, 2 * n)
    radius = 4.21e-2 
    quadrature_degree = 10  


    
    gamma = MeshFunction('size_t', mesh_3d, 1, 0)
    CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(gamma, 1)
    mesh_1d = EmbeddedMesh(gamma, 1)

    
    bc_1d = Expression("5*x[2]+2", degree=0)

    
    bc_3d = Constant(3)

    
    def boundary_3d(x, on_boundary):
        return on_boundary and not near(x[2], 0) and not near(x[2], 1)

    V = FunctionSpace(mesh_3d, 'CG', 1)
    Q = FunctionSpace(mesh_1d, 'CG', 1)
    W = (V, Q)

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))

    
    cylinder = Circle(radius=radius, degree=quadrature_degree)

    
    Pi_u = Average(u, mesh_1d, cylinder)
    T_v = Average(v, mesh_1d, None)  

    dxGamma = Measure('dx', domain=mesh_1d)

    a00 = Alpha1 * inner(grad(u), grad(v)) * dx + Alpha0 * inner(u, v) * dx + beta * inner(Pi_u, T_v) * dxGamma
    a01 = -beta * inner(p, T_v) * dxGamma
    a10 = -beta * inner(Pi_u, q) * dxGamma
    a11 = alpha1 * inner(grad(p), grad(q)) * dxGamma + (alpha0 + beta) * inner(p, q) * dxGamma

    L0 = inner(f, T_v) * dxGamma
    L1 = inner(f, q) * dxGamma

    a = [[a00, a01], [a10, a11]]
    L = [L0, L1]

    
    bcs = [[DirichletBC(V, bc_3d, boundary_3d)], [DirichletBC(Q, bc_1d, "on_boundary")]]

    return a, L, W, bcs




def setup_mms(eps=None):
    
    from common import as_expression
    import sympy as sp

    up = []
    fg = Expression('sin(2*pi*x[2]*(pow(x[0], 2)+pow(x[1], 2)))', degree=4)

    return up, fg


def setup_error_monitor(true, history, path=''):
    
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [], history, path=path)




if __name__ == '__main__':
    import matplotlib as plt

    i = 4
    f = Constant(0)
    a, L, W, bcs = setup_problem(i, f, eps=None)

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, bcs)
    A, b = map(ii_convert, (A, b))

    
    wh = ii_Function(W)
    solve(ii_convert(A), wh.vector(), ii_convert(b))

    uh3d, uh1d = wh
    File('plots/pv_pressure3d.pvd') << uh3d
    File('plots/pv_pressure1d.pvd') << uh1d
