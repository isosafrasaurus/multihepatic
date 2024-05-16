# The system from d'Angelo & Quarteroni paper on tissue perfusion
# With Omega a 3d domain and Gamma a 1d domain inside it we want
#
# A1(grad(u), grad(v))_3 + A0(u, v)_3 + (Pi u, Tv)_3 - beta(p, Tv)_1 = (f, Tv)_1
# -beta(q, Pi u)_1      + a1(grad(p), grad(q))_1 + (a0+beta)(p, q)_1 = (f, q)_1

# blood flow from vessel to tissue only in portal vein
# dirichlet boundary condition in lateral faces of 3d mesh

from dolfin import *
from xii import *
import numpy as np


def setup_problem(i, f, eps=None):
    '''Just showcase, no MMS (yet)'''
    # set up constants

    # Alpha1: darcy conductivity of liver
    Alpha1, Alpha0 = Constant(9.6e-6), Constant(0)
    # alpha1: darcy conductivity of vessel
    alpha1, alpha0 = Constant(1.45), Constant(0)
    # hydraulic conductivity between liver and vessel
    beta = Constant(3.09e-5)

    n = 2 ** i

    # create 3d unit cube mesh
    mesh_3d = UnitCubeMesh(n, n, 2 * n)
    radius = 4.21e-2 # averaging radius for cyl. surface
    quadrature_degree = 10  # quadraure degree for that integration


    # create embedded 1d line mesh
    gamma = MeshFunction('size_t', mesh_3d, 1, 0)
    CompiledSubDomain('near(x[0], 0.5) && near(x[1], 0.5)').mark(gamma, 1)
    mesh_1d = EmbeddedMesh(gamma, 1)

    # 1d bc for zero pressure at outlet
    bc_1d = Expression("5*x[2]+2", degree=0)

    # 3d dirichlet bc
    bc_3d = Constant(3)

    # define function that returns lateral faces of unit cube
    def boundary_3d(x, on_boundary):
        return on_boundary and not near(x[2], 0) and not near(x[2], 1)

    V = FunctionSpace(mesh_3d, 'CG', 1)
    Q = FunctionSpace(mesh_1d, 'CG', 1)
    W = (V, Q)

    u, p = list(map(TrialFunction, W))
    v, q = list(map(TestFunction, W))

    # Averaging surface
    cylinder = Circle(radius=radius, degree=quadrature_degree)

    # averaging operator
    Pi_u = Average(u, mesh_1d, cylinder)
    T_v = Average(v, mesh_1d, None)  # This is 3d-1d trace

    dxGamma = Measure('dx', domain=mesh_1d)

    a00 = Alpha1 * inner(grad(u), grad(v)) * dx + Alpha0 * inner(u, v) * dx + beta * inner(Pi_u, T_v) * dxGamma
    a01 = -beta * inner(p, T_v) * dxGamma
    a10 = -beta * inner(Pi_u, q) * dxGamma
    a11 = alpha1 * inner(grad(p), grad(q)) * dxGamma + (alpha0 + beta) * inner(p, q) * dxGamma

    L0 = inner(f, T_v) * dxGamma
    L1 = inner(f, q) * dxGamma

    a = [[a00, a01], [a10, a11]]
    L = [L0, L1]

    # dirichlet boundary condition in 3d and 1d
    bcs = [[DirichletBC(V, bc_3d, boundary_3d)], [DirichletBC(Q, bc_1d, "on_boundary")]]

    return a, L, W, bcs


# --------------------------------------------------------------------

def setup_mms(eps=None):
    '''Simple MMS...'''
    from common import as_expression
    import sympy as sp

    up = []
    fg = Expression('sin(2*pi*x[2]*(pow(x[0], 2)+pow(x[1], 2)))', degree=4)

    return up, fg


def setup_error_monitor(true, history, path=''):
    '''We measure error in H1 and L2 for simplicity'''
    from common import monitor_error, H1_norm, L2_norm
    return monitor_error(true, [], history, path=path)


# --------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib as plt

    i = 4
    f = Constant(0)
    a, L, W, bcs = setup_problem(i, f, eps=None)

    A, b = map(ii_assemble, (a, L))
    A, b = apply_bc(A, b, bcs)
    A, b = map(ii_convert, (A, b))

    # solve the problem by a direct solver
    wh = ii_Function(W)
    solve(ii_convert(A), wh.vector(), ii_convert(b))

    uh3d, uh1d = wh
    File('plots/pv_pressure3d.pvd') << uh3d
    File('plots/pv_pressure1d.pvd') << uh1d
