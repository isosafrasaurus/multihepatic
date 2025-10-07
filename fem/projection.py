
from dolfin import VectorFunctionSpace, TrialFunction, TestFunction, Function, Constant, inner, grad, solve

def project_velocity(
    Omega,
    uh3d,
    *,
    k_t: float,
    mu: float,
    dx
):
    V = VectorFunctionSpace(Omega, "CG", 1)
    vT, vS = TrialFunction(V), TestFunction(V)
    a = inner(vT, vS) * dx
    L = inner(Constant(-k_t/mu) * grad(uh3d), vS) * dx
    v = Function(V)
    solve(a == L, v, solver_parameters={"linear_solver": "mumps"})
    v.rename("v3d", "3D Velocity")
    return v

