import os, gc, numpy as np
from tissue import AveragingRadius
from dolfin import FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad, DirichletBC, LUSolver, UserExpression, Point, File
from xii import ii_assemble, apply_bc, ii_convert, ii_Function, Circle, Average
from graphnics import TubeFile

class Sink:
    def __init__(self, domain, order = 2):
        self.domain = domain
        self.V3 = FunctionSpace(domain.Omega, "CG", 1)
        self.V1 = FunctionSpace(domain.Lambda, "CG", 1)
        self.W = [self.V3, self.V1]
        self.u3, self.u1 = map(TrialFunction, (self.V3, self.V1))
        self.v3, self.v1 = map(TestFunction, (self.V3, self.V1))
        self.radius = AveragingRadius(domain, degree = order)
        self.circle = Circle(radius = self.radius, degree = order)
        self.u3_avg = Average(self.u3, domain.Lambda, self.circle)
        self.v3_avg = Average(self.v3, domain.Lambda, self.circle)
        self.uh3d = None
        self.uh1d = None

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp):
        D_area = Constant(np.pi) * self.radius ** 2
        D_perimeter = Constant(2.0 * np.pi) * self.radius
        k_v_expr = self.radius ** 2 / Constant(8.0)
        
        a00 = (
            Constant(k_t / mu) * inner(grad(self.u3), grad(self.v3)) * self.domain.dxOmega
            + Constant(gamma_R) * self.u3 * self.v3 * self.domain.dsOmegaSink
            + Constant(gamma) * self.u3_avg * self.v3_avg * D_perimeter * self.domain.dxLambda
        )
        a01 = (
            - Constant(gamma) * self.u1 * self.v3_avg * D_perimeter * self.domain.dxLambda
            - Constant(gamma_a / mu) * self.u1 * self.v3_avg * D_area * self.domain.dsLambdaRobin
        )
        a10 = (
            - Constant(gamma) * self.u3_avg * self.v1 * D_perimeter * self.domain.dxLambda
        )
        a11 = (
            k_v_expr / Constant(mu) * D_area * inner(grad(self.u1), grad(self.v1)) * self.domain.dxLambda
            + Constant(gamma) * self.u1 * self.v1 * D_perimeter * self.domain.dxLambda
            + Constant(gamma_a / mu) * self.u1 * self.v1 * D_area * self.domain.dsLambdaRobin
        )
        L0 = (
            Constant(gamma_R * P_cvp) * self.v3 * self.domain.dsOmegaSink
            + Constant(gamma_a * P_cvp / mu) * self.v3_avg * D_area * self.domain.dsLambdaRobin
        )
        L1 = (
            Constant(gamma_a * P_cvp / mu) * self.v1 * D_area * self.domain.dsLambdaRobin
        )

        
        a_forms = [[a00, a01], [a10, a11]]
        L_forms = [L0, L1]

        
        inlet_bc = DirichletBC(self.V1, P_in, self.domain.boundary_Lambda, 1)
        inlet_bcs = [inlet_bc] if inlet_bc.get_boundary_values() else []
        W_bcs = [[], inlet_bcs]

        
        A, b = map(ii_assemble, (a_forms, L_forms))
        if any(W_bcs[0]) or any(W_bcs[1]):
            A, b = apply_bc(A, b, W_bcs)
        else:
            raise ValueError("No Dirichlet conditions loaded")
        A, b = map(ii_convert, (A, b))

        
        wh = ii_Function(self.W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)
        self.uh3d, self.uh1d = wh
        self.uh3d.rename("3D Pressure (Pa)", "3D Pressure Distribution")
        self.uh1d.rename("1D Pressure (Pa)", "1D Pressure Distribution")
        return self.uh3d, self.uh1d

    def save_vtk(self, directory: str):
        if self.uh3d is None or self.uh1d is None:
            raise RuntimeError("No solution available. Call .solve() first.")
        os.makedirs(directory, exist_ok=True)
        TubeFile(self.domain.fenics_graph, os.path.join(directory, "pressure1d.pvd")) << self.uh1d
        File(os.path.join(directory, "pressure3d.pvd")) << self.uh3d