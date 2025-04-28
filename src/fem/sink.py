import os, gc, numpy as np
from tissue import AveragingRadius, SegmentLength
from dolfin import FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad, DirichletBC, LUSolver, UserExpression, Point, File
from xii import ii_assemble, apply_bc, ii_convert, ii_Function, Circle, Average
from graphnics import TubeFile

class Sink:
    def __init__(self, domain, order = 2):
        self.domain = domain
        V3 = FunctionSpace(domain.Omega, "CG", 1)
        V1 = FunctionSpace(domain.Lambda, "CG", 1)
        self.W  = [V3, V1]
        u3, u1 = map(TrialFunction, (V3, V1))
        v3, v1 = map(TestFunction, (V3, V1))
        radius = AveragingRadius(domain.tree, domain.G, degree = order)
        segment_length = SegmentLength(domain.tree, domain.G, degree = order)
        circle = Circle(radius = radius, degree = order)
        u3_avg = Average(u3, domain.Lambda, circle)
        v3_avg = Average(v3, domain.Lambda, circle)
        D_area = Constant(np.pi) * radius ** 2
        D_perimeter = Constant(2.0 * np.pi) * radius
        k_v_expr = (segment_length * radius ** 2) / Constant(8.0)

        self.c_gamma = Constant(0.0)
        self.c_gamma_a = Constant(0.0)
        self.c_gamma_R = Constant(0.0)
        self.c_mu = Constant(0.0)
        self.c_k_t = Constant(0.0)
        self.c_P_in = Constant(0.0)
        self.c_P_cvp = Constant(0.0)
        
        a00 = (
            ( self.c_k_t / self.c_mu ) * inner( grad( u3 ), grad( v3 ) ) * domain.dxOmega
            + self.c_gamma_R * u3 * v3 * domain.dsOmegaSink
            + self.c_gamma * u3_avg * v3_avg * D_perimeter * domain.dxLambda
        )
        a01 = (
            - self.c_gamma * u1 * v3_avg * D_perimeter * domain.dxLambda
            - ( self.c_gamma_a / self.c_mu ) * u1 * v3_avg * D_area * domain.dsLambdaRobin
        )
        a10 = (
            - self.c_gamma * u3_avg * v1 * D_perimeter * domain.dxLambda
        )
        a11 = (
            ( k_v_expr / self.c_mu ) * D_area * inner( grad( u1 ), grad( v1 ) ) * domain.dxLambda
            + self.c_gamma * u1 * v1 * D_perimeter * domain.dxLambda
            + ( self.c_gamma_a / self.c_mu ) * u1 * v1 * D_area * domain.dsLambdaRobin
        )
        L0 = (
            self.c_gamma_R * self.c_P_cvp * v3 * domain.dsOmegaSink
            + ( self.c_gamma_a * self.c_P_cvp / self.c_mu ) * v3_avg * D_area * domain.dsLambdaRobin
        )
        L1 = (
            ( self.c_gamma_a * self.c_P_cvp / self.c_mu ) * v1 * D_area * domain.dsLambdaRobin
        )

        self.a_forms = [[a00, a01], [a10, a11]]
        self.L_forms = [L0, L1]
        self.inlet_bc = DirichletBC(V1, self.c_P_in, domain.boundary_Lambda, 1)
        self.uh3d = None
        self.uh1d = None

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp):
        self.c_gamma.assign(gamma)
        self.c_gamma_a.assign(gamma_a)
        self.c_gamma_R.assign(gamma_R)
        self.c_mu.assign(mu)
        self.c_k_t.assign(k_t)
        self.c_P_in.assign(P_in)
        self.c_P_cvp.assign(P_cvp)

        A, b = map(ii_assemble, (self.a_forms, self.L_forms))
        inlet_bcs = [self.inlet_bc] if self.inlet_bc.get_boundary_values() else []
        W_bcs = [[], inlet_bcs]
        if inlet_bcs:
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

    def save_vtk(self, directory: str):
        if self.uh3d is None or self.uh1d is None:
            raise RuntimeError("No solution available. Call .solve first.")
        os.makedirs(directory, exist_ok = True)
        TubeFile(self.domain.G, os.path.join(directory, "pressure1d.pvd")) << self.uh1d
        File(os.path.join(directory, "pressure3d.pvd")) << self.uh3d