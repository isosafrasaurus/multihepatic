import os
import gc
import numpy as np
from dolfin import (
    FunctionSpace,
    TrialFunction,
    TestFunction,
    Constant,
    inner,
    grad,
    DirichletBC,
    LUSolver,
    UserExpression,
    Point,
    File
)
from xii import ii_assemble, apply_bc, ii_convert, ii_Function, Circle, Average
from graphnics import TubeFile

class Sink:
    def __init__(self, domain: "tissue.DomainBuild"):
        self.domain = domain
        
        self.V3 = FunctionSpace(domain.Omega, "CG", 1)
        self.V1 = FunctionSpace(domain.Lambda, "CG", 1)
        self.W = [self.V3, self.V1]

        self.u3 = TrialFunction(self.V3)
        self.u1 = TrialFunction(self.V1)
        self.v3 = TestFunction(self.V3)
        self.v1 = TestFunction(self.V1)

        
        class AveragingRadius(UserExpression):
            def __init__(self, **kwargs):
                self.G = domain.fenics_graph
                self.tree = domain.Lambda.bounding_box_tree()
                self.tree.build(domain.Lambda)
                super().__init__(**kwargs)

            def eval(self, value, x):
                p = Point(x[0], x[1], x[2])
                cell = self.tree.compute_first_entity_collision(p)
                if cell == np.iinfo(np.uint32).max:
                    value[0] = 0.0
                else:
                    edge_ix = self.G.mf[cell]
                    edge = list(self.G.edges())[edge_ix]
                    value[0] = self.G.edges()[edge]['radius']

        self.radius = AveragingRadius(degree=2)
        self.circle = Circle(radius=self.radius, degree=2)

        self.A = None
        self.b = None
        self.solver = None
        self.uh3d = None
        self.uh1d = None

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp):
        if self.uh3d is not None:
            del self.uh3d, self.uh1d, self.solver, self.A, self.b
            gc.collect()

        gamma_c = Constant(gamma)
        gamma_a_c = Constant(gamma_a)
        gamma_R_c = Constant(gamma_R)
        P_in_c = Constant(P_in)
        P_cvp_c = Constant(P_cvp)

        
        u3_avg = Average(self.u3, self.domain.Lambda, self.circle)
        v3_avg = Average(self.v3, self.domain.Lambda, self.circle)

        D_area = np.pi * self.radius**2
        k_v_expr = (self.radius**2) / Constant(8.0)

        a00 = (
            Constant(k_t/mu) * inner(grad(self.u3), grad(self.v3)) * self.domain.dxOmega
            + gamma_R_c * self.u3 * self.v3 * self.domain.dsOmegaSink
            + gamma_c * u3_avg * v3_avg * D_area * self.domain.dxLambda
        )
        a01 = (
            - gamma_c * self.u1 * v3_avg * D_area * self.domain.dxLambda
            - gamma_a_c/mu * self.u1 * v3_avg * D_area * self.domain.dsLambdaRobin
        )
        a10 = (
            - gamma_c * u3_avg * self.v1 * D_area * self.domain.dxLambda
        )
        a11 = (
            k_v_expr/Constant(mu) * D_area * inner(grad(self.u1), grad(self.v1)) * self.domain.dxLambda
            + gamma_c * self.u1 * self.v1 * D_area * self.domain.dxLambda
            + gamma_a_c/mu * self.u1 * self.v1 * D_area * self.domain.dsLambdaRobin
        )
        L0 = (
            gamma_R_c * P_cvp_c * self.v3 * self.domain.dsOmegaSink
            + gamma_a_c * P_cvp_c/mu * v3_avg * D_area * self.domain.dsLambdaRobin
        )
        L1 = (
            gamma_a_c * P_cvp_c/mu * self.v1 * D_area * self.domain.dsLambdaRobin
        )

        
        a_forms = [[a00, a01], [a10, a11]]
        L_forms = [L0, L1]

        
        inlet_bc = DirichletBC(self.V1, P_in_c, self.domain.boundary_Lambda, 1)
        inlet_bcs = [inlet_bc] if inlet_bc.get_boundary_values() else []
        W_bcs = [[], inlet_bcs]

        
        self.A, self.b = map(ii_assemble, (a_forms, L_forms))
        if any(W_bcs[0]) or any(W_bcs[1]):
            self.A, self.b = apply_bc(self.A, self.b, W_bcs)
        self.A, self.b = map(ii_convert, (self.A, self.b))

        
        wh = ii_Function(self.W)
        self.solver = LUSolver(self.A, "mumps")
        self.solver.solve(wh.vector(), self.b)
        self.uh3d, self.uh1d = wh

        
        self.uh3d.rename("3D Pressure (Pa)", "3D Pressure Distribution")
        self.uh1d.rename("1D Pressure (Pa)", "1D Pressure Distribution")

        return self.uh3d, self.uh1d

    def save_vtk(self, directory: str):
        if self.uh3d is None or self.uh1d is None:
            raise RuntimeError("No solution available. Call solve() first.")
        os.makedirs(directory, exist_ok=True)
        TubeFile(self.domain.fenics_graph, os.path.join(directory, "pressure1d.pvd")) << self.uh1d
        File(os.path.join(directory, "pressure3d.pvd")) << self.uh3d