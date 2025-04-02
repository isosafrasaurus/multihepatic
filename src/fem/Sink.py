import os, fem, tissue, visualize
import numpy as np
from dolfin import FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad, DirichletBC, PETScKrylovSolver, LUSolver
from graphnics import TubeFile
from xii import ii_assemble, apply_bc, ii_convert, ii_Function, Circle, Average

class Sink:
    def __init__(
        self,
        domain: tissue.DomainBuild,
        gamma: float,
        gamma_a: float,
        gamma_R: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        p_cvp: float
    ):
        V3 = FunctionSpace(domain.Omega, "CG", 1)
        V1 = FunctionSpace(domain.Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        cylinder = Circle(radius = domain.radius_map, degree = 2)
        u3_avg = Average(u3, domain.Lambda, cylinder)
        v3_avg = Average(v3, domain.Lambda, cylinder)
        D_area = np.pi * domain.radius_map**2
        D_perimeter = 2.0 * np.pi * domain.radius_map
       
        a00 = (
            Constant(k_t / mu) * inner(grad(u3), grad(v3)) * domain.dxOmega
            + Constant(gamma_R) * u3 * v3 * domain.dsOmegaSink
            + Constant(gamma) * u3_avg * v3_avg * D_perimeter * domain.dxLambda
        )
        a01 = (
            - Constant(gamma) * u1 * v3_avg * D_perimeter * domain.dxLambda
            - Constant(gamma_a / mu) * u1 * v3_avg * D_area * domain.dsLambdaRobin
        )
        a10 = (
            - Constant(gamma) * u3_avg * v1 * D_perimeter * domain.dxLambda
        )
        a11 = (
            Constant(k_v / mu) * D_area * inner(grad(u1), grad(v1)) * domain.dxLambda
            + Constant(gamma) * u1 * v1 * D_perimeter * domain.dxLambda
            + Constant(gamma_a / mu) * u1 * v1 * D_area * domain.dsLambdaRobin
        )
        L0 = (
            Constant(gamma_R) * Constant(p_cvp) * v3 * domain.dsOmegaSink
            + Constant(gamma_a / mu) * Constant(p_cvp) * v3_avg * D_area * domain.dsLambdaRobin
        )
        L1 = (
            Constant(gamma_a / mu) * Constant(p_cvp) * v1 * D_area * domain.dsLambdaRobin
        )
        a = [[a00, a01],
             [a10, a11]]
        L = [L0, L1]

        inlet_bc = DirichletBC(V1, Constant(P_in), domain.boundary_Lambda, 1)
        inlet_bcs = [inlet_bc] if inlet_bc.get_boundary_values() else []
        W_bcs = [[], inlet_bcs]

        A, b = map(ii_assemble, (a, L))
        if any(W_bcs[0]) or any(W_bcs[1]):
            A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))
        wh = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)
        self.uh3d, self.uh1d = wh
        self.uh3d.rename("3D Pressure (Pa)", "3D Pressure Distribution")
        self.uh1d.rename("1D Pressure (Pa)", "1D Pressure Distribution")

    def save_vtk(self, directory):
        os.makedirs(directory, exist_ok=True)
        TubeFile(self.fenics_graph, os.path.join(directory, "pressure1d.pvd")) << self.uh1d
        File(f"{directory}/pressure3d.pvd") << self.uh3d