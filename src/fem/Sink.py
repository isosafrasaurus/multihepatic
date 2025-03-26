import os, fem, tissue, visualize
import numpy as np
from dolfin import FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad, DirichletBC, File, PETScKrylovSolver, LUSolver
from graphnics import ii_assemble, apply_bc, ii_convert, ii_Function, TubeFile
from xii import Circle, Average

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
        self.fenics_graph = domain.fenics_graph
        for name, value in zip(["gamma", "gamma_a", "gamma_R", "mu", "k_t", "k_v", "P_in", "p_cvp"], 
                               [gamma, gamma_a, gamma_R, mu, k_t, k_v, P_in, p_cvp]):
            setattr(self, name, value)
        for attr in ["Omega", "Lambda", "radius_map", "dsOmega", "dsLambda", "dxOmega", "dxLambda", 
                     "dsOmegaNeumann", "dsOmegaSink", "dsLambdaRobin", "dsLambdaInlet", "boundary_Lambda","boundary_Omega"]:
            setattr(self, attr, getattr(domain, attr))

        V3 = FunctionSpace(self.Omega, "CG", 1)
        V1 = FunctionSpace(self.Lambda, "CG", 1)
        W = [V3, V1]
        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        cylinder = Circle(radius=self.radius_map, degree=5)
        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)
        D_area = np.pi * self.radius_map**2
        D_perimeter = 2.0 * np.pi * self.radius_map
       
        a00 = (
            Constant(self.k_t / self.mu) * inner(grad(u3), grad(v3)) * self.dxOmega
            + Constant(self.gamma_R) * u3 * v3 * self.dsOmegaSink
            + Constant(self.gamma) * u3_avg * v3_avg * D_perimeter * self.dxLambda
        )
        a01 = (
            - Constant(self.gamma) * u1 * v3_avg * D_perimeter * self.dxLambda
            - Constant(self.gamma_a / self.mu) * u1 * v3_avg * D_area * self.dsLambdaRobin
        )
        a10 = (
            - Constant(self.gamma) * u3_avg * v1 * D_perimeter * self.dxLambda
        )
        a11 = (
            Constant(self.k_v / self.mu) * D_area * inner(grad(u1), grad(v1)) * self.dxLambda
            + Constant(self.gamma) * u1 * v1 * D_perimeter * self.dxLambda
            + Constant(self.gamma_a / self.mu) * u1 * v1 * D_area * self.dsLambdaRobin
        )
        L0 = (
            Constant(self.gamma_R) * Constant(self.p_cvp) * v3 * self.dsOmegaSink
            + Constant(self.gamma_a / self.mu) * Constant(self.p_cvp) * v3_avg * D_area * self.dsLambdaRobin
        )
        L1 = (
            Constant(self.gamma_a / self.mu) * Constant(self.p_cvp) * v1 * D_area * self.dsLambdaRobin
        )
        a = [[a00, a01],
             [a10, a11]]
        L = [L0, L1]

        inlet_bc = DirichletBC(V1, Constant(self.P_in), self.boundary_Lambda, 1)
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

    def save_vtk(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        TubeFile(self.fenics_graph, os.path.join(directory, "pressure1d.pvd")) << self.uh1d
        File(f"{directory}/pressure3d.pvd") << self.uh3d