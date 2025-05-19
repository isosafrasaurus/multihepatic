import os
import numpy as np
from tissue import AveragingRadius, SegmentLength, BoundaryPoint
from dolfin import FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad, DirichletBC, LUSolver, UserExpression, Point, File, SubDomain, MeshFunction, Measure, UnitCubeMesh, facets, near, DOLFIN_EPS
from xii import ii_assemble, apply_bc, ii_convert, ii_Function, Circle, Average
from graphnics import TubeFile

class Sink:
    def __init__(self, G, Omega, Lambda_num_nodes_exp = 5, Lambda_inlet_nodes = None, Omega_sink_subdomain = None, order = 2):
        
        self.G = G
        self.Omega = Omega
        Lambda = G.mesh
        self.c_gamma = Constant(0.0)
        self.c_gamma_a = Constant(0.0)
        self.c_gamma_R = Constant(0.0)
        self.c_mu = Constant(0.0)
        self.c_k_t = Constant(0.0)
        self.c_P_in = Constant(0.0)
        self.c_P_cvp = Constant(0.0)

        
        boundary_Omega = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)
        boundary_Lambda = MeshFunction("size_t", Lambda, Lambda.topology().dim() - 1, 0)

        
        self.dsOmega = Measure("ds", domain = Omega, subdomain_data = boundary_Omega)
        if Omega_sink_subdomain is not None:
            Omega_sink_subdomain.mark(boundary_Omega, 1)
        if Lambda_inlet_nodes is not None:
            Lambda_coords = Lambda.coordinates()
            for node_id in Lambda_inlet_nodes:
                coordinate = Lambda_coords[node_id]
                inlet_subdomain = BoundaryPoint(coordinate)
                inlet_subdomain.mark(boundary_Lambda, 1)
        self.dsLambda = Measure("ds", domain = Lambda, subdomain_data = boundary_Lambda)
        self.dxOmega = Measure("dx", domain = Omega)
        self.dxLambda = Measure("dx", domain = Lambda)
        self.dsOmegaNeumann = self.dsOmega(0)
        self.dsOmegaSink = self.dsOmega(1)
        self.dsLambdaRobin = self.dsLambda(0)
        self.dsLambdaInlet = self.dsLambda(1)

        
        tree = Lambda.bounding_box_tree()
        tree.build(Lambda)
        radius = AveragingRadius(tree, G, degree = order)
        segment_length = SegmentLength(tree, G, degree = order)
        circle = Circle(radius = radius, degree = order)

        
        V3 = FunctionSpace(Omega, "CG", 1)
        V1 = FunctionSpace(Lambda, "CG", 1)
        self.W  = [V3, V1]
        u3, u1 = map(TrialFunction, (V3, V1))
        v3, v1 = map(TestFunction, (V3, V1))
        u3_avg = Average(u3, Lambda, circle)
        v3_avg = Average(v3, Lambda, circle)
        D_area = Constant(np.pi) * radius ** 2
        D_perimeter = Constant(2.0 * np.pi) * radius
        k_v_expr = (segment_length * radius ** 2) / Constant(8.0)

        
        a00 = (
            (self.c_k_t / self.c_mu) * inner(grad(u3), grad(v3)) * self.dxOmega
            + self.c_gamma_R * u3 * v3 * self.dsOmegaSink
            + self.c_gamma * u3_avg * v3_avg * D_perimeter * self.dxLambda
        )
        a01 = (
            - self.c_gamma * u1 * v3_avg * D_perimeter * self.dxLambda
            - (self.c_gamma_a / self.c_mu) * u1 * v3_avg * D_area * self.dsLambdaRobin
        )
        a10 = (
            - self.c_gamma * u3_avg * v1 * D_perimeter * self.dxLambda
        )
        a11 = (
            (k_v_expr / self.c_mu) * D_area * inner(grad(u1), grad(v1)) * self.dxLambda
            + self.c_gamma * u1 * v1 * D_perimeter * self.dxLambda
            + (self.c_gamma_a / self.c_mu) * u1 * v1 * D_area * self.dsLambdaRobin
        )
        L0 = (
            self.c_gamma_R * self.c_P_cvp * v3 * self.dsOmegaSink
            + (self.c_gamma_a * self.c_P_cvp / self.c_mu) * v3_avg * D_area * self.dsLambdaRobin
        )
        L1 = (
            (self.c_gamma_a * self.c_P_cvp / self.c_mu) * v1 * D_area * self.dsLambdaRobin
        )
        self.inlet_bc = DirichletBC(V1, self.c_P_in, boundary_Lambda, 1)
        self.a_forms = [[a00, a01], [a10, a11]]
        self.L_forms = [L0, L1]

        
        self.uh3d = None
        self.uh1d = None

    def solve(self, gamma, gamma_a, gamma_R, mu, k_t, P_in, P_cvp, directory = None):
        
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
            raise ValueError("No Dirichlet conditions")
        A, b = map(ii_convert, (A, b))
        wh = ii_Function(self.W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)
        self.uh3d, self.uh1d = wh
        self.uh3d.rename("3D Pressure (Pa)", "3D Pressure Distribution")
        self.uh1d.rename("1D Pressure (Pa)", "1D Pressure Distribution")

        
        if directory is not None:
            os.makedirs(directory, exist_ok = True)
            TubeFile(self.G, os.path.join(directory, "pressure1d.pvd")) << self.uh1d
            File(os.path.join(directory, "pressure3d.pvd")) << self.uh3d

        return self.uh3d, self.uh1d