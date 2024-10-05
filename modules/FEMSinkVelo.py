from dolfin import *
from graphnics import *
from xii import *
from rtree import index as rtree_index
import numpy as np
import os

from MeasureMeshCreator import MeasureMeshCreator, XZeroPlane
import VTKExporter
from RadiusFunction import RadiusFunction


class FEMSinkVelo:
    def __init__(
        self, 
        G: "FenicsGraph",
        gamma: float,
        gamma_R: float,
        gamma_v: float,   
        gamma_a: float,
        mu: float,
        k_t: float,
        k_v: float,
        P_in: float,
        P_cvp: float,
        inlet_points: list[int] = [0],
        Omega_sink: SubDomain = XZeroPlane()
    ):
        
        self.gamma = gamma
        self.gamma_R = gamma_R
        self.gamma_a = gamma_a
        self.P_cvp = P_cvp

        self.mu = mu
        self.k_v = k_v
        self.k_t = k_t

        
        
        
        measure_creator = MeasureMeshCreator(G, Omega_sink, Lambda_inlet=inlet_points)
        mesh_data = measure_creator.create_mesh_and_measures()

        self.Lambda = mesh_data["Lambda"]       
        self.Omega = mesh_data["Omega"]         

        
        edge_marker = mesh_data["edge_marker"]
        self.dxOmega = mesh_data["dxOmega"]
        self.dxLambda = mesh_data["dxLambda"]
        self.dsOmega_sink = mesh_data["dsOmegaSink"]       
        self.dsLambda_robin = mesh_data["dsLambdaNeumann"] 

        
        
        
        V3 = FunctionSpace(self.Omega, "CG", 1)   
        V1 = FunctionSpace(self.Lambda, "CG", 1)  
        W = [V3, V1]

        u3, u1 = map(TrialFunction, W)
        v3, v1 = map(TestFunction, W)

        
        
        
        self.radius_map = RadiusFunction(G, edge_marker, degree=5)
        cylinder = Circle(radius=self.radius_map, degree=5)  

        
        u3_avg = Average(u3, self.Lambda, cylinder)
        v3_avg = Average(v3, self.Lambda, cylinder)

        
        D_area = np.pi * self.radius_map**2
        D_perimeter = 2.0 * np.pi * self.radius_map

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        a00 = (Constant(self.k_t/self.mu)*inner(grad(u3), grad(v3))*self.dxOmega
               - Constant(self.gamma)*u3_avg*v3_avg*D_perimeter*self.dxLambda  
               + Constant(self.gamma_R)*u3*v3*self.dsOmega_sink)               

        
        a01 = (Constant(self.gamma)*u1*v3_avg*D_perimeter*self.dxLambda)

        
        a10 = (- Constant(self.gamma)*u3_avg*v1*D_perimeter*self.dxLambda)

        
        a11 = (Constant(self.k_v/self.mu)*inner(grad(u1), grad(v1))*D_area*self.dxLambda
               + Constant(self.gamma)*u1*v1*D_perimeter*self.dxLambda
               + Constant(self.gamma_a/self.mu)*u1*v1*self.dsLambda_robin)

        a = [[a00, a01],
             [a10, a11]]

        
        
        
        
        L0 = Constant(self.gamma_R)*Constant(P_cvp)*v3*self.dsOmega_sink
        
        L1 = -Constant(self.gamma_a/self.mu)*Constant(P_cvp)*v1*self.dsLambda_robin

        L = [L0, L1]

        
        
        
        
        inlet_bc = DirichletBC(
            W[1],
            Constant(P_in),
            mesh_data["Lambda_boundary_markers"],
            2  
        )
        W_bcs = [[], [inlet_bc]]

        
        
        
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))

        wh = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(wh.vector(), b)

        
        uh3d, uh1d = wh
        uh3d.rename("3D Pressure", "3D Pressure Distribution")
        uh1d.rename("1D Pressure", "1D Pressure Distribution")

        self.uh3d, self.uh1d = uh3d, uh1d

        
        
        
        
        V_vector = VectorFunctionSpace(self.Omega, "CG", 1)
        u_velocity = TrialFunction(V_vector)
        v_velocity = TestFunction(V_vector)

        a_vel = dot(u_velocity, v_velocity)*dx
        L_vel = -Constant(self.k_t/self.mu)*dot(grad(self.uh3d), v_velocity)*self.dxOmega

        A_vel = assemble(a_vel)
        b_vel = assemble(L_vel)
        solver_vel = LUSolver(A_vel, "mumps")

        velocity = Function(V_vector, name="Velocity_L2_Projection")
        solver_vel.solve(velocity.vector(), b_vel)
        velocity.rename("Velocity", "Velocity Field")
        self.velocity = velocity

    
    
    
    def calculate_3d_outflow(self) -> float:
        
        return assemble(Constant(self.gamma_R)*(self.uh3d - self.P_cvp)*self.dsOmega_sink)

    def calculate_3d_flow_from_velocity(self) -> float:
        
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*self.dsOmega_sink)

    def calculate_1d_inflow(self) -> float:
        
        mesh1d = self.Lambda

        
        v0 = Vertex(mesh1d, 0)
        edges = list(v0.entities(1))
        if not edges:
            return 0.0

        e = Edge(mesh1d, edges[0])
        verts = list(e.entities(0))
        neighbor_vid = verts[0] if verts[0] != v0.index() else verts[1]
        v1 = Vertex(mesh1d, neighbor_vid)

        coords0 = np.array(v0.point().array())
        coords1 = np.array(v1.point().array())
        distance = np.linalg.norm(coords1 - coords0)
        if distance < 1e-14:
            return 0.0

        p0 = self.uh1d(v0.point())
        p1 = self.uh1d(v1.point())
        dP_ds = (p1 - p0)/distance

        radius0 = float(self.radius_map(v0.point()))
        area0 = np.pi*(radius0**2)

        
        Q_in = - area0*(self.k_v/self.mu)*dP_ds
        return Q_in

    def calculate_total_flow_all_boundaries(self) -> float:
        
        ds_all = Measure("ds", domain=self.Omega)
        n = FacetNormal(self.Omega)
        return assemble(dot(self.velocity, n)*ds_all)

    
    
    
    def save_vtk(self, directory_path: str):
        
        os.makedirs(directory_path, exist_ok=True)

        output_file_1d = os.path.join(directory_path, "pressure1d.vtk")
        output_file_3d = os.path.join(directory_path, "pressure3d.pvd")
        output_file_velocity = os.path.join(directory_path, "velocity3d.pvd")

        
        VTKExporter.fenics_to_vtk(
            self.Lambda,
            output_file_1d,
            self.radius_map,
            uh1d=self.uh1d
        )

        
        File(output_file_3d) << self.uh3d
        File(output_file_velocity) << self.velocity
