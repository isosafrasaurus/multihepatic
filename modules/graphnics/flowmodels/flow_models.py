


import networkx as nx
from fenics import *
from xii import *
from graphnics import *

class NetworkPoisson:
    

    def __init__(self, G, f=Constant(0), p_bc=Constant(0), kappa=Constant(1), degree=1):

        
        self.G = G

        
        self.f = f
        self.p_bc = p_bc
        self.kappa = kappa

        self.V = FunctionSpace(G.mesh, "CG", degree)

    def a_form(self):
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        G = self.G

        a = self.kappa * inner(G.dds(u), G.dds(v)) * dx

        return a

    def L_form(self):
        v = TestFunction(self.V)
        return self.f * v * dx

    def get_bc(self):
        bc = DirichletBC(self.V, self.p_bc, "on_boundary")
        return bc


class HydraulicNetwork:
    

    def __init__(self, G, f=Constant(0), g = Constant(0), p_bc=Constant(0), Res=Constant(1), degree=1):

        
        self.G = G

        
        self.f = f
        self.g = g
        self.p_bc = p_bc
        self.Res = Res

        self.W = [FunctionSpace(G.mesh, "DG", degree-1),
                  FunctionSpace(G.mesh, "CG", degree)]


        self.qp = list(map(TrialFunction, self.W))
        self.vphi = list(map(TestFunction, self.W))
        
    def a_form(self):
        
        G = self.G
        q, p = self.qp
        v, phi = self.vphi

        a = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]
        
        a[0][0] += self.Res * inner(q, v) * dx
        a[0][1] += inner(G.dds(p), v) * dx
        a[1][0] -= inner(G.dds(phi), q) * dx

        return a


    def L_form(self):
        v, phi = self.vphi
        dx = Measure("dx", domain=self.G.mesh)
        L = [0, 0]
        L[0] = self.g*v*dx
        L[1] = self.f*phi*dx
        
        return L
    
    def get_bc(self):
        bcs = [[], [DirichletBC(self.W[1], self.p_bc, "on_boundary")]]
        return bcs
    
    
    def solve(self):
        
        W = self.W
        a = self.a_form()
        L = self.L_form()

        W_bcs = self.get_bc()
        
        A, b = map(ii_assemble, (a, L))
        A, b = apply_bc(A, b, W_bcs)
        A, b = map(ii_convert, (A, b))

        
        qp = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(qp.vector(), b)
        
        
        return qp





RT = {
    "flux_space": "CG",
    "flux_degree": 1,
    "pressure_space": "DG",
    "pressure_degree": 0,
}


class MixedHydraulicNetwork:
    

    def __init__(self, G, f=Constant(0), g = Constant(0), p_bc=Constant(0), space=RT):
        

        
        self.G = G

        

        self.f = f
        self.g = g
        self.p_bc = p_bc

        
        
        
        
        submeshes = list(nx.get_edge_attributes(G, "submesh").values())

        P2s = [
            FunctionSpace(msh, space["flux_space"], space["flux_degree"])
            for msh in submeshes
        ]
        P1s = [
            FunctionSpace(
                G.mesh, space["pressure_space"], space["pressure_degree"]
            )
        ]
        LMs = [FunctionSpace(G.mesh, "R", 0) for b in G.bifurcation_ixs]

        
        W = P2s + P1s + LMs
        self.W = W

        self.meshes = submeshes + [G.mesh] * (
            G.num_bifurcations + 1
        )  

        
        self.qp = list(map(TrialFunction, W))
        self.vphi = list(map(TestFunction, W))



    def a_form(self, a=None):
        

        if not a:
            a = self.init_a_form()

        qp, vphi = self.qp, self.vphi
        G = self.G
        

        
        n_edges = G.num_edges
        
        qs, vs = qp[:n_edges], vphi[:n_edges]
        p, phi = qp[n_edges], vphi[n_edges]
        lams, xis = qp[n_edges+1:], vphi[n_edges+1:]

        
        submeshes = list(nx.get_edge_attributes(G, "submesh").values())
        ps = [Restriction(p, msh) for msh in submeshes]
        phis = [Restriction(phi, msh) for msh in submeshes]

        
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain=G.edges[e]["submesh"])
            
            try: 
                res = self.G.edges()[e]["Res"]
            except KeyError:
                res = Constant(1)

            a[i][i] += res * qs[i] * vs[i] * dx_edge
            
            a[n_edges][i] += +G.dds_i(qs[i], i) * phis[i] * dx_edge
            a[i][n_edges] += -ps[i] * G.dds_i(vs[i], i) * dx_edge
            
            
        
        edge_list = list(G.edges.keys())
        
        for b_ix, b in enumerate(G.bifurcation_ixs):

            
            conn_edges = {
                **{e: (1, BIF_IN, edge_list.index(e)) for e in G.in_edges(b)},
                **{e: (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b)},
            }

            for e in conn_edges:
                ds_edge = Measure(
                    "ds", domain=G.edges[e]["submesh"], subdomain_data=G.edges[e]["vf"]
                )

                sign, tag, e_ix = conn_edges[e]

                a[e_ix][n_edges + 1 + b_ix] += (
                    sign * vs[e_ix] * lams[b_ix] * ds_edge(tag)
                )
                a[n_edges + 1 + b_ix][e_ix] += (
                    sign * qs[e_ix] * xis[b_ix] * ds_edge(tag)
                )

        return a

    def init_a_form(self):
        

        
        a = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]

        
        for i, msh in enumerate(self.meshes):
            a[i][i] += (
                Constant(0) * self.qp[i] * self.vphi[i] * Measure("dx", domain=msh)
            )

        return a


    def init_L_form(self):
        

        L = [0 for i in range(0, len(self.vphi))]

        
        for i, msh in enumerate(self.meshes):
            dx = Measure("dx", domain=msh)
            L[i] += Constant(0) * self.vphi[i] * dx
            

        return L
    

    def L_form(self):
        

        L = self.init_L_form()

        vphi = self.vphi

        
        n_edges = self.G.num_edges
        vs, phi, xis = vphi[0:n_edges], vphi[n_edges], vphi[n_edges + 1 :]

        submeshes = list(nx.get_edge_attributes(self.G, "submesh").values())
        phis = [Restriction(phi, msh) for msh in submeshes]
        
        fs = [project(self.f, FunctionSpace(msh, 'CG', 1)) for msh in submeshes]
        gs = [project(self.g, FunctionSpace(msh, 'CG', 1)) for msh in submeshes]
        p_bcs = [project(self.p_bc, FunctionSpace(msh, 'CG', 1)) for msh in submeshes]
        
        
        for i, e in enumerate(self.G.edges):
            ds_edge = Measure("ds", domain=self.G.edges[e]["submesh"], subdomain_data=self.G.edges[e]["vf"])
            dx_edge = Measure("dx", domain=self.G.edges[e]["submesh"])

            L[i] -= p_bcs[i]*vs[i]*ds_edge(BOUN_OUT) - self.p_bc*vs[i]*ds_edge(BOUN_IN)
            L[i] += gs[i] * vs[i] * dx_edge
            
            L[n_edges] += fs[i] * phis[i] * dx_edge
            
        for i in range(0, len(self.G.bifurcation_ixs)):
            L[n_edges + 1 + i] += Constant(0) * xis[i] * dx

        return L
    
    def get_bc(self):
        return [[], []]
    
    def solve(self):
        
        W = self.W
        a = self.a_form()
        L = self.L_form()

        A, b = map(ii_assemble, (a, L))
        A, b = map(ii_convert, (A, b))

        qp = ii_Function(W)
        solver = LUSolver(A, "mumps")
        solver.solve(qp.vector(), b)
        
        return qp


class NetworkStokes(MixedHydraulicNetwork):
    
    def __init__(self, G, f=Constant(0), g = Constant(0), p_bc=Constant(0), nu=Constant(1), space=RT):
        self.nu = nu
        
        super().__init__(G=G, f=f, g=g, p_bc=p_bc, space=space)
        
    def a_form(self, a=None):
        

        if not a:
            a = self.init_a_form()

        qp, vphi = self.qp, self.vphi
        G = self.G
        

        
        n_edges = G.num_edges
        
        qs, vs = qp[:n_edges], vphi[:n_edges]
        p, phi = qp[n_edges], vphi[n_edges]
        lams, xis = qp[n_edges+1:], vphi[n_edges+1:]

        
        submeshes = list(nx.get_edge_attributes(G, "submesh").values())
        ps = [Restriction(p, msh) for msh in submeshes]
        phis = [Restriction(phi, msh) for msh in submeshes]

        
        for i, e in enumerate(G.edges):
            dx_edge = Measure("dx", domain=G.edges[e]["submesh"])
            
            try: 
                res = self.G.edges()[e]["Res"]
            except KeyError:
                res = Constant(1)
                
            try: 
                Ainv = self.G.edges()[e]["Ainv"]
            except KeyError:
                Ainv = Constant(1)


            a[i][i] += res * qs[i] * vs[i] * dx_edge + (self.mu/Ainv)*G.dds_i(qs[i], i) * G.dds_i(vs[i], i) * dx_edge
            
            a[n_edges][i] += +G.dds_i(qs[i], i) * phis[i] * dx_edge
            a[i][n_edges] += -ps[i] * G.dds_i(vs[i], i) * dx_edge
            
            
        
        edge_list = list(G.edges.keys())
        
        for b_ix, b in enumerate(G.bifurcation_ixs):

            
            conn_edges = {
                **{e: (1, BIF_IN, edge_list.index(e)) for e in G.in_edges(b)},
                **{e: (-1, BIF_OUT, edge_list.index(e)) for e in G.out_edges(b)},
            }

            for e in conn_edges:
                ds_edge = Measure(
                    "ds", domain=G.edges[e]["submesh"], subdomain_data=G.edges[e]["vf"]
                )

                sign, tag, e_ix = conn_edges[e]

                a[e_ix][n_edges + 1 + b_ix] += (
                    sign * vs[e_ix] * lams[b_ix] * ds_edge(tag)
                )
                a[n_edges + 1 + b_ix][e_ix] += (
                    sign * qs[e_ix] * xis[b_ix] * ds_edge(tag)
                )

        return a
