


from fenics import *
from xii import *
from tqdm import tqdm
from .flow_models import *
from graphnics import *
set_log_level(40)

time_stepping_schemes = {"IE": {"b1": 0, "b2": 1}, "CN": {"b1": 0.5, "b2": 0.5}}


class TimeDepHydraulicNetwork(HydraulicNetwork):
    
    def __init__(self, G, f=Constant(0), g = Constant(0), p_bc=Constant(0), Res=Constant(1), Ainv=Constant(1)):
        
        self.Ainv = Ainv
        super().__init__(G, f=f, g=g, p_bc=p_bc, Res=Res)
        
    def mass_matrix_q(self, coeff):
        

        dx = Measure("dx", domain=self.G.mesh)
        
        
        M = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]
        
        M[0][0] += coeff*self.qp[0]*self.vphi[0]*dx 
                
        
        M[1][1] = Constant(0)*self.qp[1]*self.vphi[1]*dx
        
        return M

    def mass_vector_q(self, u, coeff):
        
        
        dx = Measure("dx", domain=self.G.mesh)
        
        Mv = [0 for i in range(0, len(self.qp))] 
        Mv[0] = coeff*u[0]*self.vphi[0]*dx 
        
        
        Mv[1] = Constant(0)*self.vphi[1]*dx 
        
        return Mv



class TimeDepMixedHydraulicNetwork(MixedHydraulicNetwork):

    def __init__(self, G, f=Constant(0), g = Constant(0), p_bc=Constant(0), Res=Constant(1), Ainv=Constant(1)):
        
        self.Ainv = Ainv
        super().__init__(G, f=f, g=g, p_bc=p_bc)
        
    def mass_matrix_q(self, coeff):
        

        dx = Measure("dx", domain=self.G.mesh)
        
        
        M = [[0 for i in range(0, len(self.qp))] for j in range(0, len(self.qp))]
        
        
        for i, e in enumerate(self.G.edges):
            dx_edge = Measure("dx", domain=self.G.edges()[e]['submesh'])
            M[i][i] += coeff*self.qp[i]*self.vphi[i]*dx_edge
                
        
        dx = Measure("dx", domain=self.G.mesh)
        
        for i in range(self.G.num_edges, len(self.qp)):
            M[i][i] = Constant(0)*self.qp[i]*self.vphi[i]*dx
        
        return M

    def mass_vector_q(self, u, coeff):
        
        
        dx = Measure("dx", domain=self.G.mesh)
        
        Mv = [0 for i in range(0, len(self.qp))] 
        
        
        for i, e in enumerate(self.G.edges):
            dx_edge = Measure("dx", domain=self.G.edges()[e]['submesh'])
            Mv[i] += coeff*u[i]*self.vphi[i]*dx_edge
                
        
        for i in range(self.G.num_edges, len(self.qp)):
            Mv[i] = Constant(0)*self.vphi[i]*dx
        
        return Mv


def time_stepping_stokes(
    model, t=Constant(0), t_steps=10, T=1, qp0=None, t_step_scheme="IE", reassemble_lhs=True
):
    
    
    if qp0 is None:
        qp0 = ii_Function(model.W)  

    dt = T / t_steps 
    cn, cn1 = time_stepping_schemes[t_step_scheme].values() 
    
    
    Dn1 = model.mass_matrix_q(model.Ainv)
    Dn = model.mass_vector_q(qp0, model.Ainv)
    
    a, L = model.a_form(), model.L_form()

    qpn1 = ii_Function(model.W)

    
    qp_0 = ii_Function(model.W)
    for i, qp0_comp in enumerate(qp_0):
        qp0_comp.vector()[:] = project(qp0[i], model.W[i]).vector().get_local()
    qp0 = qp_0
    
    qps = [qp_0]

    
    An, Ln, DDn = [ii_assemble(term) for term in [a, L, Dn]]

    t.assign(dt)
    
    An1, Ln1, DDn1 = [ii_assemble(term) for term in [a, L, Dn1]]
    
    for t_val in tqdm(np.linspace(dt, T, t_steps - 1)):
    
        
        Ln1 = ii_assemble(L)
        if reassemble_lhs:
            An1, DDn1 = [ii_assemble(term) for term in [a, Dn1]]
            
        
        A = (DDn1 + cn1*dt*An1)
        b = (DDn + cn1*dt*Ln1 - dt*cn*An*qp0.block_vec() + dt*cn*Ln)

        A = A.block_collapse()
        A, b = apply_bc(A, b, model.get_bc())
        
        A, b = ii_convert(A), ii_convert(b)
        solver = LUSolver(A, "mumps")
        solver.solve(qpn1.vector(), b)

        
        sol = ii_Function(model.W)
        [sol[i].assign(func) for i, func in enumerate(qpn1)]
                
        qps.append(sol)
        
        
        [qp0[i].assign(func) for i, func in enumerate(qpn1)]

        
        t.assign(t_val+dt) 
        
        
        for func in [model.p_bc, model.g, model.f]:
            try: 
                func.t = t_val+dt 
            except AttributeError:
                pass 
        
        
        An = An1
        Ln = Ln1
        DDn = DDn1*qp0.block_vec()
        
        a = model.a_form()
        L = model.L_form()

    return qps




if __name__ == "__main__":
    print("Testing time stepping")
    from IPython import embed; embed()
    test_time_stepping()