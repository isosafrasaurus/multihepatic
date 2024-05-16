import json
from dolfin import *
from xii import *
import numpy as np
import sys
sys.path.append('../data/')
from graphnics import *



G = FenicsGraph()
ind = 0
branch_points = {}

for n in range(29):
    file_name = 'pv_json1/Centerline_'+str(n)+'.mrk.json'
    f = open(file_name)
    data = json.load(f)
    f.close()

    
    points = data['markups'][0]['controlPoints']
    radius = data['markups'][0]['measurements'][3]['controlPointValues']
    G.add_nodes_from(range(ind - n, ind + len(points) - n))

    
    v1 = 0
    for key, val in branch_points.items():
        if points[0]['position'] == val:
            v1 = key
            break

    
    v2 = ind - n + 1
    pos_v1 = points[0]['position']
    pos_v2 = points[1]['position']
    G.nodes[v1]["pos"] = pos_v1
    G.nodes[v2]["pos"] = pos_v2
    G.nodes[v1]["radius"] = radius[0]
    G.nodes[v2]["radius"] = radius[1]
    
    G.add_edge(v1, v2)

    for i in range(len(points)-2):
        v1 = ind - n + 1 + i
        v2 = v1 + 1
        pos_v1 = points[i + 1]['position']
        pos_v2 = points[i + 2]['position']
        G.nodes[v1]["pos"] = pos_v1
        G.nodes[v2]["pos"] = pos_v2
        G.nodes[v1]["radius"] = radius[i + 1]
        G.nodes[v2]["radius"] = radius[i + 2]
        G.add_edge(v1, v2)


    
    ind += len(points)
    branch_points.update({ind-n-1: pos_v2})



G.make_mesh()
mesh1d = G.mesh


pos = nx.get_node_attributes(G, "pos")
node_coords = np.asarray(list(pos.values()))
xmin, ymin, zmin = np.min(node_coords, axis = 0)


d = mesh1d.coordinates()
d[:, :] += [-xmin, -ymin, -zmin]


mesh3d = UnitCubeMesh(16, 16, 32)


c = mesh3d.coordinates()
xl, yl, zl = (np.max(node_coords, axis=0)-np.min(node_coords, axis=0)) 

c[:,:] *= [xl+3, yl+3, zl] 



Alpha1 = Constant(9.6e-2)
alpha1 = Constant(1.45e4)
beta = Constant(3.09e-5)


bc_3d = Constant(3)
bc_1d = Expression("0.02*x[2]+6", degree=0)


def boundary_3d(x, on_boundary):
    return on_boundary and not near(x[2], 0) and not near(x[2], zl)


V3 = FunctionSpace(mesh3d, "CG", 1)
V1 = FunctionSpace(mesh1d, "CG", 1)
W = [V3, V1]

u3, u1 = list(map(TrialFunction, W))
v3, v1 = list(map(TestFunction, W))


cylinder = Circle(radius=1.19, degree=10)

Pi_u = Average(u3, mesh1d, cylinder)
Pi_v = Average(v3, mesh1d, cylinder)


dxGamma = Measure("dx", domain=mesh1d)


a00 = Alpha1 * inner(grad(u3), grad(v3)) * dx + beta * inner(Pi_u, Pi_v) * dxGamma
a01 = -beta * inner(u1, Pi_v) * dxGamma
a10 = -beta * inner(Pi_u, v1) * dxGamma
a11 = alpha1 * inner(grad(u1), grad(v1)) * dx + beta * inner(u1, v1) * dxGamma


L0 = inner(Constant(0), Pi_v) * dxGamma
L1 = inner(Constant(0), v1) * dxGamma

a = [[a00, a01], [a10, a11]]
L = [L0, L1]

W_bcs = [[DirichletBC(V3, bc_3d, boundary_3d)], [DirichletBC(V1, bc_1d, "on_boundary")]]

A, b = map(ii_assemble, (a, L))
A, b = apply_bc(A, b, W_bcs)
A, b = map(ii_convert, (A, b))

wh = ii_Function(W)
solver = LUSolver(A, "mumps")
solver.solve(wh.vector(), b)

uh3d, uh1d = wh
File('plots/pv1/pressure1d.pvd') << uh1d
File('plots/pv1/pressure3d.pvd') << uh3d