import sys, os, pytz, datetime, math, concurrent.futures, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphnics import FenicsGraph
from scipy.optimize import minimize


WORK_PATH = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
EXPORT_PATH = os.path.join("..", "export")
sys.path.append(SOURCE_PATH)

import fem, tissue
from dolfin import *
from xii import *
from graphnics import *


TEST_NUM_NODES_EXP = 5
TEST_GRAPH_NODES = {
    0: [0.000, 0.020, 0.015],
    1: [0.010, 0.020, 0.015],
    2: [0.022, 0.013, 0.015],
    3: [0.022, 0.028, 0.015],
    4: [0.015, 0.005, 0.015],
    5: [0.015, 0.035, 0.015],
    6: [0.038, 0.005, 0.015],
    7: [0.038, 0.035, 0.015]
}
TEST_GRAPH_EDGES = [
    (0, 1, 0.004), (1, 2, 0.003), (1, 3, 0.003),
    (2, 4, 0.002), (2, 6, 0.003), (3, 5, 0.002), (3, 7, 0.003)
]

X_DEFAULT = [2.570e-06, 1.412e-07, 3.147e-07, 1.543e-10]
TARGET_FLOW = 5.0e-6


def optimize_for_target(fixed_index, fixed_value, default, target,
                        lambda_reg=1e-3, maxiter=30):
    free_init = [val for i, val in enumerate(default) if i != fixed_index]
    free_init_log = np.log(free_init)
    y_ref = free_init_log.copy()

    def objective_log(y):
        x = default[:]
        j = 0
        for i in range(len(x)):
            if i == fixed_index:
                x[i] = fixed_value
            else:
                x[i] = np.exp(y[j]); j += 1
        net_flow = compute_flow(x)[0]
        return (net_flow - target)**2 + lambda_reg * np.sum((y - y_ref)**2)

    result = minimize(objective_log, free_init_log,
                      method='Nelder-Mead', options={'maxiter': maxiter})

    x_opt = default[:]
    j = 0
    for i in range(len(x_opt)):
        if i == fixed_index:
            x_opt[i] = fixed_value
        else:
            x_opt[i] = np.exp(result.x[j]); j += 1
    return x_opt


def compute_flow(x):
    
    GAMMA.assign(x[0]); GAMMA_A.assign(x[1])
    GAMMA_R.assign(x[2]); K_V.assign(x[3])

    
    A, b = map(ii_assemble, (A_forms, L_forms))
    A, b = apply_bc(A, b, BCS)
    A, b = map(ii_convert, (A, b))
    SOLVER.solve(WH.vector(), b)

    
    b_proj = inner(Constant(-float(K_T/MU)) * grad(WH[0]), v_test) * DX
    solve(A_proj == b_proj, V_PROJ,
          solver_parameters={"linear_solver": "mumps"})

    n = FacetNormal(WORK_DOMAIN.Omega)

    
    net_flow   = assemble(inner(V_PROJ, n) * DS_OMEGA)
    lower_out  = assemble(conditional(gt(jump(V_PROJ, n), 0), jump(V_PROJ, n), 0.0) * DS_LOWER)
    upper_in   = assemble(conditional(lt(jump(V_PROJ, n), 0), jump(V_PROJ, n), 0.0) * DS_UPPER)
    upper_out  = assemble(conditional(gt(jump(V_PROJ, n), 0), jump(V_PROJ, n), 0.0) * DS_UPPER)
    upper_net  = assemble(jump(V_PROJ, n) * DS_UPPER)

    
    del A, b
    gc.collect()

    return [net_flow, lower_out, upper_in, upper_out, upper_net]


def process_chunk(args):
    chunk, var_idx, default, var_name = args
    rows = []
    for v in chunk:
        x_opt = optimize_for_target(var_idx, v, default, TARGET_FLOW,
                                    lambda_reg=1e-10, maxiter=30)
        f = compute_flow(x_opt)
        rows.append({
            var_name: v,
            "net_flow": f[0],
            "lower_cube_flux_out": f[1],
            "upper_cube_flux_in":  f[2],
            "upper_cube_flux_out": f[3],
            "upper_cube_flux":     f[4]
        })
    return rows


def sweep_variable(variable_name, variable_values, default,
                   directory=None, n_workers=None):
    mapping = {"gamma": 0, "gamma_a": 1, "gamma_R": 2}
    if variable_name not in mapping:
        raise ValueError("Invalid variable choice")
    idx = mapping[variable_name]
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    chunk_size = math.ceil(len(variable_values) / n_workers)
    chunks = [variable_values[i:i+chunk_size]
              for i in range(0, len(variable_values), chunk_size)]

    tasks = [(chunk, idx, default, variable_name) for chunk in chunks]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers, initializer=worker_init) as executor:
        for result in executor.map(process_chunk, tasks):
            rows.extend(result)

    df = pd.DataFrame(rows).set_index(variable_name)
    if directory:
        os.makedirs(directory, exist_ok=True)
        now = datetime.datetime.now(pytz.timezone("America/Chicago"))
        ts = now.strftime("%Y%m%d_%H%M")
        df.to_csv(os.path.join(directory,
                    f"{variable_name}_sweeps_{ts}.csv"))
    return df


def plot_flow_data_semilog(df, directory=None):
    plot_dir = os.path.join(directory, "plot_flow_data_semilog")
    os.makedirs(plot_dir, exist_ok=True)
    ts = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime("%Y%m%d_%H%M")

    var = df.index.values
    name = df.index.name

    plt.figure()
    plt.semilogx(var, df['lower_cube_flux_out'], marker='o', linestyle='-')
    plt.xlabel(f'{name}'); plt.ylabel('Lower Cube Flux Out'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"lower_cube_flux_out_{ts}.png"))
    plt.close()

    plt.figure()
    plt.semilogx(var, df['upper_cube_flux_in'],  marker='s', linestyle='-')
    plt.semilogx(var, df['upper_cube_flux_out'], marker='^', linestyle='--')
    plt.semilogx(var, df['upper_cube_flux'],     marker='d', linestyle='-.')
    plt.xlabel(f'{name}'); plt.ylabel('Upper Cube Flux'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"upper_cube_flux_{ts}.png"))
    plt.close()


def worker_init():
    global WORK_DOMAIN
    global GAMMA, GAMMA_A, GAMMA_R, MU, K_T, K_V, P_IN, P_CVP
    global A_forms, L_forms, BCS, SOLVER, WH
    global A_proj, v_test, V_PROJ, DX, DS_OMEGA, DS_LOWER, DS_UPPER

    
    graph = FenicsGraph()
    for nid, pos in TEST_GRAPH_NODES.items(): graph.add_node(nid, pos=pos)
    for u, v, r in TEST_GRAPH_EDGES: graph.add_edge(u, v, radius=r)
    graph.make_mesh(n=TEST_NUM_NODES_EXP)
    graph.make_submeshes()
    omega = tissue.OmegaBuild(graph, bounds=[[0,0,0],[0.05,0.04,0.03]])
    WORK_DOMAIN = tissue.DomainBuild(
        graph, omega,
        Lambda_inlet_nodes=[0],
        Omega_sink_subdomain=tissue.AxisPlane(0,0.0)
    )

    from dolfin import (
        Constant, FunctionSpace, TrialFunction, TestFunction,
        FacetNormal, Measure, MeshFunction, VectorFunctionSpace,
        Function, inner, grad, DirichletBC, assemble, solve, jump
    )
    from dolfin import Point, UserExpression, dx
    from xii import ii_assemble, apply_bc, ii_convert, ii_Function, Circle, Average
    from dolfin import LUSolver
    import numpy as _np

    
    GAMMA   = Constant(X_DEFAULT[0]);   GAMMA_A = Constant(X_DEFAULT[1])
    GAMMA_R = Constant(X_DEFAULT[2]);   MU      = Constant(1e-3)
    K_T     = Constant(1e-10);          K_V     = Constant(X_DEFAULT[3])
    P_IN    = Constant(100.0*133.322);  P_CVP   = Constant(1.0*133.322)

    
    lower_bf = MeshFunction("size_t", WORK_DOMAIN.Omega,
                            WORK_DOMAIN.Omega.topology().dim()-1)
    lower_bf.set_all(0)
    upper_bf = MeshFunction("size_t", WORK_DOMAIN.Omega,
                            WORK_DOMAIN.Omega.topology().dim()-1)
    upper_bf.set_all(0)
    from fem.sub_cubes import CubeSubBoundary
    lc = CubeSubBoundary([0,0,0],[0.010,0.010,0.010]); lc.mark(lower_bf,1)
    uc = CubeSubBoundary([0.033,0.030,0.010],[0.043,0.040,0.020]); uc.mark(upper_bf,1)
    ds_lower = Measure("dS", domain=WORK_DOMAIN.Omega,
                       subdomain_data=lower_bf)
    ds_upper = Measure("dS", domain=WORK_DOMAIN.Omega,
                       subdomain_data=upper_bf)
    DS_LOWER = ds_lower(1); DS_UPPER = ds_upper(1)

    
    V3 = FunctionSpace(WORK_DOMAIN.Omega,  "CG", 1)
    V1 = FunctionSpace(WORK_DOMAIN.Lambda, "CG", 1)
    W  = [V3, V1]
    u3, u1 = map(TrialFunction, W)
    v3, v1 = map(TestFunction,  W)

    class AveragingRadius(UserExpression):
        def __init__(self, **kwargs):
            self.G    = WORK_DOMAIN.fenics_graph
            self.tree = WORK_DOMAIN.Lambda.bounding_box_tree()
            self.tree.build(WORK_DOMAIN.Lambda)
            super().__init__(**kwargs)
        def eval(self, value, x):
            p = Point(*x)
            cell = self.tree.compute_first_entity_collision(p)
            if cell == _np.iinfo(_np.uint32).max:
                value[0] = 0.0
            else:
                ix = self.G.mf[cell]
                edge = list(self.G.edges())[ix]
                value[0] = self.G.edges()[edge]['radius']
    radius   = AveragingRadius(degree=2)
    circle   = Circle(radius=radius, degree=2)
    u3_avg   = Average(u3, WORK_DOMAIN.Lambda, circle)
    v3_avg   = Average(v3, WORK_DOMAIN.Lambda, circle)
    D_area   = _np.pi * radius**2
    k_v_expr = (radius**2) / Constant(8.0)

    a00 = (
      (K_T/MU) * inner(grad(u3), grad(v3)) * WORK_DOMAIN.dxOmega
      + GAMMA_R * u3*v3 * WORK_DOMAIN.dsOmegaSink
      + GAMMA   * u3_avg*v3_avg * D_area * WORK_DOMAIN.dxLambda
    )
    a01 = (
      - GAMMA * u1*v3_avg * D_area * WORK_DOMAIN.dxLambda
      - (GAMMA_A/MU) * u1*v3_avg * D_area * WORK_DOMAIN.dsLambdaRobin
    )
    a10 = - GAMMA * u3_avg*v1 * D_area * WORK_DOMAIN.dxLambda
    a11 = (
      (k_v_expr/MU)*D_area*inner(grad(u1),grad(v1)) * WORK_DOMAIN.dxLambda
      + GAMMA   * u1*v1 * D_area * WORK_DOMAIN.dxLambda
      + (GAMMA_A/MU)*u1*v1 * D_area * WORK_DOMAIN.dsLambdaRobin
    )

    L0 = (
      GAMMA_R*P_CVP * v3 * WORK_DOMAIN.dsOmegaSink
      + (GAMMA_A*P_CVP/MU) * v3_avg * D_area * WORK_DOMAIN.dsLambdaRobin
    )
    L1 = (GAMMA_A*P_CVP/MU) * v1 * D_area * WORK_DOMAIN.dsLambdaRobin

    A_forms = [[a00, a01], [a10, a11]]
    L_forms = [L0, L1]

    inlet_bc = DirichletBC(V1, P_IN, WORK_DOMAIN.boundary_Lambda, 1)
    BCS      = [[], [inlet_bc]]

    A0, b0 = map(ii_assemble, (A_forms, L_forms))
    A0, b0 = apply_bc(A0, b0, BCS)
    A0, b0 = map(ii_convert, (A0, b0))
    SOLVER  = LUSolver(A0, "mumps")
    WH      = ii_Function(W)

    
    V_VEC  = VectorFunctionSpace(WORK_DOMAIN.Omega, "CG", 1)
    v_trial = TrialFunction(V_VEC)
    v_test  = TestFunction(V_VEC)
    A_proj  = inner(v_trial, v_test) * WORK_DOMAIN.dxOmega
    V_PROJ  = Function(V_VEC)

    
    DX       = WORK_DOMAIN.dxOmega
    DS_OMEGA = WORK_DOMAIN.dsOmega


if __name__ == '__main__':
    vals = np.logspace(-10, 2, 20)
    df = sweep_variable("gamma", vals, X_DEFAULT, directory=EXPORT_PATH)
    plot_flow_data_semilog(df, directory=EXPORT_PATH)
