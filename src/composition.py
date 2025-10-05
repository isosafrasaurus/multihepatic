
from typing import Optional, Sequence, Dict, Any
import numpy as np
from dolfin import (FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad,
                    DirichletBC, MeshFunction, Measure, DOLFIN_EPS)
from xii import Circle, Average
from tissue.domain import AveragingRadius, SegmentLength
from tissue.geometry import BoundaryPoint

PARAM_NAMES = ("gamma", "gamma_a", "gamma_R", "mu", "k_t", "P_in", "P_cvp")

def make_param_constants() -> Dict[str, Constant]:
    return {name: Constant(0.0) for name in PARAM_NAMES}

def assign_params(consts: Dict[str, Constant], params: Dict[str, float]) -> None:
    for k, c in consts.items():
        if k in params:
            c.assign(float(params[k]))

def build_coupled_pressure_forms(G, Omega, *,
                                 inlet_nodes: Optional[Sequence[int]] = None,
                                 Omega_sink_subdomain=None,
                                 order: int = 2) -> Dict[str, Any]:
    
    Lambda = G.mesh
    consts = make_param_constants()

    
    boundary_Omega = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)
    boundary_Lambda = MeshFunction("size_t", Lambda, Lambda.topology().dim() - 1, 0)

    if Omega_sink_subdomain is not None:
        Omega_sink_subdomain.mark(boundary_Omega, 1)

    if inlet_nodes:
        coords = Lambda.coordinates()
        for node_id in inlet_nodes:
            BoundaryPoint(coords[node_id], tolerance=DOLFIN_EPS).mark(boundary_Lambda, 1)

    dsOmega = Measure("ds", domain=Omega, subdomain_data=boundary_Omega)
    dsLambda = Measure("ds", domain=Lambda, subdomain_data=boundary_Lambda)
    dxOmega = Measure("dx", domain=Omega)
    dxLambda = Measure("dx", domain=Lambda)
    dsOmegaSink = dsOmega(1)
    dsLambdaRobin = dsLambda(0)

    
    tree = Lambda.bounding_box_tree()
    tree.build(Lambda)
    radius = AveragingRadius(tree, G, degree=order)
    seglen = SegmentLength(tree, G, degree=order)
    circle = Circle(radius=radius, degree=order)

    
    V3 = FunctionSpace(Omega, "CG", 1)
    V1 = FunctionSpace(Lambda, "CG", 1)
    u3, u1 = map(TrialFunction, (V3, V1))
    v3, v1 = map(TestFunction, (V3, V1))
    u3_avg = Average(u3, Lambda, circle)
    v3_avg = Average(v3, Lambda, circle)

    D_area = Constant(np.pi) * radius**2
    D_peri = Constant(2.0*np.pi) * radius
    k_v = (seglen * radius**2) / Constant(8.0)

    a00 = (consts["k_t"]/consts["mu"]) * inner(grad(u3), grad(v3)) * dxOmega \
        + consts["gamma_R"] * u3 * v3 * dsOmegaSink \
        + consts["gamma"] * u3_avg * v3_avg * D_peri * dxLambda

    a01 = - consts["gamma"] * u1 * v3_avg * D_peri * dxLambda \
        - (consts["gamma_a"]/consts["mu"]) * u1 * v3_avg * D_area * dsLambdaRobin

    a10 = - consts["gamma"] * u3_avg * v1 * D_peri * dxLambda

    a11 = (k_v/consts["mu"]) * D_area * inner(grad(u1), grad(v1)) * dxLambda \
        + consts["gamma"] * u1 * v1 * D_peri * dxLambda \
        + (consts["gamma_a"]/consts["mu"]) * u1 * v1 * D_area * dsLambdaRobin

    L0 = consts["gamma_R"] * consts["P_cvp"] * v3 * dsOmegaSink \
       + (consts["gamma_a"]*consts["P_cvp"]/consts["mu"]) * v3_avg * D_area * dsLambdaRobin

    L1 = (consts["gamma_a"]*consts["P_cvp"]/consts["mu"]) * v1 * D_area * dsLambdaRobin

    inlet_bc = DirichletBC(V1, consts["P_in"], boundary_Lambda, 1)

    return dict(
        W=[V3, V1],
        a_blocks=[[a00, a01], [a10, a11]],
        L_blocks=[L0, L1],
        inlet_bc=inlet_bc,
        measures=dict(dxOmega=dxOmega, dxLambda=dxLambda, dsOmega=dsOmega, dsOmegaSink=dsOmegaSink),
        consts=consts
    )

