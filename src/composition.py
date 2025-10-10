
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any, List
import numpy as np
from graphnics import FenicsGraph
from dolfin import Mesh, FunctionSpace, TrialFunction, TestFunction, Constant, inner, grad, DirichletBC, MeshFunction, Measure, DOLFIN_EPS
from xii import Circle, Average
from tissue.domain import AveragingRadius, SegmentLength
from tissue.geometry import BoundaryPoint

@dataclass
class Parameters:
    gamma: float
    gamma_a: float
    gamma_R: float
    mu: float
    k_t: float
    P_in: float
    P_cvp: float

@dataclass
class ParamConstants:
    gamma: Constant = field(default_factory=lambda: Constant(0.0))
    gamma_a: Constant = field(default_factory=lambda: Constant(0.0))
    gamma_R: Constant = field(default_factory=lambda: Constant(0.0))
    mu: Constant = field(default_factory=lambda: Constant(0.0))
    k_t: Constant = field(default_factory=lambda: Constant(0.0))
    P_in: Constant = field(default_factory=lambda: Constant(0.0))
    P_cvp: Constant = field(default_factory=lambda: Constant(0.0))

    def assign_from(self, p: Parameters) -> None:
        self.gamma.assign(float(p.gamma))
        self.gamma_a.assign(float(p.gamma_a))
        self.gamma_R.assign(float(p.gamma_R))
        self.mu.assign(float(p.mu))
        self.k_t.assign(float(p.k_t))
        self.P_in.assign(float(p.P_in))
        self.P_cvp.assign(float(p.P_cvp))

@dataclass(frozen=True)
class Measures:
    dxOmega: Measure
    dxLambda: Measure
    dsOmega: Measure
    dsOmegaSink: Any

@dataclass(frozen=True)
class Spaces:
    V3: FunctionSpace
    V1: FunctionSpace

    @property
    def W(self) -> List[FunctionSpace]:
        return [self.V3, self.V1]

@dataclass(frozen=True)
class AssembledForms:
    spaces: Spaces
    a_blocks: Any
    L_blocks: Any
    inlet_bc: DirichletBC
    measures: Measures
    consts: ParamConstants

def build_assembled_forms(
    G: FenicsGraph,
    Omega: Mesh,
    *,
    inlet_nodes: Optional[List[int]] = None,
    Omega_sink_subdomain = None,
    order: int = 2
) -> AssembledForms:
    Lambda = G.mesh
    consts = ParamConstants()

    
    boundary_Omega = MeshFunction("size_t", Omega, Omega.topology().dim() - 1, 0)
    boundary_Lambda = MeshFunction("size_t", Lambda, Lambda.topology().dim() - 1, 0)

    if Omega_sink_subdomain is not None:
        Omega_sink_subdomain.mark(boundary_Omega, 1)

    if inlet_nodes:
        for node_id in inlet_nodes:
            BoundaryPoint(G.nodes[node_id]["pos"], tolerance=DOLFIN_EPS).mark(boundary_Lambda, 1)

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
    spaces = Spaces(V3=V3, V1=V1)

    u3, u1 = map(TrialFunction, (V3, V1))
    v3, v1 = map(TestFunction, (V3, V1))
    u3_avg = Average(u3, Lambda, circle)
    v3_avg = Average(v3, Lambda, circle)

    D_area = Constant(np.pi) * radius**2
    D_peri = Constant(2.0 * np.pi) * radius
    k_v = (seglen * radius**2) / Constant(8.0)

    a00 = (consts.k_t/consts.mu) * inner(grad(u3), grad(v3)) * dxOmega \
        + consts.gamma_R * u3 * v3 * dsOmegaSink \
        + consts.gamma * u3_avg * v3_avg * D_peri * dxLambda

    a01 = - consts.gamma * u1 * v3_avg * D_peri * dxLambda \
        - (consts.gamma_a/consts.mu) * u1 * v3_avg * D_area * dsLambdaRobin

    a10 = - consts.gamma * u3_avg * v1 * D_peri * dxLambda

    a11 = (k_v/consts.mu) * D_area * inner(grad(u1), grad(v1)) * dxLambda \
        + consts.gamma * u1 * v1 * D_peri * dxLambda \
        + (consts.gamma_a/consts.mu) * u1 * v1 * D_area * dsLambdaRobin

    L0 = consts.gamma_R * consts.P_cvp * v3 * dsOmegaSink \
       + (consts.gamma_a*consts.P_cvp/consts.mu) * v3_avg * D_area * dsLambdaRobin

    L1 = (consts.gamma_a*consts.P_cvp/consts.mu) * v1 * D_area * dsLambdaRobin

    inlet_bc = DirichletBC(V1, consts.P_in, boundary_Lambda, 1)

    measures = Measures(
        dxOmega=dxOmega, dxLambda=dxLambda, dsOmega=dsOmega, dsOmegaSink=dsOmegaSink
    )

    return AssembledForms(
        spaces=spaces,
        a_blocks=[[a00, a01], [a10, a11]],
        L_blocks=[L0, L1],
        inlet_bc=inlet_bc,
        measures=measures,
        consts=consts,
    )

