
import gc
from dataclasses import dataclass
from typing import Optional
from graphnics import FenicsGraph
from dolfin import Function, Mesh
from fem.solver import solve_block
from fem.projection import project_velocity
from .composition import AssembledForms, Parameters

@dataclass
class PressureSolution:
    p3d: Function
    p1d: Function

    def free(self) -> None:
        self.p3d = None
        self.p1d = None
        gc.collect()

@dataclass
class PressureVelocitySolution(PressureSolution):
    v3d: Optional[Function] = None

    def free(self) -> None:
        self.v3d = None
        super().free()

class PressureProblem:
    def __init__(self, *, G: FenicsGraph, Omega: Mesh, forms: AssembledForms):
        self.G = G
        self.Omega = Omega
        self.forms = forms

    def solve(self, params: Parameters) -> PressureSolution:
        self.forms.consts.assign_from(params)
        uh3d, uh1d = solve_block(
            self.forms.spaces.W,
            self.forms.a_blocks,
            self.forms.L_blocks,
            inlet_bc=self.forms.inlet_bc,
            linear_solver="mumps",
        )
        return PressureSolution(p3d=uh3d, p1d=uh1d)

    def dispose(self) -> None:
        self.forms = None  
        gc.collect()


class PressureVelocityProblem(PressureProblem):
    def solve(self, params: Parameters) -> PressureVelocitySolution:
        base = super().solve(params)
        v = project_velocity(
            self.Omega,
            base.p3d,
            k_t=params.k_t,
            mu=params.mu,
            dx=self.forms.measures.dxOmega,
        )
        return PressureVelocitySolution(p3d=base.p3d, p1d=base.p1d, v3d=v)
