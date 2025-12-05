import gc
from dataclasses import dataclass

from fem.solver import BlockLinearSolver
from fem.operators import ProjectionOperator

from .forms import AssembledForms
from .parameters import Parameters
from .contracts import Solution


@dataclass
class PressureSolution(Solution):
    """p3d, p1d solution only."""


@dataclass
class PressureVelocitySolution(Solution):
    """p3d, p1d + v3d."""


class PressureProblem:
    """
    Owns solver and assembled forms; assigns constants and solves the block system.
    """

    def __init__(
            self,
            *,
            forms: AssembledForms,
            Omega=None,
            linear_solver: str = "mumps",
    ) -> None:
        self.forms = forms
        self.solver = BlockLinearSolver(linear_solver=linear_solver)
        self._closed = False

    def solve(self, params: Parameters) -> PressureSolution:
        if self._closed:
            raise RuntimeError("PressureProblem is closed")

        # Push scalar parameters into Constant objects
        self.forms.consts.assign_from(params)

        uh3d, uh1d = self.solver.solve_block(
            self.forms.spaces.W,
            self.forms.a_blocks,
            self.forms.L_blocks,
            inlet_bc=self.forms.inlet_bc,
        )
        return PressureSolution(p3d=uh3d, p1d=uh1d)

    def close(self) -> None:
        if self._closed:
            return
        if self.solver:
            self.solver.close()
        self.forms = None  # type: ignore[assignment]
        self._closed = True
        gc.collect()


class PressureVelocityProblem(PressureProblem):
    """
    Extends PressureProblem by projecting 3D velocity via ProjectionOperator.
    """

    def __init__(
            self,
            *,
            forms: AssembledForms,
            Omega,
            linear_solver: str = "mumps",
    ) -> None:
        super().__init__(forms=forms, Omega=Omega, linear_solver=linear_solver)
        self._proj = ProjectionOperator(Omega)

    def solve(self, params: Parameters) -> PressureVelocitySolution:
        base = super().solve(params)
        v = self._proj.project_velocity(
            base.p3d,
            k_t=params.k_t,
            mu=params.mu,
            dx=self.forms.measures.dxOmega,  # type: ignore[attr-defined]
        )
        return PressureVelocitySolution(p3d=base.p3d, p1d=base.p1d, v3d=v)

    def close(self) -> None:
        if getattr(self, "_proj", None) is not None:
            self._proj.close()
        super().close()
