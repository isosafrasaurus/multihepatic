from dataclasses import dataclass, field

from dolfin import Constant


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


__all__ = ["Parameters", "ParamConstants"]
