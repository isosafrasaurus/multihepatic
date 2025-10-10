# fem/__init__.py
from .solver import BlockLinearSolver
from .operators import ProjectionOperator

__all__ = ["BlockLinearSolver", "ProjectionOperator"]

