import warnings
from typing import Tuple, Optional

import math
import numpy as np
from dolfin import UserExpression, UnitCubeMesh, Point


class AveragingRadius(UserExpression):
    def __init__(self, tree, G, **kwargs):
        super().__init__(**kwargs)
        self.tree = tree
        self.G = G

    def eval(self, value, x):
        p = Point(x[0], x[1], x[2])
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
            return
        edge_ix = self.G.mf[cell]
        edge = list(self.G.edges())[edge_ix]
        value[0] = float(self.G.edges()[edge]['radius'])


class SegmentLength(UserExpression):
    def __init__(self, tree, G, **kwargs):
        super().__init__(**kwargs)
        self.tree = tree
        self.G = G

    def eval(self, value, x):
        p = Point(*x)
        cell = self.tree.compute_first_entity_collision(p)
        if cell == np.iinfo(np.uint32).max:
            value[0] = 0.0
            return
        edge_ix = self.G.mf[cell]
        u, v = list(self.G.edges())[edge_ix]
        pos_u = np.array(self.G.nodes[u]['pos'], dtype=float)
        pos_v = np.array(self.G.nodes[v]['pos'], dtype=float)
        value[0] = float(np.linalg.norm(pos_v - pos_u))


def cells_from_mm_resolution(Lx_mm: float, Ly_mm: float, Lz_mm: float, h_mm: float) -> Tuple[int, int, int]:
    nx = max(1, int(math.ceil(Lx_mm / h_mm)))
    ny = max(1, int(math.ceil(Ly_mm / h_mm)))
    nz = max(1, int(math.ceil(Lz_mm / h_mm)))
    return nx, ny, nz


def _graph_bounds(G) -> Tuple[np.ndarray, np.ndarray]:
    
    positions = [data['pos'] for _, data in G.nodes(data=True)]
    pos = np.asarray(positions, dtype=float)
    return np.min(pos, axis=0), np.max(pos, axis=0)


def _compute_bounds_and_scale(
        G,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        padding_m: float = 0.008,
        *,
        strict_bounds: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    lam_min, lam_max = _graph_bounds(G)

    if bounds is None:
        scales = (lam_max - lam_min) + 2.0 * padding_m
        lower = lam_min - padding_m
        upper = lower + scales
    else:
        lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
        if strict_bounds:
            if not (np.all(lam_min >= lower) and np.all(lam_max <= upper)):
                raise ValueError("Graph coordinates are not fully contained within the provided bounds.")
        else:
            
            if not (np.all(lam_min >= lower) and np.all(lam_max <= upper)):
                warnings.warn(
                    "Graph coordinates extend beyond provided bounds; proceeding with partial coupling.",
                    RuntimeWarning,
                )
        scales = upper - lower

    return lower, upper, scales


def _scale_unitcube_to_box(mesh, lower: np.ndarray, upper: np.ndarray) -> None:
    coords = mesh.coordinates()
    scales = (upper - lower).astype(coords.dtype)
    shifts = lower.astype(coords.dtype)
    coords[:] = coords * scales + shifts


def build_mesh_by_counts(
        G,
        counts: Tuple[int, int, int] = (16, 16, 16),
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        padding_m: float = 0.008,
        *,
        strict_bounds: bool = True,
):
    
    lower, upper, _ = _compute_bounds_and_scale(
        G, bounds=bounds, padding_m=padding_m, strict_bounds=strict_bounds
    )
    mesh = UnitCubeMesh(*tuple(int(max(1, c)) for c in counts))
    _scale_unitcube_to_box(mesh, lower, upper)
    return mesh, [lower, upper]


def build_mesh_by_spacing(
        G,
        spacing_m: float = 1e-3,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        padding_m: float = 0.008,
        *,
        strict_bounds: bool = True,
):
    
    lower, upper, scales = _compute_bounds_and_scale(
        G, bounds=bounds, padding_m=padding_m, strict_bounds=strict_bounds
    )
    nx = max(1, int(np.ceil(scales[0] / spacing_m)))
    ny = max(1, int(np.ceil(scales[1] / spacing_m)))
    nz = max(1, int(np.ceil(scales[2] / spacing_m)))
    mesh = UnitCubeMesh(nx, ny, nz)
    _scale_unitcube_to_box(mesh, lower, upper)
    return mesh, [lower, upper]


def build_mesh_by_mm_resolution(
        G,
        h_mm: float = 1.0,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        padding_m: float = 0.008,
        *,
        strict_bounds: bool = True,
):
    
    lower, upper, scales_m = _compute_bounds_and_scale(
        G, bounds=bounds, padding_m=padding_m, strict_bounds=strict_bounds
    )
    Lx_mm, Ly_mm, Lz_mm = (scales_m * 1000.0).tolist()
    nx, ny, nz = cells_from_mm_resolution(Lx_mm, Ly_mm, Lz_mm, h_mm)
    mesh = UnitCubeMesh(nx, ny, nz)
    _scale_unitcube_to_box(mesh, lower, upper)
    return mesh, [lower, upper]
