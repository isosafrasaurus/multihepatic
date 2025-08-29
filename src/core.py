import os
import gc
import json
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type, Any, Dict

import numpy as np
from dolfin import Mesh

from .tissue import get_Omega_rect, get_Omega_rect_from_res
from .meshing import get_fg_from_json
from .fem import Sink, Velo, SubCubes
from .checkpoint import CsvWriter


class Domain1D:
    
    def __init__(self, G, Lambda_num_nodes_exp: int = 5, inlet_nodes: Optional[Sequence[int]] = None):
        self.G = G
        self.Lambda_num_nodes_exp = Lambda_num_nodes_exp
        self.inlet_nodes = list(inlet_nodes) if inlet_nodes is not None else None

    @classmethod
    def from_json(cls, directory: str, Lambda_num_nodes_exp: int = 5, inlet_nodes: Optional[Sequence[int]] = None):
        G = get_fg_from_json(directory)
        
        if not getattr(G, "mesh", None):
            
            G.make_mesh(num_nodes_exp=Lambda_num_nodes_exp)
        
        has_submesh = any(("submesh" in G.edges[e]) for e in G.edges)
        if not has_submesh and hasattr(G, "make_submeshes"):
            G.make_submeshes()
        has_tangent = all(("tangent" in G.edges[e]) for e in G.edges) if len(G.edges) > 0 else True
        if not has_tangent and hasattr(G, "compute_tangents"):
            G.compute_tangents()
        return cls(G, Lambda_num_nodes_exp, inlet_nodes)

    @property
    def Lambda(self) -> Mesh:
        return self.G.mesh

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dispose()

    def dispose(self):
        
        try:
            for e in list(self.G.edges):
                self.G.edges[e].pop("submesh", None)
                self.G.edges[e].pop("tangent", None)
        except Exception:
            pass
        
        if hasattr(self.G, "mesh"):
            self.G.mesh = None
        gc.collect()


class Domain3D:
    
    def __init__(self, Omega: Mesh, bounds: Sequence[np.ndarray]):
        self.Omega = Omega
        self.bounds = bounds

    @classmethod
    def from_graph(cls, G, bounds: Optional[Sequence[np.ndarray]] = None,
                   voxel_res: Optional[float] = None, voxel_dim: Tuple[int, int, int] = (16, 16, 16),
                   padding: float = 8e-3):
        if voxel_res is not None:
            Omega, bounds = get_Omega_rect_from_res(G, bounds=bounds, voxel_res=voxel_res, padding=padding)
        else:
            Omega, bounds = get_Omega_rect(G, bounds=bounds, voxel_dim=voxel_dim, padding=padding)
        return cls(Omega, bounds)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dispose()

    def dispose(self):
        self.Omega = None
        gc.collect()


@dataclass
class PhysicsParams:
    gamma: float
    gamma_a: float
    gamma_R: float
    mu: float
    k_t: float
    P_in: float
    P_cvp: float


@dataclass
class SolveResult:
    uh3d: Any
    uh1d: Any
    velocity: Optional[Any] = None
    extras: Optional[Dict[str, float]] = None

    def free_fields(self):
        self.uh3d = None
        self.uh1d = None
        self.velocity = None
        gc.collect()


class Simulation:
    
    def __init__(self, Lambda: Domain1D, Omega: Domain3D,
                 problem_cls: Type[Sink] = Sink,
                 out_dir: Optional[str] = None,
                 lower_cube_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 upper_cube_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.Lambda = Lambda
        self.Omega = Omega
        self.out_dir = out_dir

        if problem_cls is SubCubes:
            assert lower_cube_bounds is not None and upper_cube_bounds is not None, \
                "SubCubes requires lower_cube_bounds and upper_cube_bounds."
            self.problem = SubCubes(Lambda.G, Omega.Omega,
                                    lower_cube_bounds, upper_cube_bounds,
                                    Lambda_num_nodes_exp=Lambda.Lambda_num_nodes_exp,
                                    Lambda_inlet_nodes=Lambda.inlet_nodes)
        else:
            self.problem = problem_cls(Lambda.G, Omega.Omega,
                                       Lambda_num_nodes_exp=Lambda.Lambda_num_nodes_exp,
                                       Lambda_inlet_nodes=Lambda.inlet_nodes)

    def solve(self, params: PhysicsParams, save_pvd: bool = False) -> SolveResult:
        out = self.out_dir if (self.out_dir and save_pvd) else None

        if isinstance(self.problem, Velo) or isinstance(self.problem, SubCubes):
            uh3d, uh1d, velocity = self.problem.solve(
                params.gamma, params.gamma_a, params.gamma_R,
                params.mu, params.k_t, params.P_in, params.P_cvp,
                directory=out
            )
            return SolveResult(uh3d=uh3d, uh1d=uh1d, velocity=velocity, extras={})
        else:
            uh3d, uh1d = self.problem.solve(
                params.gamma, params.gamma_a, params.gamma_R,
                params.mu, params.k_t, params.P_in, params.P_cvp,
                directory=out
            )
            return SolveResult(uh3d=uh3d, uh1d=uh1d, velocity=None, extras={})

    def save_path_pressure_csv(self, path: Sequence[int], csv_name: str = "path_pressure.csv"):
        from .pressure_drop import get_path_pressure
        df = get_path_pressure(self.Lambda.G, self.problem.uh1d, path)
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            df.to_csv(os.path.join(self.out_dir, csv_name), index=False)
        return df

    def save_fluxes_csv(self, csv_name: str = "fluxes.csv"):
        records = {}
        p = self.problem

        if hasattr(p, "compute_net_flow_sink"):
            records["net_flow_sink"] = float(p.compute_net_flow_sink())
        if hasattr(p, "compute_inflow_sink"):
            records["inflow_sink"] = float(p.compute_inflow_sink())
        if hasattr(p, "compute_outflow_sink"):
            records["outflow_sink"] = float(p.compute_outflow_sink())

        if hasattr(p, "compute_net_flow_all"):
            records["net_flow_all"] = float(p.compute_net_flow_all())

        if isinstance(p, SubCubes):
            records["lower_cube_flux"]     = float(p.compute_lower_cube_flux())
            records["upper_cube_flux"]     = float(p.compute_upper_cube_flux())
            records["lower_cube_flux_in"]  = float(p.compute_lower_cube_flux_in())
            records["lower_cube_flux_out"] = float(p.compute_lower_cube_flux_out())
            records["upper_cube_flux_in"]  = float(p.compute_upper_cube_flux_in())
            records["upper_cube_flux_out"] = float(p.compute_upper_cube_flux_out())

        if self.out_dir and records:
            CsvWriter.write_dict(os.path.join(self.out_dir, csv_name), records)
        return records

    def dispose(self):
        try:
            for name in ("uh3d", "uh1d", "velocity"):
                if hasattr(self.problem, name):
                    setattr(self.problem, name, None)
        except Exception:
            pass
        self.problem = None
        gc.collect()

    def __del__(self):
        self.dispose()
