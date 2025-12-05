import gc
from typing import List, Optional, Tuple

import numpy as np
from graphnics import FenicsGraph
from dolfin import Mesh

from tissue.domain import build_mesh_by_counts, build_mesh_by_spacing
from tissue.meshing import get_fg_from_vtk, mesh_from_vtk


class Domain1D:
    def __init__(
            self,
            G: FenicsGraph,
            *,
            Lambda_num_nodes_exp: int = 5,
            inlet_nodes: Optional[List[int]] = None,
    ) -> None:
        self.G = G
        self.Lambda_num_nodes_exp = Lambda_num_nodes_exp
        self.inlet_nodes = list(inlet_nodes) if inlet_nodes else None

    @classmethod
    def from_vtk(
            cls,
            filename: str,
            *,
            Lambda_num_nodes_exp: int = 5,
            inlet_nodes: Optional[List[int]] = None,
            radius_field: str = "Radius",
    ) -> "Domain1D":
        G = get_fg_from_vtk(filename, radius_field=radius_field)

        if not getattr(G, "mesh", None):
            G.make_mesh(n=Lambda_num_nodes_exp)

        if not all(("tangent" in G.edges[e]) for e in G.edges) and hasattr(
                G, "compute_tangents"
        ):
            G.compute_tangents()

        return cls(
            G,
            Lambda_num_nodes_exp=Lambda_num_nodes_exp,
            inlet_nodes=inlet_nodes,
        )

    @property
    def Lambda(self) -> Mesh:
        return self.G.mesh

    def close(self) -> None:
        try:
            for e in list(self.G.edges):
                self.G.edges[e].pop("submesh", None)
                self.G.edges[e].pop("tangent", None)
        except Exception:
            pass
        if hasattr(self.G, "mesh"):
            self.G.mesh = None
        gc.collect()

    
    def __enter__(self) -> "Domain1D":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Domain3D:
    def __init__(
            self,
            Omega: Mesh,
            bounds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.Omega = Omega
        self.bounds = bounds

    @classmethod
    def from_graph(
            cls,
            G: FenicsGraph,
            bounds=None,
            voxel_res: Optional[float] = None,
            voxel_dim: Tuple[int, int, int] = (16, 16, 16),
            padding: float = 8e-3,
            *,
            enforce_graph_in_bounds: bool = False,
    ) -> "Domain3D":
        if voxel_res is not None:
            Omega, bounds_out = build_mesh_by_spacing(
                G,
                spacing_m=float(voxel_res),
                bounds=bounds,
                padding_m=padding,
                strict_bounds=enforce_graph_in_bounds,
            )
        else:
            Omega, bounds_out = build_mesh_by_counts(
                G,
                counts=tuple(int(v) for v in voxel_dim),
                bounds=bounds,
                padding_m=padding,
                strict_bounds=enforce_graph_in_bounds,
            )
        return cls(Omega, bounds_out)

    @classmethod
    def from_vtk(cls, filename: str) -> "Domain3D":
        Omega = mesh_from_vtk(filename)
        coords = Omega.coordinates()
        lower = np.min(coords, axis=0)
        upper = np.max(coords, axis=0)
        return cls(Omega, (lower, upper))

    def close(self) -> None:
        self.Omega = None
        gc.collect()

    def __enter__(self) -> "Domain3D":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
