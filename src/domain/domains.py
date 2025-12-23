from __future__ import annotations

import gc
from typing import List, Tuple

import numpy as np
from dolfin import Mesh
from graphnics import FenicsGraph

from .io import vtk_to_graph, vtk_to_mesh
from .mesh import build_mesh_by_counts, build_mesh_by_spacing


class Domain1D:
    def __init__(
            self,
            graph: FenicsGraph,
            edge_resolution_exp: int = 1,
            inlet_node_idxs: List[int] = None,
    ) -> None:
        self.graph = graph
        self.edge_resolution_exp = edge_resolution_exp
        self.inlet_nodes = list(inlet_node_idxs) if inlet_node_idxs else None
        self.graph.make_mesh(self.edge_resolution_exp)

    @classmethod
    def from_vtk(
            cls,
            path: str,
            edge_resolution_exp: int = 1,
            inlet_node_idxs: List[int] = None,
            radius_field: str = "Radius",
    ) -> Domain1D:
        graph = vtk_to_graph(path, radius_field=radius_field)
        dom = cls(graph, edge_resolution_exp=edge_resolution_exp, inlet_node_idxs=inlet_node_idxs)

        if (not all(("tangent" in dom.graph.edges[e]) for e in dom.graph.edges) and
                hasattr(dom.graph, "compute_tangents")):
            dom.graph.compute_tangents()

        return dom

    @property
    def mesh(self) -> Mesh:
        return self.graph.mesh

    def close(self) -> None:
        try:
            for e in list(self.graph.edges):
                self.graph.edges[e].pop("submesh", None)
                self.graph.edges[e].pop("tangent", None)
        except Exception:
            pass
        if hasattr(self.graph, "mesh"):
            self.graph.mesh = None
        gc.collect()

    def __enter__(self) -> Domain1D:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class Domain3D:
    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    @classmethod
    def from_graph(
            cls,
            G: FenicsGraph,
            bounds=None,
            voxel_res: float = None,
            voxel_dim: Tuple[int, int, int] = (16, 16, 16),
            padding: float = 8e-3,
    ) -> Domain3D:
        if voxel_res is not None:
            Omega, bounds_out = build_mesh_by_spacing(
                G,
                spacing_m=float(voxel_res),
                bounds=bounds,
                padding_m=padding,
            )
        else:
            Omega, bounds_out = build_mesh_by_counts(
                G,
                counts=tuple(int(v) for v in voxel_dim),
                bounds=bounds,
                padding_m=padding,
            )
        return cls(Omega)

    @classmethod
    def from_vtk(cls, filename: str) -> Domain3D:
        Omega = vtk_to_mesh(filename)
        coords = Omega.coordinates()
        lower = np.min(coords, axis=0)
        upper = np.max(coords, axis=0)
        return cls(Omega)

    def close(self) -> None:
        self.mesh = None
        gc.collect()

    def __enter__(self) -> Domain3D:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["Domain1D", "Domain3D"]
