from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np
import ufl
from dolfinx import default_scalar_type
from fenicsx_ii import Average, Circle

from .config import Parameters
from .domain import Domain1D, Domain3D


@dataclass(frozen=True, slots=True)
class Measures:
    dx_tissue: ufl.Measure
    dx_network: ufl.Measure
    ds_tissue: ufl.Measure
    ds_network: ufl.Measure
    ds_network_outlet: ufl.Measure


@dataclass(frozen=True, slots=True)
class Spaces:
    tissue_pressure: fem.FunctionSpace
    network_pressure: fem.FunctionSpace
    mixed_pressure: Any  # ufl.MixedFunctionSpace
    cell_radius: fem.FunctionSpace


@dataclass(frozen=True, slots=True)
class PressureForms:
    spaces: Spaces
    measures: Measures

    a: ufl.Form
    L: ufl.Form
    bcs: list[Any]

    # Geometry/radii
    cell_radius: fem.Function
    vertex_radius: fem.Function
    radius_per_cell: np.ndarray
    default_radius: float
    circle_trial: Any
    circle_test: Any

    # Constants/expressions needed for postprocessing
    gamma: fem.Constant
    gamma_a: fem.Constant
    mu_network: fem.Constant
    P_cvp_network: fem.Constant
    D_perimeter_cell: Any
    D_area_vertex: Any


def _default_radius_from_mapping(radius_by_tag: Mapping[int, float] | np.ndarray | None) -> float:
    if radius_by_tag is None:
        return 1.0
    if isinstance(radius_by_tag, np.ndarray):
        return float(np.max(radius_by_tag)) if radius_by_tag.size else 1.0
    return float(max(radius_by_tag.values())) if radius_by_tag else 1.0


def _radius_lookup_array(
        radius_by_tag: Mapping[int, float] | np.ndarray,
        *,
        max_tag: int,
        default_radius: float,
) -> np.ndarray:
    lookup = np.full((max_tag + 2,), float(default_radius), dtype=np.float64)
    if isinstance(radius_by_tag, np.ndarray):
        n = min(radius_by_tag.size, max_tag + 1)
        lookup[:n] = radius_by_tag[:n].astype(np.float64, copy=False)
    else:
        for tag, radius in radius_by_tag.items():
            if tag < 0:
                raise ValueError(f"Radius tags must be nonnegative integers, got tag={tag}")
            if tag <= max_tag:
                lookup[int(tag)] = float(radius)
    return lookup


def build_cell_radius_field(
        mesh_1d: dmesh.Mesh,
        subdomains: dmesh.MeshTags,
        radius_by_tag: Mapping[int, float] | np.ndarray,
        *,
        default_radius: float,
        name: str = "radius_cell",
) -> tuple[fem.Function, np.ndarray]:
    DG0 = fem.functionspace(mesh_1d, ("DG", 0))
    r_cell = fem.Function(DG0)
    r_cell.name = name

    tdim = mesh_1d.topology.dim
    cell_map = mesh_1d.topology.index_map(tdim)
    num_cells_local = int(cell_map.size_local)
    num_cells = num_cells_local + int(cell_map.num_ghosts)

    max_tag_mesh = int(np.max(subdomains.values)) if subdomains.values.size else 0
    max_tag_map = int(radius_by_tag.size - 1) if isinstance(radius_by_tag, np.ndarray) else (
        int(max(radius_by_tag.keys())) if radius_by_tag else 0
    )
    max_tag = max(max_tag_mesh, max_tag_map, 0)
    default_tag = max_tag + 1

    cell_tags = np.full((num_cells,), default_tag, dtype=np.int32)
    cell_tags[subdomains.indices] = subdomains.values

    lookup = _radius_lookup_array(radius_by_tag, max_tag=max_tag, default_radius=default_radius)
    radius_per_cell = lookup[cell_tags]

    r_cell.x.array[:num_cells_local] = radius_per_cell[:num_cells_local].astype(r_cell.x.array.dtype, copy=False)
    r_cell.x.scatter_forward()

    # For DG0, dof ordering usually matches cell index; keep it simple and return cell-indexed array.
    return r_cell, radius_per_cell


def build_boundary_vertex_radius_field(
        V1: fem.FunctionSpace,
        mesh_1d: dmesh.Mesh,
        *,
        cell_radius: fem.Function,
        inlet_vertices: np.ndarray,
        outlet_vertices: np.ndarray,
        default_radius: float,
        name: str = "radius_vertex",
) -> fem.Function:
    r_vertex = fem.Function(V1)
    r_vertex.name = name
    r_vertex.x.array[:] = float(default_radius)

    mesh_1d.topology.create_connectivity(0, 1)
    v2c = mesh_1d.topology.connectivity(0, 1)

    DG0 = cell_radius.function_space

    def radius_at_vertex(v: int) -> float:
        cells = v2c.links(v)
        if len(cells) == 0:
            return float(default_radius)
        c0 = int(cells[0])
        dof = int(DG0.dofmap.cell_dofs(c0)[0])
        return float(cell_radius.x.array[dof])

    # IMPORTANT: exactly one locate_dofs_topological call per marker (inlet/outlet) on each rank.
    inlet_vertices = inlet_vertices.astype(np.int32, copy=False)
    outlet_vertices = outlet_vertices.astype(np.int32, copy=False)

    inlet_dofs = fem.locate_dofs_topological(V1, 0, inlet_vertices)
    outlet_dofs = fem.locate_dofs_topological(V1, 0, outlet_vertices)

    if inlet_vertices.size:
        if len(inlet_dofs) != len(inlet_vertices):
            raise RuntimeError(
                "Expected 1 dof per vertex for scalar Lagrange space on the network. "
                f"Got inlet_dofs={len(inlet_dofs)} but inlet_vertices={len(inlet_vertices)}."
            )
        r_vertex.x.array[inlet_dofs] = np.array(
            [radius_at_vertex(int(v)) for v in inlet_vertices],
            dtype=r_vertex.x.array.dtype,
        )

    if outlet_vertices.size:
        if len(outlet_dofs) != len(outlet_vertices):
            raise RuntimeError(
                "Expected 1 dof per vertex for scalar Lagrange space on the network. "
                f"Got outlet_dofs={len(outlet_dofs)} but outlet_vertices={len(outlet_vertices)}."
            )
        r_vertex.x.array[outlet_dofs] = np.array(
            [radius_at_vertex(int(v)) for v in outlet_vertices],
            dtype=r_vertex.x.array.dtype,
        )

    r_vertex.x.scatter_forward()
    return r_vertex


def make_cell_radius_callable(mesh_1d: dmesh.Mesh, radius_per_cell: np.ndarray, *, default_radius: float):
    from dolfinx import geometry

    tdim = mesh_1d.topology.dim
    tree = geometry.bb_tree(mesh_1d, tdim)

    def radius_fn(xT: np.ndarray) -> np.ndarray:
        points = np.asarray(xT.T, dtype=np.float64)
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)

        candidates = geometry.compute_collisions_points(tree, points)
        colliding = geometry.compute_colliding_cells(mesh_1d, candidates, points)

        out = np.full((points.shape[0],), float(default_radius), dtype=np.float64)
        for i in range(points.shape[0]):
            links = colliding.links(i)
            if len(links) > 0:
                out[i] = float(radius_per_cell[int(links[0])])
        return out

    return radius_fn


def build_pressure_forms(
        tissue: Domain3D,
        network: Domain1D,
        params: Parameters,
        *,
        degree_3d: int = 1,
        degree_1d: int = 1,
        circle_quadrature_degree: int = 20,
        radius_by_tag: Mapping[int, float] | np.ndarray | None = None,
        default_radius: float | None = None,
        cell_radius: fem.Function | None = None,
        vertex_radius: fem.Function | None = None,
) -> PressureForms:
    if default_radius is None:
        default_radius = _default_radius_from_mapping(radius_by_tag)

    V3 = fem.functionspace(tissue.mesh, ("Lagrange", degree_3d))
    V1 = fem.functionspace(network.mesh, ("Lagrange", degree_1d))
    W = ufl.MixedFunctionSpace(V3, V1)
    DG0 = fem.functionspace(network.mesh, ("DG", 0))

    spaces = Spaces(tissue_pressure=V3, network_pressure=V1, mixed_pressure=W, cell_radius=DG0)

    # Cell radius
    if cell_radius is None:
        if network.subdomains is None or radius_by_tag is None:
            raise ValueError("To build cell_radius automatically, provide network.subdomains and radius_by_tag.")
        cell_radius, radius_per_cell = build_cell_radius_field(
            network.mesh,
            network.subdomains,
            radius_by_tag,
            default_radius=float(default_radius),
        )
    else:
        # Ensure ghosts are valid before reading.
        cell_radius.x.scatter_forward()
        tdim = network.mesh.topology.dim
        cell_map = network.mesh.topology.index_map(tdim)
        num_cells = int(cell_map.size_local + cell_map.num_ghosts)
        radius_per_cell = np.empty((num_cells,), dtype=np.float64)
        for c in range(num_cells):
            dof = int(cell_radius.function_space.dofmap.cell_dofs(c)[0])
            radius_per_cell[c] = float(cell_radius.x.array[dof])

    # Vertex radius (inlet/outlet only)
    if vertex_radius is None:
        vertex_radius = build_boundary_vertex_radius_field(
            V1,
            network.mesh,
            cell_radius=cell_radius,
            inlet_vertices=network.inlet_vertices,
            outlet_vertices=network.outlet_vertices,
            default_radius=float(default_radius),
        )

    # Circle quadrature
    radius_fn = make_cell_radius_callable(network.mesh, radius_per_cell, default_radius=float(default_radius))
    circle_trial = Circle(network.mesh, radius=radius_fn, degree=circle_quadrature_degree)
    circle_test = Circle(network.mesh, radius=radius_fn, degree=circle_quadrature_degree)

    # Measures
    dx_tissue = ufl.Measure("dx", domain=tissue.mesh)
    dx_network = ufl.Measure("dx", domain=network.mesh)
    ds_tissue = ufl.Measure("ds", domain=tissue.mesh)
    ds_network = ufl.Measure("ds", domain=network.mesh, subdomain_data=network.boundaries)
    ds_network_outlet = ds_network(network.outlet_marker)
    measures = Measures(dx_tissue=dx_tissue, dx_network=dx_network, ds_tissue=ds_tissue, ds_network=ds_network,
                        ds_network_outlet=ds_network_outlet)

    # Trial/Test
    p_t, P = ufl.TrialFunctions(W)
    v_t, w = ufl.TestFunctions(W)

    p_avg = Average(p_t, circle_trial, V1)
    v_avg = Average(v_t, circle_test, V1)

    # Constants
    k_t = fem.Constant(tissue.mesh, default_scalar_type(params.k_t))
    mu_t = fem.Constant(tissue.mesh, default_scalar_type(params.mu))
    gamma_R = fem.Constant(tissue.mesh, default_scalar_type(params.gamma_R))
    P_cvp_t = fem.Constant(tissue.mesh, default_scalar_type(params.P_cvp))

    mu_n = fem.Constant(network.mesh, default_scalar_type(params.mu))
    gamma = fem.Constant(network.mesh, default_scalar_type(params.gamma))
    gamma_a = fem.Constant(network.mesh, default_scalar_type(params.gamma_a))
    P_cvp_n = fem.Constant(network.mesh, default_scalar_type(params.P_cvp))

    # Geometry expressions
    D_area_cell = ufl.pi * cell_radius ** 2
    D_perimeter_cell = 2.0 * ufl.pi * cell_radius
    k_v_cell = (cell_radius ** 2) / 8.0
    D_area_vertex = ufl.pi * vertex_radius ** 2

    # Weak forms
    a00 = (k_t / mu_t) * ufl.inner(ufl.grad(p_t), ufl.grad(v_t)) * dx_tissue
    a00 += gamma_R * p_t * v_t * ds_tissue
    a00 += gamma * p_avg * v_avg * D_perimeter_cell * dx_network

    a01 = -gamma * P * v_avg * D_perimeter_cell * dx_network
    a01 += -(gamma_a / mu_n) * P * v_avg * D_area_vertex * ds_network_outlet

    a10 = -gamma * p_avg * w * D_perimeter_cell * dx_network

    a11 = (k_v_cell / mu_n) * D_area_cell * ufl.inner(ufl.grad(P), ufl.grad(w)) * dx_network
    a11 += gamma * P * w * D_perimeter_cell * dx_network
    a11 += (gamma_a / mu_n) * P * w * D_area_vertex * ds_network_outlet

    a = a00 + a01 + a10 + a11

    L0 = gamma_R * P_cvp_t * v_t * ds_tissue
    L0 += (gamma_a / mu_n) * P_cvp_n * v_avg * D_area_vertex * ds_network_outlet
    L1 = (gamma_a / mu_n) * P_cvp_n * w * D_area_vertex * ds_network_outlet
    L = L0 + L1

    # BCs
    inlet_vertices = network.inlet_vertices
    inlet_dofs = fem.locate_dofs_topological(V1, 0, inlet_vertices)

    inlet_val = fem.Function(V1)
    inlet_val.x.array[:] = default_scalar_type(params.P_in)
    bc_inlet = fem.dirichletbc(inlet_val, inlet_dofs)
    bcs = [bc_inlet]

    return PressureForms(
        spaces=spaces,
        measures=measures,
        a=a,
        L=L,
        bcs=bcs,
        cell_radius=cell_radius,
        vertex_radius=vertex_radius,
        radius_per_cell=radius_per_cell,
        default_radius=float(default_radius),
        circle_trial=circle_trial,
        circle_test=circle_test,
        gamma=gamma,
        gamma_a=gamma_a,
        mu_network=mu_n,
        P_cvp_network=P_cvp_n,
        D_perimeter_cell=D_perimeter_cell,
        D_area_vertex=D_area_vertex,
    )
