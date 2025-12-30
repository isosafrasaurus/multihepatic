from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import dolfinx.fem as fem
import dolfinx.mesh as dmesh
import numpy as np
import ufl
from dolfinx import default_scalar_type
from fenicsx_ii import Average, Circle

from .core import Parameters
from .domains import Domain1D, Domain3D


@dataclass(slots=True)
class Measures:
    dx_tissue: ufl.Measure
    dx_network: ufl.Measure
    ds_tissue: ufl.Measure
    ds_network: ufl.Measure
    ds_network_outlet: ufl.Measure


@dataclass(slots=True)
class Spaces:
    tissue_pressure: fem.FunctionSpace
    network_pressure: fem.FunctionSpace
    mixed_pressure: Any  # ufl.MixedFunctionSpace
    cell_radius: fem.FunctionSpace


@dataclass(slots=True)
class AssembledForms:
    spaces: Spaces
    measures: Measures

    a: ufl.Form
    L: ufl.Form
    bcs: list[Any]  # fem.DirichletBC

    cell_radius: fem.Function
    vertex_radius: fem.Function
    radius_per_cell: np.ndarray
    default_radius: float

    circle_trial: Any
    circle_test: Any

    gamma: fem.Constant
    gamma_a: fem.Constant
    mu_network: fem.Constant
    P_cvp_network: fem.Constant
    D_perimeter_cell: Any  # ufl.Expr
    D_area_vertex: Any  # ufl.Expr


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
            if tag > max_tag:
                continue
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
    """
    Build DG0 cellwise radius r_cell and return also radius_per_cell.
    """
    radius_space = fem.functionspace(mesh_1d, ("DG", 0))
    r_cell = fem.Function(radius_space)
    r_cell.name = name

    tdim = mesh_1d.topology.dim
    cell_map = mesh_1d.topology.index_map(tdim)
    num_cells_local = cell_map.size_local
    num_cells = num_cells_local + cell_map.num_ghosts

    # Determine the max tag we need to represent.
    max_tag_in_mesh = int(np.max(subdomains.values)) if subdomains.values.size else 0
    if isinstance(radius_by_tag, np.ndarray):
        max_tag_in_mapping = int(radius_by_tag.size - 1)
    else:
        max_tag_in_mapping = int(max(radius_by_tag.keys())) if radius_by_tag else 0
    max_tag = max(max_tag_in_mesh, max_tag_in_mapping, 0)

    default_tag = max_tag + 1
    cell_tags = np.full((num_cells,), default_tag, dtype=np.int32)
    cell_tags[subdomains.indices] = subdomains.values

    lookup = _radius_lookup_array(radius_by_tag, max_tag=max_tag, default_radius=default_radius)
    radius_per_cell = lookup[cell_tags]

    # Assign owned dofs; ghosts filled by scatter.
    r_cell.x.array[:num_cells_local] = radius_per_cell[:num_cells_local].astype(
        r_cell.x.array.dtype, copy=False
    )
    r_cell.x.scatter_forward()

    # DG0 is usually identity cell->dof, but be robust.
    dofmap = radius_space.dofmap
    sample_ok = True
    if num_cells_local > 0:
        sample_cells = np.linspace(
            0, num_cells_local - 1, num=min(5, num_cells_local), dtype=np.int32
        )
        for c in sample_cells:
            dofs = dofmap.cell_dofs(int(c))
            if len(dofs) != 1 or int(dofs[0]) != int(c):
                sample_ok = False
                break

    if sample_ok:
        # Fast path: radius_per_cell matches r_cell dof ordering.
        radius_per_cell_by_cell = r_cell.x.array.copy().astype(np.float64, copy=False)
    else:
        # Fallback: build explicit cell->dof mapping.
        radius_per_cell_by_cell = np.empty((num_cells,), dtype=np.float64)
        for c in range(num_cells):
            dof = int(dofmap.cell_dofs(c)[0])
            radius_per_cell_by_cell[c] = float(r_cell.x.array[dof])

    return r_cell, radius_per_cell_by_cell


def build_boundary_vertex_radius_field(
        network_pressure_space: fem.FunctionSpace,
        mesh_1d: dmesh.Mesh,
        *,
        cell_radius: fem.Function,
        inlet_vertices: np.ndarray,
        outlet_vertices: np.ndarray,
        default_radius: float,
        name: str = "radius_vertex",
) -> fem.Function:
    """
    Build a vertex radius field on the pressure space.
    Default is default_radius; inlet/outlet vertices get the radius of an adjacent cell.
    """
    r_vertex = fem.Function(network_pressure_space)
    r_vertex.name = name
    r_vertex.x.array[:] = float(default_radius)

    # Need vertex->cell adjacency
    mesh_1d.topology.create_connectivity(0, 1)
    v2c = mesh_1d.topology.connectivity(0, 1)

    cell_radius_space = cell_radius.function_space

    def radius_at_vertex(vertex: int) -> float:
        cells = v2c.links(vertex)
        if len(cells) == 0:
            return float(default_radius)
        cell = int(cells[0])
        dof = int(cell_radius_space.dofmap.cell_dofs(cell)[0])
        return float(cell_radius.x.array[dof])

    boundary_vertices = np.unique(np.concatenate([inlet_vertices, outlet_vertices]).astype(np.int32))
    for vtx in boundary_vertices.tolist():
        dofs = fem.locate_dofs_topological(
            network_pressure_space, entity_dim=0, entities=np.array([vtx], dtype=np.int32)
        )
        for dof in dofs.tolist():
            r_vertex.x.array[dof] = radius_at_vertex(int(vtx))

    r_vertex.x.scatter_forward()
    return r_vertex


def make_cell_radius_callable(
        mesh_1d: dmesh.Mesh,
        radius_per_cell: np.ndarray,
        *,
        default_radius: float,
):
    """
    Callable radius(x): used by fenicsx_ii.Circle quadrature.
    Returns a radius for each query point based on the first colliding cell.
    """
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


def build_assembled_forms(
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
) -> AssembledForms:
    """
    Assemble UFL forms and build FE spaces/measures.
    """
    if cell_radius is None:
        if network.subdomains is None:
            raise ValueError(
                "Domain1D.subdomains is required unless `cell_radius` is provided explicitly."
            )
        if radius_by_tag is None:
            raise ValueError(
                "`radius_by_tag` must be provided when building radii from Domain1D.subdomains."
            )

    if default_radius is None:
        if radius_by_tag is not None and not isinstance(radius_by_tag, np.ndarray):
            default_radius = float(max(radius_by_tag.values())) if radius_by_tag else 1.0
        elif isinstance(radius_by_tag, np.ndarray) and radius_by_tag.size:
            default_radius = float(np.max(radius_by_tag))
        else:
            default_radius = 1.0

    tissue_pressure_space = fem.functionspace(tissue.mesh, ("Lagrange", degree_3d))
    network_pressure_space = fem.functionspace(network.mesh, ("Lagrange", degree_1d))
    mixed_space = ufl.MixedFunctionSpace(tissue_pressure_space, network_pressure_space)
    radius_space = fem.functionspace(network.mesh, ("DG", 0))

    spaces = Spaces(
        tissue_pressure=tissue_pressure_space,
        network_pressure=network_pressure_space,
        mixed_pressure=mixed_space,
        cell_radius=radius_space,
    )

    if cell_radius is None:
        cell_radius, radius_per_cell = build_cell_radius_field(
            network.mesh,
            network.subdomains,  # type: ignore[arg-type]
            radius_by_tag,  # type: ignore[arg-type]
            default_radius=float(default_radius),
        )
    else:
        # Derive a best-effort cell-indexed radius array.
        tdim = network.mesh.topology.dim
        cell_map = network.mesh.topology.index_map(tdim)
        num_cells_local = cell_map.size_local
        num_cells = num_cells_local + cell_map.num_ghosts
        radius_per_cell = np.empty((num_cells,), dtype=np.float64)
        for c in range(num_cells):
            dof = int(cell_radius.function_space.dofmap.cell_dofs(c)[0])
            radius_per_cell[c] = float(cell_radius.x.array[dof])

    if vertex_radius is None:
        vertex_radius = build_boundary_vertex_radius_field(
            network_pressure_space,
            network.mesh,
            cell_radius=cell_radius,
            inlet_vertices=network.inlet_vertices,
            outlet_vertices=network.outlet_vertices,
            default_radius=float(default_radius),
        )

    radius_fn = make_cell_radius_callable(
        network.mesh, radius_per_cell, default_radius=float(default_radius)
    )
    circle_trial = Circle(network.mesh, radius=radius_fn, degree=circle_quadrature_degree)
    circle_test = Circle(network.mesh, radius=radius_fn, degree=circle_quadrature_degree)

    dx_tissue = ufl.Measure("dx", domain=tissue.mesh)
    dx_network = ufl.Measure("dx", domain=network.mesh)
    ds_tissue = ufl.Measure("ds", domain=tissue.mesh)

    ds_network = ufl.Measure("ds", domain=network.mesh, subdomain_data=network.boundaries)
    ds_network_outlet = ds_network(network.outlet_marker)

    measures = Measures(
        dx_tissue=dx_tissue,
        dx_network=dx_network,
        ds_tissue=ds_tissue,
        ds_network=ds_network,
        ds_network_outlet=ds_network_outlet,
    )

    p_tissue, p_network = ufl.TrialFunctions(mixed_space)
    v_tissue, w_network = ufl.TestFunctions(mixed_space)

    p_avg = Average(p_tissue, circle_trial, network_pressure_space)
    v_avg = Average(v_tissue, circle_test, network_pressure_space)

    k_t = fem.Constant(tissue.mesh, default_scalar_type(params.k_t))
    mu_tissue = fem.Constant(tissue.mesh, default_scalar_type(params.mu))
    gamma_R = fem.Constant(tissue.mesh, default_scalar_type(params.gamma_R))
    P_cvp_tissue = fem.Constant(tissue.mesh, default_scalar_type(params.P_cvp))

    mu_network = fem.Constant(network.mesh, default_scalar_type(params.mu))
    gamma = fem.Constant(network.mesh, default_scalar_type(params.gamma))
    gamma_a = fem.Constant(network.mesh, default_scalar_type(params.gamma_a))
    P_cvp_network = fem.Constant(network.mesh, default_scalar_type(params.P_cvp))

    D_area_cell = ufl.pi * cell_radius ** 2
    D_perimeter_cell = 2.0 * ufl.pi * cell_radius
    k_v_cell = (cell_radius ** 2) / 8.0
    D_area_vertex = ufl.pi * vertex_radius ** 2

    a00 = (k_t / mu_tissue) * ufl.inner(ufl.grad(p_tissue), ufl.grad(v_tissue)) * dx_tissue
    a00 += gamma_R * p_tissue * v_tissue * ds_tissue
    a00 += gamma * p_avg * v_avg * D_perimeter_cell * dx_network

    a01 = -gamma * p_network * v_avg * D_perimeter_cell * dx_network
    a01 += -(gamma_a / mu_network) * p_network * v_avg * D_area_vertex * ds_network_outlet

    a10 = -gamma * p_avg * w_network * D_perimeter_cell * dx_network

    a11 = (k_v_cell / mu_network) * D_area_cell * ufl.inner(ufl.grad(p_network), ufl.grad(w_network)) * dx_network
    a11 += gamma * p_network * w_network * D_perimeter_cell * dx_network
    a11 += (gamma_a / mu_network) * p_network * w_network * D_area_vertex * ds_network_outlet

    a = a00 + a01 + a10 + a11

    L0 = gamma_R * P_cvp_tissue * v_tissue * ds_tissue
    L0 += (gamma_a / mu_network) * P_cvp_network * v_avg * D_area_vertex * ds_network_outlet
    L1 = (gamma_a / mu_network) * P_cvp_network * w_network * D_area_vertex * ds_network_outlet
    L = L0 + L1

    inlet_dofs = fem.locate_dofs_topological(network_pressure_space, 0, network.inlet_vertices)

    inlet_value = fem.Function(network_pressure_space)
    inlet_value.x.array[:] = default_scalar_type(params.P_in)
    bc_inlet = fem.dirichletbc(inlet_value, inlet_dofs)
    bcs = [bc_inlet]

    return AssembledForms(
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
        mu_network=mu_network,
        P_cvp_network=P_cvp_network,
        D_perimeter_cell=D_perimeter_cell,
        D_area_vertex=D_area_vertex,
    )
