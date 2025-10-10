#!/usr/bin/env python3
"""
Test: partial 1D/3D coupling
- 3D domain: axis-aligned cube [0, 0.05] x [0, 0.04] x [0, 0.03]
- 1D domain: a single vessel segment starting OUTSIDE the cube and ending INSIDE
  (start: x = -0.010 m, end: x = 0.030 m, y=0.020 m, z=0.015 m)

Goal: exercise whether a simulation can proceed when part of the 1D domain
doesn't geometrically overlap the 3D domain.
"""

import os
import sys
import math
import numpy as np

# Ensure the multihepatic package root (this directory) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graphnics import FenicsGraph
from dolfin import (
    MPI, HDF5File, Mesh, MeshFunction, MeshEditor, refine, adapt,
    set_log_level, LogLevel, FacetNormal, Measure, dot, assemble,
)

from tissue import AxisPlane
from src import Domain1D, Domain3D, Simulation, release_solution
from src.problem import PressureVelocityProblem
from src.composition import Parameters

# -----------------------------
# Geometry and graph definition
# -----------------------------

# Straight segment: start OUTSIDE cube (x < 0), end INSIDE cube
V0 = [-0.010, 0.020, 0.015]  # outside
V1 = [ 0.030, 0.020, 0.015]  # inside
RADIUS = 0.0035

# 3D cube bounds
BOUNDS_LO = [0.0, 0.0, 0.0]
BOUNDS_HI = [0.05, 0.04, 0.03]

# Refinement on the 1D mesh (interval) used to create a reasonably fine line mesh
TEST_NUM_NODES_EXP = 5

# Optional HDF5 cache for parallel graph distribution
MESH_CACHE_DIR = os.environ.get(
    "MESH_CACHE_DIR",
    os.path.join(os.environ.get("SCRATCH", "/tmp"), "3d-1d", ".mesh_cache"),
)
LAMBDA_H5 = os.path.join(MESH_CACHE_DIR, "lambda_1seg.h5")


# -----------------------------
# Utilities
# -----------------------------

def is_master() -> bool:
    comm = MPI.comm_world
    size = MPI.size(comm)
    rank = MPI.rank(comm)
    if size > 1:
        return rank == 0
    return os.getenv("SLURM_PROCID", "0") == "0"

def rprint(*args, **kwargs):
    if is_master():
        print(*args, **kwargs)
        sys.stdout.flush()

def sanity_check_parallel() -> None:
    size = MPI.size(MPI.comm_world)
    slurm_ntasks = int(os.getenv("SLURM_NTASKS", "1"))
    if slurm_ntasks > 1 and size == 1:
        msg = [
            "[FATAL] Detected SLURM_NTASKS>1 but MPI size==1 inside the container.",
            "        Your container likely dropped the PMI/SLURM env (e.g., using '--cleanenv').",
            "        Fix the launcher to use:  apptainer exec --mpi   (and remove --cleanenv).",
            "        Example srun:  srun -n ${TASKS} --mpi=pmi2 apptainer exec --mpi IMAGE python script.py",
        ]
        print("\n".join(msg), file=sys.stderr, flush=True)
        sys.exit(2)

def _build_1d_mesh_serial_from_graph(G: FenicsGraph, nref: int):
    """Build a simple interval mesh in 3D for the (possibly polyline) graph."""
    node_ids = list(G.nodes())
    node_to_local = {nid: i for i, nid in enumerate(node_ids)}

    vertex_coords = np.asarray([G.nodes[n]["pos"] for n in node_ids], dtype=float)
    cells_array = np.asarray(
        [[node_to_local[u], node_to_local[v]] for (u, v) in G.edges()],
        dtype=np.int64,
    )

    mesh = Mesh(MPI.comm_self)
    editor = MeshEditor()
    editor.open(mesh, "interval", 1, vertex_coords.shape[1])  # topo dim=1, geom dim=3
    editor.init_vertices(vertex_coords.shape[0])
    editor.init_cells(cells_array.shape[0])
    for i, xi in enumerate(vertex_coords):
        editor.add_vertex(i, xi)
    for i, c in enumerate(cells_array):
        editor.add_cell(i, c.tolist())
    editor.close()

    mf = MeshFunction("size_t", mesh, 1)
    mf.array()[:] = np.arange(cells_array.shape[0], dtype=np.int32)

    for _ in range(max(0, int(nref))):
        mesh = refine(mesh)
        mf = adapt(mf, mesh)

    return mesh, mf

def build_single_segment_graph() -> FenicsGraph:
    """Create a FenicsGraph with one edge from V0 (outside) to V1 (inside)."""
    import numpy as _np

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    size = MPI.size(comm)

    G = FenicsGraph()
    G.add_node(0, pos=V0, radius=RADIUS)
    G.add_node(1, pos=V1, radius=RADIUS)
    G.add_edge(0, 1, radius=RADIUS)

    try:
        if size == 1:
            mesh, mf = _build_1d_mesh_serial_from_graph(G, TEST_NUM_NODES_EXP)
            G.mesh, G.mf = mesh, mf
        else:
            if rank == 0:
                os.makedirs(MESH_CACHE_DIR, exist_ok=True)
                mesh0, mf0 = _build_1d_mesh_serial_from_graph(G, TEST_NUM_NODES_EXP)
                with HDF5File(MPI.comm_self, LAMBDA_H5, "w") as h5w:
                    h5w.write(mesh0, "/mesh")
                    h5w.write(mf0, "/mf")
            MPI.barrier(comm)

            mesh = Mesh(comm)
            with HDF5File(comm, LAMBDA_H5, "r") as h5r:
                h5r.read(mesh, "/mesh", False)

            mf = MeshFunction("size_t", mesh, 1)
            with HDF5File(comm, LAMBDA_H5, "r") as h5r:
                h5r.read(mf, "/mf")

            try:
                mesh.init(1)
            except Exception:
                pass

            G.mesh, G.mf = mesh, mf

        # Tangent for the single edge
        t = _np.asarray(G.nodes[1]["pos"], float) - _np.asarray(G.nodes[0]["pos"], float)
        t /= _np.linalg.norm(t)
        G.edges[0, 1]["tangent"] = t

    except Exception as e:
        if is_master():
            print(f"[FATAL] 1D mesh build failed: {e}", flush=True)
        sys.exit(4)

    MPI.barrier(MPI.comm_world)
    return G

def segment_length_inside_box(p0, p1, lo, hi) -> float:
    """
    Compute the length of the straight segment p(t)=p0 + t*(p1-p0), t in [0,1],
    that lies inside the axis-aligned box [lo, hi].
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    d = p1 - p0

    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-15:
            # Segment parallel to axis; must be within slab to intersect
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return 0.0
        else:
            ta = (lo[i] - p0[i]) / d[i]
            tb = (hi[i] - p0[i]) / d[i]
            tmin, tmax = (ta, tb) if ta <= tb else (tb, ta)
            t0 = max(t0, tmin)
            t1 = min(t1, tmax)
            if t0 > t1:
                return 0.0

    total_len = float(np.linalg.norm(d))
    return max(0.0, (t1 - t0) * total_len)


# -----------------------------
# Main
# -----------------------------

def main() -> float:
    set_log_level(LogLevel.ERROR)
    sanity_check_parallel()

    # Build the 1D vessel that starts outside and ends inside the cube
    G = build_single_segment_graph()

    # Robin sink on the plane x=0 for the 3D boundary
    X_ZERO_PLANE = AxisPlane(axis=0, coordinate=0.0)
    bounds = [BOUNDS_LO, BOUNDS_HI]

    # Diagnostic: how much of the segment is inside the cube?
    inside_len = segment_length_inside_box(V0, V1, BOUNDS_LO, BOUNDS_HI)
    total_len = float(np.linalg.norm(np.asarray(V1) - np.asarray(V0)))
    frac_inside = (inside_len / total_len) if total_len > 0 else 0.0
    rprint(f"[geom] segment length total = {total_len:.6f} m, inside cube = {inside_len:.6f} m ({100*frac_inside:.1f}%)")

    # Inlet at the first node (outside the cube)
    inlet_nodes = [0]

    with Domain1D(G, Lambda_num_nodes_exp=TEST_NUM_NODES_EXP, inlet_nodes=inlet_nodes) as Lambda, \
         Domain3D.from_graph(G, bounds=bounds) as Omega, \
         Simulation(
             Lambda,
             Omega,
             problem_cls=PressureVelocityProblem,
             Omega_sink_subdomain=X_ZERO_PLANE,
         ) as sim:

        params = Parameters(
            gamma=3.6e-05,
            gamma_a=8.2e-08,
            gamma_R=8.6e-08,
            mu=1.0e-3,
            k_t=1.0e-10,
            P_in=100.0 * 133.322,
            P_cvp=1.0 * 133.322,
        )

        # Run the coupled solve
        sol = sim.run(params)

        # Net outward flux across the entire 3D boundary
        n = FacetNormal(Omega.Omega)
        ds = Measure("ds", domain=Omega.Omega)
        net_flow_all = float(assemble(dot(sol.v3d, n) * ds))

        # Cleanup
        release_solution(sol)

    MPI.barrier(MPI.comm_world)
    rprint("net_flow_all =", net_flow_all)
    return net_flow_all


if __name__ == "__main__":
    main()

