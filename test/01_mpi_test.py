#!/usr/bin/env python3
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

TEST_NUM_NODES_EXP = 5

TEST_GRAPH_NODES = {
    0: [0.000, 0.020, 0.015],
    1: [0.010, 0.020, 0.015],
    2: [0.022, 0.013, 0.015],
    3: [0.022, 0.028, 0.015],
    4: [0.015, 0.005, 0.015],
    5: [0.015, 0.035, 0.015],
    6: [0.038, 0.005, 0.015],
    7: [0.038, 0.035, 0.015],
}
TEST_GRAPH_EDGES = [
    (0, 1, 0.004),
    (1, 2, 0.003),
    (1, 3, 0.003),
    (2, 4, 0.002),
    (2, 6, 0.003),
    (3, 5, 0.002),
    (3, 7, 0.003),
]

MESH_CACHE_DIR = os.environ.get(
    "MESH_CACHE_DIR",
    os.path.join(os.environ.get("SCRATCH", "/tmp"), "3d-1d", ".mesh_cache"),
)
LAMBDA_H5 = os.path.join(MESH_CACHE_DIR, "lambda_1d.h5")


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
    node_ids = list(G.nodes())
    node_to_local = {nid: i for i, nid in enumerate(node_ids)}

    vertex_coords = np.asarray([G.nodes[n]["pos"] for n in node_ids], dtype=float)
    cells_array = np.asarray(
        [[node_to_local[u], node_to_local[v]] for (u, v) in G.edges()],
        dtype=np.int64,
    )

    mesh = Mesh(MPI.comm_self)
    editor = MeshEditor()
    editor.open(mesh, "interval", 1, vertex_coords.shape[1])  
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


def build_test_graph() -> FenicsGraph:
    import numpy as _np

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    size = MPI.size(comm)

    G = FenicsGraph()
    for node_id, pos in TEST_GRAPH_NODES.items():
        G.add_node(node_id, pos=pos)
    for (u, v, radius) in TEST_GRAPH_EDGES:
        G.add_edge(u, v, radius=radius)

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

        for (u, v) in G.edges():
            t = _np.asarray(G.nodes[v]["pos"], float) - _np.asarray(G.nodes[u]["pos"], float)
            t /= _np.linalg.norm(t)
            G.edges[u, v]["tangent"] = t

    except Exception as e:
        if is_master():
            print(f"[FATAL] 1D mesh build failed: {e}", flush=True)
        sys.exit(4)

    MPI.barrier(MPI.comm_world)
    return G


def main() -> float:
    set_log_level(LogLevel.ERROR)
    sanity_check_parallel()

    G = build_test_graph()

    
    X_ZERO_PLANE = AxisPlane(axis=0, coordinate=0.0)
    bounds = [[0.0, 0.0, 0.0], [0.05, 0.04, 0.03]]

    with Domain1D(G, Lambda_num_nodes_exp=TEST_NUM_NODES_EXP, inlet_nodes=[0]) as Lambda, \
            Domain3D.from_graph(G, bounds=bounds) as Omega, \
            Simulation(
                Lambda,
                Omega,
                problem_cls=PressureVelocityProblem,
                Omega_sink_subdomain=X_ZERO_PLANE,
            ) as sim:
        params = Parameters(
            gamma=3.6145827741262347e-05,
            gamma_a=8.225197366649115e-08,
            gamma_R=8.620057937882969e-08,
            mu=1.0e-3,
            k_t=1.0e-10,
            P_in=100.0 * 133.322,
            P_cvp=1.0 * 133.322,
        )

        
        sol = sim.solve(params)

        
        n = FacetNormal(Omega.Omega)
        ds = Measure("ds", domain=Omega.Omega)
        net_flow_all = float(assemble(dot(sol.v3d, n) * ds))

        release_solution(sol)

    MPI.barrier(MPI.comm_world)
    rprint("net_flow_all =", net_flow_all)
    return net_flow_all


if __name__ == "__main__":
    main()
