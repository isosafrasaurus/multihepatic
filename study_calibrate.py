#!/usr/bin/env python3


import os
import sys
import json
import time
from pathlib import Path
import numpy as np

import warnings
warnings.filterwarnings("ignore",
    message="The cbc.block repository has moved",
    category=UserWarning
)







TARGET_FLOW   = 5.0e-6           
VOXEL_RES     = 1.0e-3           
LAMBDA_EXP    = 5                


X0_EXP        = [6.0, 6.0, 6.0]  
STEP_EXP      = [1.0, 1.0, 1.0]  
TOL           = 1e-10            
MAX_ITERS     = 50               
MAX_FEVAL     = 200              
ECHO_SIMPLEX  = True             


MU            = 1.0e-3
K_T           = 1.0e-10
P_IN          = 100.0 * 133.322
P_CVP         =   1.0 * 133.322


MESH_CACHE_DIR = os.environ.get("MESH_CACHE_DIR", os.path.join(os.environ.get("SCRATCH", "/tmp"), "3d-1d", ".mesh_cache"))
LAMBDA_H5      = os.path.join(MESH_CACHE_DIR, "lambda_1d.h5")
LAMBDA_MF_NPY  = os.path.join(MESH_CACHE_DIR, "lambda_1d_mf.npy")
CACHE_TIMEOUTS = dict(create=180, wait=180)  






PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graphnics import FenicsGraph
from dolfin import (
    MPI, set_log_level, LogLevel,
    FacetNormal, Measure, dot, assemble
)
from dolfin import MPI, HDF5File, Mesh, MeshFunction

from tissue import AxisPlane
from src import Domain1D, Domain3D, Simulation, release_solution
from src.problem import PressureVelocityProblem
from src.composition import Parameters

try:
    from scipy.optimize import minimize
except Exception as e:
    raise SystemExit(
        "[FATAL] SciPy is required for this script inside the container."
    ) from e






TEST_NUM_NODES_EXP = LAMBDA_EXP

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


from dolfin import MPI, HDF5File, Mesh, MeshFunction  


_MESH_CACHE_DIR = os.path.join(PROJECT_ROOT, ".mesh_cache")
_LAMBDA_H5 = os.path.join(_MESH_CACHE_DIR, "lambda_1d.h5")
_LAMBDA_MF_NPY = os.path.join(_MESH_CACHE_DIR, "lambda_1d_mf.npy")

def _log_rank0(msg: str):
    if MPI.rank(MPI.comm_world) == 0:
        print(msg, flush=True)

from dolfin import MPI, Mesh, MeshPartitioning
from graphnics import FenicsGraph



def _serial_build_and_cache_1d(G: FenicsGraph) -> None:
    
    
    try:
        G.make_mesh(num_nodes_exp=TEST_NUM_NODES_EXP)
    except TypeError:
        G.make_mesh(n=TEST_NUM_NODES_EXP)

    
    if hasattr(G, "make_submeshes"):
        G.make_submeshes()
    if hasattr(G, "compute_tangents"):
        G.compute_tangents()

    os.makedirs(MESH_CACHE_DIR, exist_ok=True)
    
    with HDF5File(MPI.comm_self, LAMBDA_H5, "w") as h5:
        h5.write(G.mesh, "/mesh")
        mf = getattr(G, "mf", None)
        if isinstance(mf, MeshFunction):
            h5.write(mf, "/mf")
        elif mf is not None:
            
            np.save(LAMBDA_MF_NPY, np.asarray(mf, dtype=np.int32))





from dolfin import (MPI, Mesh, MeshEditor, MeshFunction, HDF5File,
                    refine, adapt)

def _build_1d_mesh_serial_from_graph(G, nref):
    
    import numpy as np

    
    node_ids = list(G.nodes())
    node_to_local = {nid: i for i, nid in enumerate(node_ids)}

    
    vertex_coords = np.asarray([G.nodes[n]["pos"] for n in node_ids], dtype=float)
    cells_array   = np.asarray([[node_to_local[u], node_to_local[v]] for (u, v) in G.edges()],
                               dtype=np.int64)

    
    mesh = Mesh(MPI.comm_self)
    editor = MeshEditor()
    geom_dim = len(vertex_coords[0])
    editor.open(mesh, "interval", 1, geom_dim)
    editor.init_vertices(vertex_coords.shape[0])
    editor.init_cells(cells_array.shape[0])

    for i, xi in enumerate(vertex_coords):
        editor.add_vertex(i, xi)
    for i, c in enumerate(cells_array):
        editor.add_cell(i, c.tolist())
    editor.close()

    
    mf = MeshFunction("size_t", mesh, 1)
    mf.array()[:] = range(len(G.edges()))

    
    for _ in range(nref):
        mesh = refine(mesh)
        mf   = adapt(mf, mesh)

    return mesh, mf


def build_test_graph() -> FenicsGraph:
    import numpy as np
    from graphnics import FenicsGraph

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    size = MPI.size(comm)

    
    G = FenicsGraph()
    for nid, pos in TEST_GRAPH_NODES.items():
        G.add_node(nid, pos=pos)
    for (u, v, r) in TEST_GRAPH_EDGES:
        G.add_edge(u, v, radius=r)

    try:
        if size == 1:
            
            mesh, mf = _build_1d_mesh_serial_from_graph(G, TEST_NUM_NODES_EXP)
            G.mesh, G.mf = mesh, mf
        else:
            
            if rank == 0:
                mesh0, mf0 = _build_1d_mesh_serial_from_graph(G, TEST_NUM_NODES_EXP)
                os.makedirs(MESH_CACHE_DIR, exist_ok=True)
                with HDF5File(MPI.comm_self, LAMBDA_H5, "w") as h5:
                    h5.write(mesh0, "/mesh")
                    h5.write(mf0,   "/mf")
            MPI.barrier(comm)

            
            mesh = Mesh(comm)
            with HDF5File(comm, LAMBDA_H5, "r") as h5:
                
                h5.read(mesh, "/mesh", False)

            
            mf = MeshFunction("size_t", mesh, 1)
            with HDF5File(comm, LAMBDA_H5, "r") as h5:
                h5.read(mf, "/mf")

            
            try:
                mesh.init(1)  
            except Exception:
                pass

            G.mesh, G.mf = mesh, mf

        
        
        
        
        for (u, v) in G.edges():
            t = np.asarray(G.nodes[v]["pos"], float) - np.asarray(G.nodes[u]["pos"], float)
            t /= np.linalg.norm(t)
            G.edges[u, v]["tangent"] = t

    except Exception as e:
        if rank == 0:
            print(f"[FATAL] 1D mesh build failed: {e}", flush=True)
        sys.exit(4)

    MPI.barrier(comm)
    return G

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
    
    comm = MPI.comm_world
    size = MPI.size(comm)
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


def main():
    set_log_level(LogLevel.ERROR)  

    sanity_check_parallel()

    comm = MPI.comm_world
    rank = MPI.rank(comm)
    size = MPI.size(comm)

    rprint("=== initialize.py (SciPy Nelder–Mead; no CLI) ===")
    rprint(f"[MPI] size={size} rank={rank}")
    rprint(f"Target net flow      : {TARGET_FLOW:g}")
    rprint(f"3D voxel resolution  : {VOXEL_RES:g} m")
    rprint(f"1D mesh exponent     : {LAMBDA_EXP}")
    rprint(f"log10 initial guess  : gamma={X0_EXP[0]}, gamma_a={X0_EXP[1]}, gamma_R={X0_EXP[2]}")
    rprint(f"log10 simplex steps  : d=[{STEP_EXP[0]}, {STEP_EXP[1]}, {STEP_EXP[2]}]")
    rprint(f"NM tolerances        : tol={TOL:g}, max_iters={MAX_ITERS}, max_feval={MAX_FEVAL}")

    
    G = build_test_graph()
    inlet_nodes = [0]  

    Lambda = Domain1D(G, Lambda_num_nodes_exp=LAMBDA_EXP, inlet_nodes=inlet_nodes)
    Omega  = Domain3D.from_graph(G, voxel_res=VOXEL_RES)

    lower = np.array(Omega.bounds[0], dtype=float)
    sink_plane = AxisPlane(axis=2, coordinate=float(lower[2]))  

    
    with Simulation(
        Lambda,
        Omega,
        problem_cls=PressureVelocityProblem,
        inlet_nodes=inlet_nodes,
        Omega_sink_subdomain=sink_plane,
        order=2,
    ) as sim:

        n = FacetNormal(Omega.Omega)
        ds_total = Measure("ds", domain=Omega.Omega)

        eval_counter = {"n": 0}
        t_start = time.time()

        def objective(log10_params: np.ndarray) -> float:
            
            eval_counter["n"] += 1
            t0 = time.time()

            gamma, gamma_a, gamma_R = (10.0 ** log10_params).tolist()
            params = Parameters(
                gamma=float(gamma),
                gamma_a=float(gamma_a),
                gamma_R=float(gamma_R),
                mu=MU,
                k_t=K_T,
                P_in=P_IN,
                P_cvp=P_CVP,
            )

            sol = None
            try:
                sol = sim.solve(params)
                flux = float(assemble(dot(sol.v3d, n) * ds_total))
            finally:
                release_solution(sol)

            misfit = abs(flux - TARGET_FLOW)

            if is_master():
                print(
                    f"[eval {eval_counter['n']:04d}] "
                    f"log10=[{log10_params[0]:8.4f},{log10_params[1]:8.4f},{log10_params[2]:8.4f}]  "
                    f"gamma=[{gamma: .3e},{gamma_a: .3e},{gamma_R: .3e}]  "
                    f"flux={flux: .6e}  |Δ|={misfit: .3e}  "
                    f"dt={time.time()-t0:5.2f}s",
                    flush=True,
                )
            return misfit

        
        x0 = np.array(X0_EXP, dtype=float)
        step = np.array(STEP_EXP, dtype=float)
        initial_simplex = np.vstack([x0, *(x0 + np.eye(3) * step)])

        if ECHO_SIMPLEX:
            rprint("\nInitial simplex (log10 space):")
            if is_master():
                for i, xi in enumerate(initial_simplex):
                    vals = ", ".join(f"{v:8.4f}" for v in xi)
                    print(f"  S[{i}] = [{vals}]")
                print("Evaluating initial simplex...", flush=True)
            
            for i, xi in enumerate(initial_simplex):
                val = objective(xi)  
                if is_master():
                    print(f"  f(S[{i}]) = {val: .6e}", flush=True)
            rprint("End initial simplex evaluation.\n")

        
        iter_box = {"k": 0, "t": time.time()}
        def nm_callback(xk, convergence=None):
            iter_box["k"] += 1
            now = time.time()
            if is_master():
                print(
                    f"[iter {iter_box['k']:03d}] best_log10=[{xk[0]:8.4f},{xk[1]:8.4f},{xk[2]:8.4f}]  "
                    f"wall={(now - iter_box['t']):6.2f}s since last",
                    flush=True,
                )
            iter_box["t"] = now

        
        rprint("Starting SciPy Nelder–Mead optimization...")
        result = minimize(
            objective,
            x0=x0,
            method="Nelder-Mead",
            callback=nm_callback,
            options=dict(
                xatol=TOL,
                fatol=TOL,
                maxiter=MAX_ITERS,
                maxfev=MAX_FEVAL,
                initial_simplex=initial_simplex,
                adaptive=True,
                disp=False,  
            ),
        )
        rprint("Optimization finished.")

        
        x_best = np.array(result.x, dtype=float)
        gamma_opt, gamma_a_opt, gamma_R_opt = (10.0 ** x_best).tolist()

        
        final_params = Parameters(
            gamma=float(gamma_opt),
            gamma_a=float(gamma_a_opt),
            gamma_R=float(gamma_R_opt),
            mu=MU,
            k_t=K_T,
            P_in=P_IN,
            P_cvp=P_CVP,
        )
        sol = sim.solve(final_params)
        final_flux = float(assemble(dot(sol.v3d, n) * ds_total))
        release_solution(sol)

    
    Lambda.dispose()
    Omega.dispose()

    
    if is_master():
        wall = time.time() - t_start
        print("\n=== Optimization summary ===")
        print(f"  success       : {getattr(result, 'success', False)}")
        print(f"  message       : {getattr(result, 'message', '')}")
        print(f"  iterations    : {getattr(result, 'nit', 'n/a')}")
        print(f"  evals         : {getattr(result, 'nfev', 'n/a')}")
        print(f"  best |Δ|      : {float(result.fun):.6e}")
        print("  best params   :")
        print(f"    gamma       = {gamma_opt:.9e}")
        print(f"    gamma_a     = {gamma_a_opt:.9e}")
        print(f"    gamma_R     = {gamma_R_opt:.9e}")
        print(f"  achieved flux : {final_flux:.9e}  (target {TARGET_FLOW:.9e})")
        print(f"  total walltime: {wall:.2f}s")

        results_dir = Path("./results")
        results_dir.mkdir(parents=True, exist_ok=True)

        out_json = {
            "success": bool(result.success),
            "message": str(result.message),
            "nit": int(getattr(result, "nit", -1) or -1),
            "nfev": int(getattr(result, "nfev", -1) or -1),
            "best_abs_error": float(result.fun),
            "best_log10": [float(v) for v in x_best],
            "gamma": float(gamma_opt),
            "gamma_a": float(gamma_a_opt),
            "gamma_R": float(gamma_R_opt),
            "achieved_flux": float(final_flux),
            "target_flow": float(TARGET_FLOW),
            "voxel_res": float(VOXEL_RES),
            "lambda_exp": int(LAMBDA_EXP),
            "x0_exp": [float(v) for v in X0_EXP],
            "step_exp": [float(v) for v in STEP_EXP],
            "tol": float(TOL),
            "max_iters": int(MAX_ITERS),
            "max_feval": int(MAX_FEVAL),
            "mu": float(MU),
            "k_t": float(K_T),
            "P_in": float(P_IN),
            "P_cvp": float(P_CVP),
        }
        with open(results_dir / "calibration_result.json", "w") as f:
            json.dump(out_json, f, indent=2)
        np.savez(
            results_dir / "calibration_result.npz",
            best_log10=x_best.astype(float),
            best_params=np.array([gamma_opt, gamma_a_opt, gamma_R_opt], dtype=float),
            achieved_flux=np.array([final_flux], dtype=float),
        )
        print(f"Saved results to {results_dir}/calibration_result.json and .npz")
        print("Done.")


if __name__ == "__main__":
    main()

