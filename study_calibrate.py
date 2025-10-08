#!/usr/bin/env python3
import os
import sys
import json
import time
from pathlib import Path
import numpy as np

TARGET_FLOW = 5.0e-6
VOXEL_RES = 1.0e-3
LAMBDA_EXP = 5
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results')


X0_EXP = [6.0, 6.0, 6.0]
STEP_EXP = [1.0, 1.0, 1.0]
TOL = 1e-10
MAX_ITERS = 50
MAX_FEVAL = 200
ECHO_SIMPLEX = True


MU = 1.0e-3
K_T = 1.0e-10
P_IN = 100.0 * 133.322
P_CVP = 1.0 * 133.322


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graphnics import FenicsGraph
from dolfin import MPI, set_log_level, LogLevel, FacetNormal, Measure, dot, assemble

from tissue import AxisPlane
from src import Domain1D, Domain3D, Simulation, release_solution
from src.problem import PressureVelocityProblem
from src.composition import Parameters

try:
    from scipy.optimize import minimize
except Exception as e:
    raise SystemExit(
        "[FATAL] SciPy is required for this script. "
        "Install it in the container/image."
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

def build_test_graph() -> FenicsGraph:
    G = FenicsGraph()
    for node_id, pos in TEST_GRAPH_NODES.items():
        G.add_node(node_id, pos=pos)
    for (u, v, radius) in TEST_GRAPH_EDGES:
        G.add_edge(u, v, radius=radius)
    try:
        G.make_mesh(num_nodes_exp=TEST_NUM_NODES_EXP)
    except TypeError:
        G.make_mesh(n=TEST_NUM_NODES_EXP)
    G.make_submeshes()
    if hasattr(G, "compute_tangents"):
        G.compute_tangents()
    return G

def rprint(*args, **kwargs):
    comm = MPI.comm_world
    if MPI.rank(comm) == 0:
        print(*args, **kwargs)
        sys.stdout.flush()

def main():
    comm = MPI.comm_world
    rank = MPI.rank(comm)
    size = MPI.size(comm)
    set_log_level(LogLevel.ERROR)

    rprint("=== initialize.py ===")
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

            try:
                sol = sim.solve(params)
                flux = float(assemble(dot(sol.v3d, n) * ds_total))
            finally:
                release_solution(locals().get("sol"))

            misfit = abs(flux - TARGET_FLOW)

            if rank == 0:
                print(
                    f"[eval {eval_counter['n']:04d}] "
                    f"log10=[{log10_params[0]:8.4f},{log10_params[1]:8.4f},{log10_params[2]:8.4f}]  "
                    f"gamma=[{gamma: .3e},{gamma_a: .3e},{gamma_R: .3e}]  "
                    f"flux={flux: .6e}  |delta|={misfit: .3e}  "
                    f"dt={time.time()-t0:5.2f}s",
                    flush=True,
                )
            return misfit

        
        x0 = np.array(X0_EXP, dtype=float)
        step = np.array(STEP_EXP, dtype=float)
        initial_simplex = np.vstack([x0, *(x0 + np.eye(3) * step)])

        if ECHO_SIMPLEX and rank == 0:
            rprint("\nInitial simplex (log10 space):")
            for i, xi in enumerate(initial_simplex):
                vals = ", ".join(f"{v:8.4f}" for v in xi)
                print(f"  S[{i}] = [{vals}]")
            rprint("Evaluating initial simplex...")
            for i, xi in enumerate(initial_simplex):
                val = objective(xi)
                print(f"  f(S[{i}]) = {val: .6e}")
            rprint("End initial simplex evaluation.\n")

        
        iter_box = {"k": 0, "t": time.time()}
        def nm_callback(xk, convergence=None):
            iter_box["k"] += 1
            now = time.time()
            if rank == 0:
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

    
    if MPI.rank(MPI.comm_world) == 0:
        wall = time.time() - t_start
        print("\n=== Optimization summary ===")
        print(f"  success       : {getattr(result, 'success', False)}")
        print(f"  message       : {getattr(result, 'message', '')}")
        print(f"  iterations    : {getattr(result, 'nit', 'n/a')}")
        print(f"  evals         : {getattr(result, 'nfev', 'n/a')}")
        print(f"  best |delta|  : {float(result.fun):.6e}")
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

