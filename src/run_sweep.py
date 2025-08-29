from src.core import Domain1D, Domain3D, PhysicsParams, Simulation
from src.fem import Sink

json_dir = "/path/to/jsons"
out_root = "/tmp/sweep"

gammas = [0.5, 1.0, 2.0]
mus    = [3.0e-3, 3.5e-3, 4.0e-3]

for gi, gamma in enumerate(gammas):
    for mi, mu in enumerate(mus):
        out_dir = f"{out_root}/g{gi}_m{mi}"
        with Domain1D.from_json(json_dir, Lambda_num_nodes_exp=5, inlet_nodes=[0]) as Lambda:
            with Domain3D.from_graph(Lambda.G, voxel_res=1e-3, padding=8e-3) as Omega:
                params = PhysicsParams(gamma, 1.0, 0.0, mu, 1.0e-14, 1500.0, 0.0)
                sim = Simulation(Lambda, Omega, problem_cls=Sink, out_dir=out_dir)

                res = sim.solve(params, save_pvd=False)
                sim.save_fluxes_csv(csv_name="fluxes.csv")  # scalar results for quick QA
                res.free_fields()
                sim.dispose()

