from src.core import Domain1D, Domain3D, PhysicsParams, Simulation
from src.fem import Velo

json_dir = "/path/to/jsons"
out_dir  = "/tmp/run1"

with Domain1D.from_json(json_dir, Lambda_num_nodes_exp=5, inlet_nodes=[0]) as Lambda:
    with Domain3D.from_graph(Lambda.G, voxel_res=1e-3, padding=8e-3) as Omega:
        params = PhysicsParams(
            gamma=1.0, gamma_a=1.0, gamma_R=0.0,
            mu=3.5e-3, k_t=1.0e-14, P_in=1000.0, P_cvp=0.0
        )
        sim = Simulation(Lambda, Omega, problem_cls=Velo, out_dir=out_dir)

        
        result = sim.solve(params, save_pvd=True)

        
        sim.save_path_pressure_csv(path=[0, 1, 2, 5, 9], csv_name="path_pressure.csv")
        sim.save_fluxes_csv(csv_name="fluxes.csv")

        
        result.free_fields()
        sim.dispose()
