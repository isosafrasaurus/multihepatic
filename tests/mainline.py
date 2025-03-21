import sys, os, numpy as np
from graphnics import FenicsGraph

WORK_PATH = "./"
SOURCE_PATH = os.path.join(WORK_PATH, "src")
sys.path.append(SOURCE_PATH)

import tissue, fem, visualize

TEST_GRAPH = FenicsGraph()
TEST_GRAPH_NODES = {
    0: [0.000, 0.020, 0.015],
    1: [0.010, 0.020, 0.015],
    2: [0.022, 0.013, 0.015],
    3: [0.022, 0.028, 0.015],
    4: [0.015, 0.005, 0.015],
    5: [0.015, 0.035, 0.015],
    6: [0.038, 0.005, 0.015],
    7: [0.038, 0.035, 0.015]
}
for node_id, pos in TEST_GRAPH_NODES.items():
    TEST_GRAPH.add_node(node_id, pos=pos)
TEST_GRAPH_EDGES = [
    (0, 1, 0.004),
    (1, 2, 0.003),
    (1, 3, 0.003),
    (2, 4, 0.002),
    (2, 6, 0.003),
    (3, 5, 0.002),
    (3, 7, 0.003)
]
for (u, v, radius) in TEST_GRAPH_EDGES:
    TEST_GRAPH.add_edge(u, v, radius=radius)

Omega_bounds = np.array([[0, 0, 0], [0.05, 0.04, 0.03]])
mesh_builder = tissue.MeshBuild(
    TEST_GRAPH,
    Omega_bounds=Omega_bounds,
    Omega_mesh_voxel_dim=(16, 16, 16),
    Lambda_num_nodes_exp=3
)

sink_face = mesh_builder.get_Omega_axis_plane("left")

measure_builder = tissue.MeasureBuild(
    mesh_build=mesh_builder,
    Lambda_inlet=[0],  # We'll say node_id=0 is an inlet
    Omega_sink=sink_face
)

_pos = np.array(TEST_GRAPH.nodes[7]['pos'])
_offset = np.array([0.005, 0.005, 0.005])
_upper_cube_bounds_val = [_pos - _offset, _pos + _offset]

CUBES_TEST = fem.SubCubes(
    domain = measure_builder,
    gamma = 1.0,
    gamma_R = 1.0e-10,
    gamma_a = 1.0e-12,
    mu = 1.0e-3, # Viscosity
    k_t = 1.0e-10, # Tissue permeability in 3D
    k_v = 4.0e-11, # Vessel permeability in 1D
    P_in = 100 * 133.322,
    p_cvp = 1.0 * 133.322,
    lower_cube_bounds = [[0.000,0.000,0.000],[0.01, 0.01, 0.01]],
    upper_cube_bounds = _upper_cube_bounds_val
)

CUBES_TEST_2 = fem.SubCubes(
    domain = measure_builder,
    gamma = 1.0,
    gamma_R = 1.0e-10,
    gamma_a = 1.0e-12,
    mu = 1.0e-3, # Viscosity
    k_t = 1.0e-10, # Tissue permeability in 3D
    k_v = 4.0e-11, # Vessel permeability in 1D
    P_in = 100 * 133.322,
    p_cvp = 1.0 * 133.322,
    lower_cube_bounds = [[0.000,0.000,0.000],[0.01, 0.01, 0.01]],
    upper_cube_bounds = _upper_cube_bounds_val
)


CUBES_TEST_3 = fem.SubCubes(
    domain = measure_builder,
    gamma = 1.0,
    gamma_R = 1.0e-10,
    gamma_a = 1.0e-12,
    mu = 1.0e-3, # Viscosity
    k_t = 1.0e-10, # Tissue permeability in 3D
    k_v = 4.0e-11, # Vessel permeability in 1D
    P_in = 100 * 133.322,
    p_cvp = 1.0 * 133.322,
    lower_cube_bounds = [[0.000,0.000,0.000],[0.01, 0.01, 0.01]],
    upper_cube_bounds = _upper_cube_bounds_val
)

print(f"Total Outflow (m^3/s): {CUBES_TEST.compute_outflow_all()}")
print(f"Total Outflow (m^3/s): {CUBES_TEST.compute_outflow_sink()}")
print(f"Net flux through lower: {CUBES_TEST.compute_lower_cube_flux()}")

print(f"Inflow through lower: {CUBES_TEST.compute_lower_cube_flux_in()}")
print(f"Outflow through lower: {CUBES_TEST.compute_lower_cube_flux_out()}")
print(f"Inflow through upper: {CUBES_TEST.compute_upper_cube_flux_in()}")
print(f"Outflow through upper: {CUBES_TEST.compute_upper_cube_flux_out()}")