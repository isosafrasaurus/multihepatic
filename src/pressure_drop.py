def plot_path_pressure(fem_sink, path):
    node_ids = fem_sink.domain.get_cells_along_path(path)
    mesh = fem_sink.uh1d.function_space().mesh()
    coords = mesh.coordinates()
    pressure = fem_sink.uh1d.compute_vertex_values(mesh)
    path_coords, path_pressure = coords[node_ids], pressure[node_ids]
    
    print("Node Pressure Data:")
    for n, p in zip(node_ids, path_pressure):
        print(f"Node {n}: Pressure {p:.1f}")
        
    cum_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1))))
    fig = plt.figure(figsize=(8, 6))
    plt.plot(cum_dist, path_pressure, marker='o', markersize=2)
    plt.xlabel('Cumulative Distance Along Path (m)')
    plt.ylabel('1D Pressure (Pa)')
    plt.title('Pressure Along Path')
    plt.grid(True)
    return fig