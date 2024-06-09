import matplotlib
import matplotlib.pyplot as plt
import numpy
import plotly
import scipy
import plotly.graph_objects as go
import numpy as np
import scipy.interpolate

def visualize(mesh1d, uh1d, mesh3d=None, uh3d=None, z_level=50, elev=10, azim=30, boundary_list=None):
    node_coords = mesh1d.coordinates()
    pressure_values = uh1d.compute_vertex_values(mesh1d)

    x_min, x_max = node_coords[:, 0].min(), node_coords[:, 0].max()
    y_min, y_max = node_coords[:, 1].min(), node_coords[:, 1].max()
    x_width, y_width = x_max - x_min, y_max - y_min

    num_columns = 2 if (uh3d is not None and mesh3d is not None) else 1
    fig, ax1 = plt.subplots(1, num_columns, figsize=(20, 8) if num_columns == 2 else (10, 8))
    ax1 = fig.add_subplot(121, projection='3d')

    if num_columns == 2:
        pressure_3d_values = uh3d.compute_vertex_values(mesh3d)
        coords_3d = mesh3d.coordinates()

        z_coords = coords_3d[:, 2]
        mask = np.isclose(z_coords, z_level, atol=mesh3d.hmin())
        filtered_coords = coords_3d[mask]
        filtered_pressure = pressure_3d_values[mask]

        # grid for the heatmap
        x = filtered_coords[:, 0]
        y = filtered_coords[:, 1]
        z = filtered_pressure
        grid_size = 200
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        zi = scipy.interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

        fig.get_axes()[0].axis('off')
        fig.get_axes()[1].axis('off')
    else:
        fig.get_axes()[0].axis('off')

    # (1D) 2D plane at Z = z_level
    x_plane, y_plane = np.meshgrid(
        np.linspace(x_min, x_max, num=10),
        np.linspace(y_min, y_max, num=10)
    )
    z_plane = np.full(x_plane.shape, z_level)

    ax1.plot_surface(x_plane, y_plane, z_plane, color='m', alpha=0.3, zorder=10) # ensure plane is in front
    ax1.view_init(elev=elev, azim=azim)
    sc = ax1.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], c=pressure_values, cmap='viridis', s=0.1)
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('1D Pressure')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('1D Domain')

    if boundary_list is not None:
        for point in boundary_list:
            ax1.scatter(*point, color='red')

    # (3D) 2D heatmap subplot
    if num_columns == 2:
        ax2 = fig.add_subplot(122)
        cf = ax2.contourf(xi, yi, zi, levels=100, cmap='viridis')
        cbar = plt.colorbar(cf, ax=ax2)
        cbar.set_label('Pressure')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'3D Domain Section at Z = {z_level}')

    plt.gca().set_aspect('equal')
    plt.show()

def visualize_scatter(mesh1d, uh1d, z_level=50, boundary_list=None):
    node_coords = mesh1d.coordinates()
    pressure_values = uh1d.compute_vertex_values(mesh1d)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=node_coords[:, 0], y=node_coords[:, 1], z=node_coords[:, 2],
                              mode='markers',
                              marker=dict(size=2, color=pressure_values, colorscale='Viridis', colorbar=dict(title='1D Pressure')), hovertext=[f"Pressure: {val:.4f}" for val in pressure_values],
                              name=''))

    # Plane at Z = z_level
    x_min, x_max, y_min, y_max = node_coords[:, 0].min(), node_coords[:, 0].max(), node_coords[:, 1].min(), node_coords[:, 1].max()
    x_plane, y_plane = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    z_plane = z_level * np.ones_like(x_plane)

    fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_plane, opacity=0.3, name=f'Plane at Z={z_level}'))

    # Boundary points
    if boundary_list:
        fig.add_traces([go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], showlegend = False, mode='markers', marker=dict(size=5, color='red')) for p in boundary_list])

    # Set camera view
    fig.update_layout(
        scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=-1.25, y=-1.25, z=1.25)),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=1000, height=800
    )

    fig.show()

def visualize_contour(mesh3d, uh3d, z_level=50):
    if uh3d is not None and mesh3d is not None:
        pressure_3d_values = uh3d.compute_vertex_values(mesh3d)
        coords_3d = mesh3d.coordinates()

        z_coords = coords_3d[:, 2]
        mask = np.isclose(z_coords, z_level, atol=mesh3d.hmin())
        filtered_coords = coords_3d[mask]
        filtered_pressure = pressure_3d_values[mask]

        x = filtered_coords[:, 0]
        y = filtered_coords[:, 1]
        z = filtered_pressure
        grid_size = 200
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        zi = scipy.interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

        fig = go.Figure()
        fig.add_trace(go.Contour(
            z=zi,
            x=xi,
            y=yi,
            colorscale='Viridis',
            colorbar=dict(title='Pressure'),
            name=f'3D Domain Section at Z = {z_level}'
        ))

        fig.update_layout(
            xaxis = dict(title='X'),
            yaxis = dict(title='Y', scaleanchor="x", scaleratio=1),
            width=800,
            height=800
        )

        fig.show()