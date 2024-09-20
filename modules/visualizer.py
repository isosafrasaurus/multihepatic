import matplotlib
import matplotlib.pyplot as plt
import numpy
import plotly
import scipy
import plotly.graph_objects as go
import numpy as np
import scipy.interpolate

def visualize(mesh1d, sol1d, mesh3d=None, sol3d=None, z_level=50, elev=90, azim=-90):
    # Two columns if mesh3d included, otherwise only one column for scatter
    num_columns = 2 if (sol3d is not None and mesh3d is not None) else 1
    if num_columns == 1:
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(111, projection='3d')  # 111 means 1 row, 1 column, 1st subplot 
    else:
        fig, ax1 = plt.subplots(1, num_columns, figsize=(20, 10))
        ax1 = fig.add_subplot(121, projection='3d')

    # Plot mesh1d via scatter
    scatter_coords = mesh1d.coordinates()
    scatter_values = sol1d.compute_vertex_values(mesh1d)     
    x_min, x_max = scatter_coords[:, 0].min(), scatter_coords[:, 0].max()
    y_min, y_max = scatter_coords[:, 1].min(), scatter_coords[:, 1].max()
    x_width, y_width = x_max - x_min, y_max - y_min
        
    x_plane, y_plane = np.meshgrid( #2D plane at z_level
        np.linspace(x_min, x_max, num=10),
        np.linspace(y_min, y_max, num=10)
    )
    z_plane = np.full(x_plane.shape, z_level)

    # Ensure 1:1:1 scaling for the scatter plot axes
    ax1.set_box_aspect((np.ptp(scatter_coords[:,0]), np.ptp(scatter_coords[:,1]), np.ptp(scatter_coords[:,2])))

    ax1.plot_surface(x_plane, y_plane, z_plane, color='m', alpha=0.3, zorder=10) # ensure plane is in front
    ax1.view_init(elev=elev, azim=azim)
    sc = ax1.scatter(scatter_coords[:, 0], scatter_coords[:, 1], scatter_coords[:, 2], c=scatter_values, cmap='viridis', s=0.5)
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label('1D Pressure (Pa)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('1D Domain')
    
    if num_columns == 2:
        heat_values = sol3d.compute_vertex_values(mesh3d)
        heat_coords = mesh3d.coordinates()
        z_coords = heat_coords[:, 2]
        mask = np.isclose(z_coords, z_level, atol=mesh3d.hmin())
        filtered_coords = heat_coords[mask]
        filtered_pressure = heat_values[mask]

        # grid for the heatmap
        x = filtered_coords[:, 0]
        y = filtered_coords[:, 1]
        z = filtered_pressure
        grid_size = 500
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        zi = scipy.interpolate.griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear')

        # Apply Gaussian filtering
        sigma = 5
        zi_smoothed = scipy.ndimage.gaussian_filter(zi, sigma=sigma)
        
        ax2 = fig.add_subplot(122)
        cf = ax2.contourf(xi, yi, zi_smoothed, levels=100, cmap='viridis')
        cbar = plt.colorbar(cf, ax=ax2)
        cbar.set_label('3D Pressure (Pa)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title(f'3D Domain Section at Z = {z_level}')
        fig.get_axes()[1].axis('off')
    
    fig.get_axes()[0].axis('off')
    plt.gca().set_aspect('equal')
    plt.show()

def visualize_scatter(mesh1d, sol1d, z_level=50):
    scatter_coords = mesh1d.coordinates()
    scatter_values = sol1d.compute_vertex_values(mesh1d)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=scatter_coords[:, 0], y=scatter_coords[:, 1], z=scatter_coords[:, 2],
                              mode='markers',
                              marker=dict(size=2, color=scatter_values, colorscale='Viridis', colorbar=dict(title='1D Pressure')), hovertext=[f"Pressure: {val:.4f}" for val in scatter_values],
                              name=''))

    # Plane at Z = z_level
    x_min, x_max, y_min, y_max = scatter_coords[:, 0].min(), scatter_coords[:, 0].max(), scatter_coords[:, 1].min(), scatter_coords[:, 1].max()
    x_plane, y_plane = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
    z_plane = z_level * np.ones_like(x_plane)

    fig.add_trace(go.Surface(x=x_plane, y=y_plane, z=z_plane, opacity=0.3, name=f'Plane at Z={z_level}'))

    # Set camera view
    fig.update_layout(
        scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=-1.25, y=-1.25, z=1.25)),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        width=1000, height=800
    )

    fig.show()

def visualize_contour(mesh3d, sol3d, z_level=50):
    if sol3d is not None and mesh3d is not None:
        heat_values = sol3d.compute_vertex_values(mesh3d)
        heat_coords = mesh3d.coordinates()

        z_coords = heat_coords[:, 2]
        mask = np.isclose(z_coords, z_level, atol=mesh3d.hmin())
        filtered_coords = heat_coords[mask]
        filtered_pressure = heat_values[mask]

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