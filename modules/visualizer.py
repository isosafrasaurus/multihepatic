import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

class Visualizer:
  @staticmethod
  def plot(uh1d, uh3d, z_level=None):
      mesh1d = uh1d.function_space().mesh()
      coords1d = mesh1d.coordinates()
      values1d = uh1d.compute_vertex_values(mesh1d)
      mesh3d = uh3d.function_space().mesh()
      coords3d = mesh3d.coordinates()
      values3d = uh3d.compute_vertex_values(mesh3d)

      if z_level is None:
          z_level = np.median(coords3d[:, 2])

      fig = plt.figure(figsize=(14, 6))

      ax1 = fig.add_subplot(1, 2, 1, projection='3d')
      sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2],
                      c=values1d, cmap='viridis', marker='o')
      fig.colorbar(sc, ax=ax1, label='1D Pressure')
      ax1.set_title('1D Pressure Scatter')
      ax1.set_xlabel('X')
      ax1.set_ylabel('Y')
      ax1.set_zlabel('Z')

      tol = 1e-3  
      mask = np.abs(coords3d[:, 2] - z_level) < tol
      if not np.any(mask):
          print(f"No data found at Z={z_level}")
          return
      x = coords3d[mask, 0]
      y = coords3d[mask, 1]
      z = values3d[mask]

      
      xi = np.linspace(x.min(), x.max(), 100)
      yi = np.linspace(y.min(), y.max(), 100)
      xi, yi = np.meshgrid(xi, yi)

      
      zi = griddata((x, y), z, (xi, yi), method='cubic')

      
      ax2 = fig.add_subplot(1, 2, 2)
      heatmap = ax2.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()),
                            origin='lower', cmap='viridis', aspect='auto')
      fig.colorbar(heatmap, ax=ax2, label='3D Pressure')
      ax2.set_title(f'3D Pressure Heatmap at Z={z_level:.3f}')
      ax2.set_xlabel('X')
      ax2.set_ylabel('Y')

      plt.tight_layout()
      plt.show()

  @staticmethod
  def plot_with_streamlines(uh1d, uh3d, velocity, z_level=None):
      
      
      mesh1d = uh1d.function_space().mesh()
      coords1d = mesh1d.coordinates()
      values1d = uh1d.compute_vertex_values(mesh1d)
      
      
      mesh3d = uh3d.function_space().mesh()
      coords3d = mesh3d.coordinates()
      values3d = uh3d.compute_vertex_values(mesh3d)
      
      
      if z_level is None:
          z_level = np.median(coords3d[:, 2])
      
      fig = plt.figure(figsize=(16, 6))
      
      
      ax1 = fig.add_subplot(1, 3, 1, projection='3d')
      sc = ax1.scatter(coords1d[:, 0], coords1d[:, 1], coords1d[:, 2],
                      c=values1d, cmap='viridis', marker='o')
      fig.colorbar(sc, ax=ax1, label='1D Pressure')
      ax1.set_title('1D Pressure Scatter')
      ax1.set_xlabel('X')
      ax1.set_ylabel('Y')
      ax1.set_zlabel('Z')
      
      
      tol = 1e-3  
      mask = np.abs(coords3d[:, 2] - z_level) < tol
      if not np.any(mask):
          print(f"No data found at Z={z_level}")
          return
      
      x = coords3d[mask, 0]
      y = coords3d[mask, 1]
      z = values3d[mask]
      
      
      velocity_values = velocity.compute_vertex_values(mesh3d)
      
      
      u = velocity_values[0::velocity.function_space().mesh().geometry().dim()]
      v = velocity_values[1::velocity.function_space().mesh().geometry().dim()]
      
      u_slice = u[mask]
      v_slice = v[mask]
      
      
      xi = np.linspace(x.min(), x.max(), 100)
      yi = np.linspace(y.min(), y.max(), 100)
      xi, yi = np.meshgrid(xi, yi)
      
      
      zi = griddata((x, y), z, (xi, yi), method='cubic')
      
      
      ui = griddata((x, y), u_slice, (xi, yi), method='cubic')
      vi = griddata((x, y), v_slice, (xi, yi), method='cubic')
      
      
      ax2 = fig.add_subplot(1, 3, 2)
      heatmap = ax2.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()),
                          origin='lower', cmap='viridis', aspect='auto')
      fig.colorbar(heatmap, ax=ax2, label='3D Pressure')
      ax2.set_title(f'3D Pressure Heatmap at Z={z_level:.3f}')
      ax2.set_xlabel('X')
      ax2.set_ylabel('Y')
      
      
      
      mask_stream = ~np.isnan(ui) & ~np.isnan(vi)
      if np.any(mask_stream):
          ax2.streamplot(xi, yi, ui, vi, color='white', density=1.5, linewidth=1, arrowsize=1)
      else:
          print("Insufficient velocity data for streamlines.")
      
      
      
      """
      ax3 = fig.add_subplot(1, 3, 3)
      ax3.imshow(zi, extent=(x.min(), x.max(), y.min(), y.max()),
                origin='lower', cmap='viridis', aspect='auto')
      ax3.streamplot(xi, yi, ui, vi, color='white', density=1.5, linewidth=1, arrowsize=1)
      ax3.set_title(f'3D Pressure Heatmap with Streamlines at Z={z_level:.3f}')
      ax3.set_xlabel('X')
      ax3.set_ylabel('Y')
      fig.colorbar(heatmap, ax=ax3, label='3D Pressure')
      """
      
      plt.tight_layout()
      plt.show()