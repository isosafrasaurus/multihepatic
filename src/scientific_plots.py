import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_path_pressure_csv(csv_path: str, out_png: str = None):
    """
    Load 'culum_dist' vs 'path_pressure' CSV and plot a quick line chart.
    """
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["culum_dist"], df["path_pressure"], linewidth=2)
    ax.set_xlabel("Cumulative distance (m)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Path Pressure")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=200)
    return fig, ax

