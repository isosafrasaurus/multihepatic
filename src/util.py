import numpy as np
from graphnics import *

BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

def get_box_edges(corners):
    x_line, y_line, z_line = [], [], []
    for i, j in BOX_EDGES:
        x_line += [corners[i, 0], corners[j, 0], None]
        y_line += [corners[i, 1], corners[j, 1], None]
        z_line += [corners[i, 2], corners[j, 2], None]
    return np.array(x_line), np.array(y_line), np.array(z_line)

def compute_boundaries(coords):
    return (coords[:, 0].min(), coords[:, 0].max(),
            coords[:, 1].min(), coords[:, 1].max(),
            coords[:, 2].min(), coords[:, 2].max())