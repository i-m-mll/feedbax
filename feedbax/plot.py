""" """

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_2D_positions(xy, links=True, cmap_func=mpl.cm.viridis,
                      unit='m', ax=None, add_root=True):
    """Plot 2D position trajectories.
    Could also plot the controls on the joints.
    Args:
        xy ():
        links (bool):
        cmap_func ():
    """
    if ax is None:
        fig = plt.figure(figsize=(4,8))
        ax = fig.add_subplot()

    if add_root:
        xy = np.pad(xy, ((0,0), (0,0), (1,0)))

    cmap = cmap_func(np.linspace(0, 0.66, num=xy.shape[0], endpoint=True))
    cmap = mpl.colors.ListedColormap(cmap)

    ax.plot(*xy[0], c=cmap(0.), lw=2, marker="o")
    ax.plot(*xy[len(xy)//2], c=cmap(0.5), lw=2, marker='o')
    ax.plot(*xy[-1], c=cmap(1.), lw=2, marker='o')

    for j in range(xy.shape[2]):
        ax.scatter(*xy[..., j].T, marker='.', s=4, linewidth=0, c=cmap.colors)

    ax.margins(0.1, 0.2)
    ax.set_aspect('equal')
    return ax