# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.axes import Axes



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_centerline(
    centerline: np.ndarray,
    highlight_centerline: bool = False,
) -> tuple:

    # 1. Instantiates the Plot
    fig_3d = plt.figure(figsize=(10, 6))
    ax = fig_3d.add_subplot(projection='3d', computed_zorder= not highlight_centerline)

    # 3. Loads Plots the centerline
    if highlight_centerline:
        centerline_z_order = 10
    else:
        centerline_z_order = None

    x = centerline[:, 0]
    y = centerline[:, 1]
    z = centerline[:, 2]
    ax.scatter(x, y, z, s=1, color='red', label="Centerline", zorder=centerline_z_order) # type: ignore
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    # 4. Remove extra visual clutter
    ax.grid(False)
    ax.set_axis_off()

    return fig_3d, ax

def add_centerline(
    ax: Axes,
    centerline: np.ndarray,
    highlight_centerline: bool = False,
) -> Axes:
    # 1. defines highlight params
    if highlight_centerline:
        zorder = 10
        size = 4
    else:
        zorder = None
        size = 1

    # 2. Loads and plots the cente3rline
    x = centerline[:, 0]
    y = centerline[:, 1]
    z = centerline[:, 2]
    ax.scatter(x, y, z, s=size, color="red", zorder=zorder)  # type: ignore

    return ax
