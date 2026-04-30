# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_3d_clear_points(points: np.ndarray, plot_subsample: int = 80) -> tuple:
    # 1. Extractes and downsamples the data
    reduced_points = points[::plot_subsample]
    x = reduced_points[:, 0]
    y = reduced_points[:, 1]
    z = reduced_points[:, 2]

    # 2. Instantiates the Plot
    fig_3d = plt.figure(figsize=(10, 6))
    ax = fig_3d.add_subplot(111, projection='3d')

    # 3. Does the ploting of the points
    ax.scatter(x, y, z, s=1, label="High-density cFos cells") # type: ignore

    # 4. Makes sure everything is proportional
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    # 5. Remove extra visual clutter
    ax.grid(False)
    ax.set_axis_off()

    return fig_3d, ax
