# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..SpinalCenterline import SpinalCenterline
from ..ClearVolume import ClearVolume


# ================================================================
# 1. Section: Functions
# ================================================================
def plot_spinal_direction(
    vol_tissue: ClearVolume, centerline: SpinalCenterline, smooth_window: int = 9
) -> tuple[Figure, Axes]:
    derivatives = centerline.smooth_derivative(smooth_window)
    centers = centerline.points

    Y = centers[:, 1]  # vertical image coordinate
    X = centers[:, 2]  # horizontal image coordinate

    V = derivatives[:, 1]  # vertical vector component
    U = derivatives[:, 2]  # horizontal vector component

    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(U) & np.isfinite(V)

    fig, ax = plt.subplots()
    ax.imshow(vol_tissue.volume[50, :, :], cmap="hot", origin="upper")

    ax.quiver(
        X[valid],
        Y[valid],
        U[valid],
        V[valid],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="white",
    )

    return fig, ax


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
