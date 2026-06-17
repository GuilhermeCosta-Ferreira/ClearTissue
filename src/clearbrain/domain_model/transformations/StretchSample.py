# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt
import numpy as np



from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume, SpinalCenterline
from .utils import get_centerline, stretch_tissue
from .AbstractTransformations import AbstractTransformations



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class StretchSample(AbstractTransformations):
    smooth_window_size: int

    def apply(self, batch: SampleBatch) -> SampleBatch:
        centerline = get_centerline(batch.tissue)
        plot_spinal_direction(batch.tissue, centerline, self.smooth_window_size)
        plt.show()

        stretched_tissue = stretch_tissue(batch.tissue, centerline, self.smooth_window_size)
        if isinstance(batch.cells, ClearVolume):
            stretched_cell_volume = stretch_tissue(batch.cells, centerline, self.smooth_window_size)
        else:
            raise TypeError(
                f"Expected ClearVolume, got {type(batch.cells)} for the cells,"
                " make sure you run RegularizeSample first"
            )

        return batch.copy_with(tissue=stretched_tissue, cells=stretched_cell_volume)

def plot_spinal_direction(
    vol_tissue: ClearVolume, centerline: SpinalCenterline, smooth_window: int
):
    derivatives = centerline.smooth_derivative(smooth_window)
    centers = centerline.data

    Y = centers[:, 0]  # vertical image coordinate
    X = centers[:, 2]  # horizontal image coordinate

    V = derivatives[:, 0]  # vertical vector component
    U = derivatives[:, 2]  # horizontal vector component

    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(U) & np.isfinite(V)

    fig, ax = plt.subplots()
    ax.imshow(vol_tissue.data[:, 50, :], cmap="hot", origin="upper")

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
