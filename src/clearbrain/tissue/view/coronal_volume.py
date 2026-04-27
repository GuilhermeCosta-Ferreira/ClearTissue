# ================================================================
# 0. Section: IMPORTS
# ================================================================
import math
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.figure import Figure

from ..ClearVolume import ClearVolume
from ...plots import PlotSettings, plot_imshow_grid



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_volume_coronal(
    volume_tissue: ClearVolume,
    nr_cuts: int,
    show_centers: bool = False
) -> tuple[Figure, np.ndarray]:
    # 1. Extract the data
    volume = volume_tissue.volume
    shape = np.asarray(volume.shape)
    nr_cuts += 1

    # 2. Makes the slices for each cut
    cut_fractions = np.asarray([cut / nr_cuts for cut in range(1, nr_cuts)])
    imgs, titles = [], []
    for frac in cut_fractions:
        cut = int(shape[1] * frac)
        img = volume[:, cut, :]
        imgs.append(img)
        titles.append(f"Coronal {frac * 100:.0f}%")

    # 3. Makes the config
    nr_cols = 5
    nr_rows = math.ceil((nr_cuts) / nr_cols)
    plt_config = PlotSettings(nr_cols=5, nr_rows=nr_rows, show_imshow_data_center=show_centers)

    return plot_imshow_grid(
        imgs=np.asarray(imgs),
        titles=np.asarray(titles),
        plt_cfg=plt_config
    )
