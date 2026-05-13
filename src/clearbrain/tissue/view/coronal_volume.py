# ================================================================
# 0. Section: IMPORTS
# ================================================================
import math
from pathlib import Path
import numpy as np
import pandas as pd

from matplotlib.figure import Figure

from ..ClearVolume import ClearVolume
from ...plots import PlotSettings, plot_imshow_grid
from ...save import SaveSettings


# ================================================================
# 1. Section: Functions
# ================================================================
def plot_volume_coronal(
    volume_tissue: ClearVolume,
    nr_cuts: int,
    show_centers: bool = False,
    is_save: bool = False,
    save_settings: SaveSettings | None = None,
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
    plt_config = PlotSettings(
        nr_cols=5, nr_rows=nr_rows, show_imshow_data_center=show_centers
    )

    # 5. Generates the plot
    fig, axes = plot_imshow_grid(np.asarray(imgs), np.asarray(titles), plt_config)

    # 6. Deals with the save config
    if save_settings is None:
        save_settings = SaveSettings(
            name=f"coronal_slices_{nr_cuts-1}_cuts",
            out_path=Path(f"out/{volume_tissue.metadata.mouse}"),
        )

    # 7. Save if needed
    if is_save:
        save_settings.save_plot(fig)

    return fig, axes
