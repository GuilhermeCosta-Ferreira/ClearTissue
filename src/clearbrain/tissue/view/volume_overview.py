# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from matplotlib.figure import Figure
from pathlib import Path

from ..ClearVolume import ClearVolume
from ...plots import PlotSettings, plot_imshow_grid
from ...save import SaveSettings



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_volume_overview(
    volume_tissue: ClearVolume,
    nr_cuts: int,
    is_save: bool = False,
    save_settings: SaveSettings | None = None
) -> tuple[Figure, np.ndarray]:
    # 1. Extract the data
    volume = volume_tissue.volume
    shape = np.asarray(volume.shape)
    nr_cuts += 1

    # 2. Makes the slices for each cut
    cut_fractions = np.asarray([cut / nr_cuts for cut in range(1, nr_cuts)])
    cut_names = [f"{c}%" for c in np.round(cut_fractions * 100)]
    axis_names = ["axial", "coronal", "sagittal"]

    # 3. Iterates to build each img
    imgs, titles = [], []
    for i, frac in enumerate(cut_fractions):
        for col in range(3):
            cuts = (shape * frac).astype(int)
            img = get_axis_based_cut(volume, col, cuts)
            imgs.append(img)
            titles.append(f"{cut_names[i]} | fixed {axis_names[col]}")

     # 4. Makes the config
    titles = np.asarray(titles)
    nr_rows = nr_cuts
    plt_cfg = PlotSettings(nr_cols=3, nr_rows=nr_rows)

    fig, axes = plot_imshow_grid(imgs, titles, plt_cfg)

    # 6. Deals with the save config
    if save_settings is None:
        save_settings = SaveSettings(
            name=f"overview_slices_{nr_cuts-1}_cuts",
            out_path=Path(f"out/{volume_tissue.metadata.mouse}")
        )

    # 7. Save if needed
    if is_save:
        save_settings.save_plot(fig)

    return fig, axes


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_axis_based_cut(volume: np.ndarray, col: int, cuts: tuple[int, int, int]):
    if col == 0:
        img = volume[cuts[0], :, :]
    elif col == 1:
        img = volume[:, cuts[1], :]
    else:
        img = volume[:, :, cuts[2]]

    return img
