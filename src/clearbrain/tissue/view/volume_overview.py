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
    save_settings: SaveSettings | None = None,
) -> tuple[Figure, np.ndarray]:
    volume = volume_tissue.volume
    imgs, titles = build_volume_overview_slices(volume, nr_cuts)

    plt_cfg = PlotSettings(nr_cols=3, nr_rows=nr_cuts)
    fig, axes = plot_imshow_grid(imgs, np.asarray(titles), plt_cfg)

    if save_settings is None:
        save_settings = SaveSettings(
            name=f"overview_slices_{nr_cuts}_cuts",
            out_path=Path(f"out/{volume_tissue.metadata.mouse}"),
        )

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


def build_volume_overview_slices(
    volume: np.ndarray,
    nr_cuts: int,
) -> tuple[list[np.ndarray], list[str]]:
    shape = np.asarray(volume.shape)
    cut_denominator = nr_cuts + 1

    cut_fractions = np.asarray(
        [cut / cut_denominator for cut in range(1, cut_denominator)]
    )
    cut_names = [f"{c}%" for c in np.round(cut_fractions * 100)]
    axis_names = ["axial", "coronal", "sagittal"]

    imgs = []
    titles = []

    for i, frac in enumerate(cut_fractions):
        cuts = (shape * frac).astype(int)

        for axis, axis_name in enumerate(axis_names):
            imgs.append(get_axis_based_cut(volume, axis, cuts))
            titles.append(f"{cut_names[i]} | fixed {axis_name}")

    return imgs, titles
