"""Render projection views and per-slice PCA of one spinal cord along each axis.

One figure, one column per volume axis (0, 1, 2). Rows:
- mean: mean intensity over the top half of the cord along the axis (from the
  centre of the non-background extent to its high-index edge).
- view: intensity of the first voxel above VIEW_THRESHOLD met looking down
  from the high-index end; rays that hit nothing are background.
- cumsum: per ray over the top half, intensity is summed from the top edge
  inward and normalised by the ray total; the pixel is the depth (0 = top
  edge, 1 = centre) where that cumsum first reaches 90 %.
- PCA: per-slice PC1/PC2 ratio vs slice index, with the signed PC1 angle to
  the last in-plane axis (degrees, twin axis, mean in the title).

Edit the INPUTS section in place, then run:
    poetry run python scripts/cli/render_cord_views.py
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Callable, Literal
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject
from cleartissue.domain_model.analysis import (
    cumsum_projection,
    mean_projection,
    pc1_angle_along_axis,
    pca_profiles_along_axis,
    view_projection,
)

# ================================================================
# 1. Section: Types
# ================================================================
AXES: tuple[int, ...] = (0, 1, 2)
VIEWS: dict[str, str] = {
    "mean": "mean (top half)",
    "view": "view (from top)",
    "cumsum": "cumsum depth (top half)",
}


# ================================================================
# 2. Section: Inputs
# ================================================================
MOUSE: str = "190R"
PIPELINE_ID: int = 2
STEP_ID: int = 7
TISSUE_TYPE: TissueType = TissueType.SPINAL_CORD

FMT: Literal["png", "svg"] = "png"
VIEW_THRESHOLD: float = 0  # voxels at or below this are background
OUTPUT_ROOT: Path = Path("data/cord_views")


# ================================================================
# 3. Section: Functions
# ================================================================
def main() -> None:
    project = ClearTissueProject.load(mouse=MOUSE, tissue_type=TISSUE_TYPE)
    batch = project.io.load_batch(pipeline_id=PIPELINE_ID, step=STEP_ID)
    volume = np.asarray(batch.tissue.data)
    resolution = tuple(float(r) for r in batch.tissue.resolution)

    fig = _render(volume, resolution)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / f"cord_views_{MOUSE}_p{PIPELINE_ID}_s{STEP_ID}.{FMT}"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _render(volume: NDArray, resolution: tuple[float, ...]) -> Figure:
    n_rows = len(VIEWS) + 1
    fig, grid = plt.subplots(
        n_rows,
        len(AXES),
        figsize=(4.5 * len(AXES), 4.2 * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 0.45, "hspace": 0.35},
    )
    projections: dict[str, Callable[[int], NDArray]] = {
        "mean": lambda ax: mean_projection(volume, ax, VIEW_THRESHOLD),
        "view": lambda ax: view_projection(volume, ax, VIEW_THRESHOLD),
        "cumsum": lambda ax: cumsum_projection(volume, ax, VIEW_THRESHOLD),
    }
    for col, axis in enumerate(AXES):
        for row, (key, label) in enumerate(VIEWS.items()):
            _plot_projection(grid[row, col], projections[key](axis), axis, resolution)
            grid[row, col].set_title(f"axis {axis}: {label}", fontsize=9)
        _plot_pca(grid[-1, col], volume, axis)

    fig.suptitle(f"{MOUSE}  pipeline {PIPELINE_ID}  step {STEP_ID}", fontsize=12)
    fig.subplots_adjust(top=0.95, left=0.06, right=0.94, bottom=0.04)
    return fig


def _plot_projection(
    ax: Axes, image: NDArray, axis: int, resolution: tuple[float, ...]
) -> None:
    rows_axis, cols_axis = (a for a in AXES if a != axis)
    h, w = image.shape
    ax.imshow(
        image,
        cmap="gray",
        extent=(0, w * resolution[cols_axis], h * resolution[rows_axis], 0),
        aspect="equal",
        interpolation="nearest",
    )
    # Square boxes keep panels identical; datalim pads limits instead of distorting.
    ax.set_adjustable("datalim")
    ax.set_box_aspect(1)
    ax.set_xlabel(f"axis {cols_axis} (µm)", fontsize=8)
    ax.set_ylabel(f"axis {rows_axis} (µm)", fontsize=8)
    ax.tick_params(labelsize=7)


def _plot_pca(ax: Axes, volume: NDArray, axis: int) -> None:
    pca = pca_profiles_along_axis(volume, axis)
    angle = pc1_angle_along_axis(volume, axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pca.pc2 > 0, pca.pc1 / pca.pc2, np.nan)
    index = np.arange(len(ratio))
    in_plane = [a for a in AXES if a != axis]

    ax.plot(index, ratio, color="tab:blue", lw=1)
    ax.set_xlabel(f"slice index (axis {axis})", fontsize=8)
    ax.set_ylabel("PC1 / PC2", color="tab:blue", fontsize=8)
    ax.tick_params(labelsize=7)

    twin = ax.twinx()
    twin.plot(index, angle, color="tab:red", lw=1)
    twin.set_ylim(-90, 90)
    twin.set_ylabel(
        f"PC1 angle to axis {in_plane[-1]} (°)", color="tab:red", fontsize=8
    )
    twin.tick_params(labelsize=7)

    finite = np.isfinite(angle)
    mean_angle = float(angle[finite].mean()) if finite.any() else np.nan
    ax.set_title(
        f"axis {axis}: per-slice PCA, mean angle {mean_angle:.1f}°", fontsize=9
    )
    ax.set_box_aspect(1)


if __name__ == "__main__":
    main()
