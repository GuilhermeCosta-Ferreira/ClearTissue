"""Render projection views and per-slice PCA of one spinal cord along each axis.

One file per volume axis (0, 1, 2), each a row of equal-size panels:
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
MOUSE: str = "EESpaddle_2mm_1"
PIPELINE_ID: int = 2
STEP_ID: int = 7
TISSUE_TYPE: TissueType = TissueType.SPINAL_CORD

FMT: Literal["png", "svg"] = "png"
VIEW_THRESHOLD: float = 10  # voxels at or below this are background
OUTPUT_ROOT: Path = Path("data/cord_views")
# Any matplotlib colormap name per view mode, e.g. "magma", "viridis", "gray_r".
CMAPS: dict[str, str] = {
    "mean": "gray",
    "view": "hot",
    "cumsum": "gray",
}


# ================================================================
# 3. Section: Functions
# ================================================================
def main() -> None:
    project = ClearTissueProject.load(mouse=MOUSE, tissue_type=TISSUE_TYPE)
    batch = project.io.load_batch(pipeline_id=PIPELINE_ID, step=STEP_ID)
    volume = np.asarray(batch.tissue.data)
    resolution = tuple(float(r) for r in batch.tissue.resolution)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for axis in AXES:
        fig = _render(volume, resolution, axis)
        name = f"cord_views_{MOUSE}_p{PIPELINE_ID}_s{STEP_ID}_axis{axis}.{FMT}"
        out = OUTPUT_ROOT / name
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"saved {out}")


# ──────────────────────────────────────────────────────
# 3.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _render(volume: NDArray, resolution: tuple[float, ...], axis: int) -> Figure:
    projections: dict[str, Callable[[], NDArray]] = {
        "mean": lambda: mean_projection(volume, axis, VIEW_THRESHOLD),
        "view": lambda: view_projection(volume, axis, VIEW_THRESHOLD),
        "cumsum": lambda: cumsum_projection(volume, axis, VIEW_THRESHOLD),
    }
    n_panels = len(VIEWS) + 1
    # Same figsize for every axis so panels match in size across the files.
    fig, row = plt.subplots(
        1,
        n_panels,
        figsize=(4.5 * n_panels, 4.6),
        squeeze=False,
        gridspec_kw={"wspace": 0.45},
    )
    panels = row[0]
    for panel, (key, label) in zip(panels, VIEWS.items()):
        _plot_projection(panel, projections[key](), axis, resolution, CMAPS[key])
        panel.set_title(f"axis {axis}: {label}", fontsize=9)
    _plot_pca(panels[-1], volume, axis)

    fig.suptitle(
        f"{MOUSE}  pipeline {PIPELINE_ID}  step {STEP_ID}  axis {axis}", fontsize=12
    )
    fig.subplots_adjust(top=0.85, left=0.04, right=0.96, bottom=0.12)
    return fig


def _plot_projection(
    ax: Axes, image: NDArray, axis: int, resolution: tuple[float, ...], cmap: str
) -> None:
    rows_axis, cols_axis = (a for a in AXES if a != axis)
    h, w = image.shape
    ax.imshow(
        image,
        cmap=cmap,
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
