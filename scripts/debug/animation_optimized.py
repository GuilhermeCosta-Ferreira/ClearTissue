# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # faster/non-interactive backend for batch rendering

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import (
    binary_dilation,
    binary_opening,
    gaussian_filter,
    generate_binary_structure,
)
from scipy.spatial import cKDTree
from tqdm import tqdm

from brainglobe_atlasapi import BrainGlobeAtlas
from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject


# ================================================================
# 1. Section: FUNCTIONS
# ================================================================
def get_density(points: np.ndarray, density_radius: float) -> np.ndarray:
    """
    Faster version of the original KDTree loop.
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)

    tree = cKDTree(points)
    return tree.query_ball_point(
        points,
        r=density_radius,
        return_length=True,
    ).astype(np.int32) - 1


def get_descendant_ids_from_root(
    atlas_original: BrainGlobeAtlas,
    root_id: int,
) -> set[int]:
    descendant_ids = {root_id}

    for structure_id, structure_info in atlas_original.structures.items():
        structure_path = structure_info.get("structure_id_path", [])

        if root_id in structure_path:
            descendant_ids.add(int(structure_id))

    return descendant_ids


def make_membership_lut(atlas: np.ndarray, ids: set[int]) -> np.ndarray:
    """
    Fast replacement for repeated np.isin(slice, ids).
    Assumes atlas IDs are non-negative integers.
    """
    max_id = int(atlas.max())
    lut = np.zeros(max_id + 1, dtype=bool)

    valid_ids = [int(i) for i in ids if 0 <= int(i) <= max_id]
    lut[valid_ids] = True

    return lut


def rolling_z_sum(volume: np.ndarray, depth: int) -> np.ndarray:
    """
    For each z index i, computes:

        volume[i : i + depth].sum(axis=0)

    for all i at once using cumulative sums.
    """
    depth = max(1, int(depth))
    z_size = volume.shape[0]

    cumsum = np.empty((z_size + 1, *volume.shape[1:]), dtype=np.float32)
    cumsum[0] = 0
    np.cumsum(volume, axis=0, dtype=np.float32, out=cumsum[1:])

    starts = np.arange(z_size)
    ends = np.minimum(starts + depth, z_size)

    return cumsum[ends] - cumsum[starts]


def make_cell_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "cell_red_alpha",
        [
            (1.0, 1.0, 1.0, 0.1),
            (0.976, 0.322, 0.357, 1.0),
        ],
    )

    # Make NaN transparent.
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    return cmap


# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    output_dir = Path("frames_opt")
    output_dir.mkdir(exist_ok=True)

    project = ClearTissueProject.load(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )

    atlas_name = project.source.atlas_name
    atlas_original = BrainGlobeAtlas(atlas_name)

    final_batch = project.io.load_batch(pipeline_id=12, step=7)
    final_batch_2 = project.io.load_batch(pipeline_id=3, step=2)

    atlas = np.asarray(final_batch_2.atlas.data)
    cells = np.asarray(final_batch.atlas.hemisphere)

    atlas = atlas.astype(np.int32, copy=False)
    cells = cells.astype(np.float32, copy=False)

    z_size, height, width = atlas.shape

    atlas_mask = atlas > 0

    nonempty_slices = np.flatnonzero(atlas_mask.any(axis=(1, 2)))
    if len(nonempty_slices) == 0:
        raise ValueError("Atlas contains no non-zero slices.")

    start = int(nonempty_slices[0])
    missing_frames = z_size - start

    # ------------------------------------------------------------
    # Tissue masks
    # ------------------------------------------------------------
    gm_root_id = 71
    wm_root_id = 130

    gm_ids = get_descendant_ids_from_root(atlas_original, gm_root_id)
    wm_ids = get_descendant_ids_from_root(atlas_original, wm_root_id)

    gm_lut = make_membership_lut(atlas, gm_ids)
    wm_lut = make_membership_lut(atlas, wm_ids)

    gm_colour = np.array((0.5, 0.5, 0.5, 1.0), dtype=np.float32)
    wm_colour = np.array((0.8, 0.8, 0.8, 1.0), dtype=np.float32)

    tissue_rgba = np.zeros((height, width, 4), dtype=np.float32)

    # ------------------------------------------------------------
    # Clean cells
    # ------------------------------------------------------------
    cell_mask = (cells > 0) & atlas_mask

    structure = generate_binary_structure(rank=3, connectivity=1)

    cell_mask_opened = binary_opening(
        cell_mask,
        structure=structure,
        iterations=1,
    )

    cell_mask_opened = binary_dilation(
        cell_mask_opened,
        structure=structure,
        iterations=1,
    )

    # Important: your original code computed cell_mask_opened but did not use it.
    cells_clean = np.where(atlas_mask, cells, 0).astype(np.float32, copy=False)

    # If you want the previous behavior exactly, use this instead:
    # cells_clean = np.where(atlas_mask, cells, 0).astype(np.float32, copy=False)

    # ------------------------------------------------------------
    # Precompute projected cell images
    # ------------------------------------------------------------
    # Your original code used final_batch.cells.shape[1] // 4.
    # Since projection happens along axis 0, using cells.shape[0] is usually more correct.
    projection_depth = max(1, final_batch.cells.shape[1] // 4)

    projected_cells = rolling_z_sum(cells_clean, projection_depth)

    cell_vmin = np.nanpercentile(projected_cells, 1)
    cell_vmax = np.nanpercentile(projected_cells, 99.5)

    # Equivalent to applying gaussian_filter(cell_slice, 1.2) per frame,
    # but vectorized over all z slices.
    projected_cells = gaussian_filter(
        projected_cells,
        sigma=(0, 1.2, 1.2),
    )

    projected_cells = np.where(
        (projected_cells > 0) & atlas_mask,
        projected_cells,
        np.nan,
    )

    # ------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------
    cell_cmap = make_cell_cmap()

    colour_outline = "#000000"

    # This controls PNG pixel size.
    # OUTPUT_SCALE=1 gives one output pixel per atlas pixel.
    # Increase to 2, 4, etc. if you need larger PNGs.
    output_scale = 4
    dpi = 300

    fig = plt.figure(
        figsize=(width * output_scale / dpi, height * output_scale / dpi),
        dpi=dpi,
        frameon=False,
    )

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # ------------------------------------------------------------
    # Render frames
    # ------------------------------------------------------------
    for added in tqdm(range(missing_frames)):
        idx = start + added

        ax.clear()
        ax.set_axis_off()

        atlas_slice = atlas[idx]
        cell_slice = projected_cells[idx]

        tissue_rgba.fill(0)

        gm_mask = gm_lut[atlas_slice]
        wm_mask = wm_lut[atlas_slice]

        tissue_rgba[gm_mask] = gm_colour
        tissue_rgba[wm_mask] = wm_colour

        ax.imshow(tissue_rgba, interpolation="nearest")

        # Only contour labels that are actually present in this slice.
        labels_present = np.unique(atlas_slice)
        labels_present = labels_present[labels_present > 0]

        for label in labels_present:
            mask = atlas_slice == label

            ax.contour(
                mask.astype(np.float32),
                levels=[0.5],
                colors=[colour_outline],
                linewidths=0.5,
                antialiased=True,
            )

        ax.imshow(
            cell_slice,
            cmap=cell_cmap,
            interpolation="nearest",
            vmin=cell_vmin,
            vmax=cell_vmax,
        )

        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)

        fig.savefig(
            output_dir / f"frame_{idx:04d}.png",
            transparent=True,
            pad_inches=0,
        )

    plt.close(fig)
