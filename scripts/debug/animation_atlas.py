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

def rolling_label_sum(label_sums_by_z: np.ndarray, depth: int) -> np.ndarray:
    """
    For each z index i, computes:

        label_sums_by_z[i : i + depth].sum(axis=0)

    Returns an array shaped:

        (z_size, n_labels)
    """
    depth = max(1, int(depth))
    z_size = label_sums_by_z.shape[0]

    cumsum = np.empty((z_size + 1, label_sums_by_z.shape[1]), dtype=np.float32)
    cumsum[0] = 0
    np.cumsum(label_sums_by_z, axis=0, dtype=np.float32, out=cumsum[1:])

    starts = np.arange(z_size)
    ends = np.minimum(starts + depth, z_size)

    return cumsum[ends] - cumsum[starts]


def make_region_cell_overlay(
    atlas_slice: np.ndarray,
    region_sums: np.ndarray,
    vmax: float,
    red_rgb: tuple[float, float, float] = (0.976, 0.322, 0.357),
    max_alpha: float = 0.85,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Creates an RGBA overlay where each atlas region is colored red
    proportional to the summed cell value in that region.

    exclude_mask can be used to prevent overlay on white matter.
    """
    overlay = np.zeros((*atlas_slice.shape, 4), dtype=np.float32)

    region_sum_image = region_sums[atlas_slice]

    valid = (atlas_slice > 0) & (region_sum_image > 0)

    if exclude_mask is not None:
        valid &= ~exclude_mask

    if vmax <= 0:
        return overlay

    intensity = np.clip(region_sum_image / vmax, 0, 1)

    overlay[..., 0] = red_rgb[0]
    overlay[..., 1] = red_rgb[1]
    overlay[..., 2] = red_rgb[2]
    overlay[..., 3] = intensity * max_alpha

    overlay[~valid, 3] = 0

    return overlay

def map_display_z_to_count_z(
    display_idx: int,
    display_z: int,
    count_z: int,
) -> int:
    if display_z == count_z:
        return display_idx

    return int(round(display_idx * (count_z - 1) / max(display_z - 1, 1)))


# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    output_dir = Path("frames_atlas_2")
    output_dir.mkdir(exist_ok=True)

    project = ClearTissueProject.load(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )

    atlas_name = project.source.atlas_name
    atlas_original = BrainGlobeAtlas(atlas_name)

    final_batch = project.io.load_batch(pipeline_id=11, step=8)

    # Atlas used for counting cells.
    # This is the atlas aligned to your sample/cell volume.
    count_atlas = np.asarray(final_batch.atlas.data).astype(np.int32, copy=False)

    # Original BrainGlobe atlas used only for visualization.
    display_atlas = np.asarray(atlas_original.annotation).astype(np.int32, copy=False)

    cells = np.asarray(final_batch.cells.data).astype(np.float32, copy=False)

    count_z, count_height, count_width = count_atlas.shape
    display_z, height, width = display_atlas.shape

    count_atlas_mask = count_atlas > 0
    display_atlas_mask = display_atlas > 0

    nonempty_slices = np.flatnonzero(count_atlas_mask.any(axis=(1, 2)))

    if len(nonempty_slices) == 0:
        raise ValueError("Atlas contains no non-zero slices.")

    start = int(nonempty_slices[0])
    missing_frames = display_z - start

    # ------------------------------------------------------------
    # Tissue masks
    # ------------------------------------------------------------
    gm_root_id = 71
    wm_root_id = 130

    gm_ids = get_descendant_ids_from_root(atlas_original, gm_root_id)
    wm_ids = get_descendant_ids_from_root(atlas_original, wm_root_id)

    gm_lut = make_membership_lut(display_atlas, gm_ids)
    wm_lut = make_membership_lut(display_atlas, wm_ids)

    gm_colour = np.array((0.5, 0.5, 0.5, 1.0), dtype=np.float32)
    wm_colour = np.array((0.8, 0.8, 0.8, 1.0), dtype=np.float32)

    tissue_rgba = np.zeros((height, width, 4), dtype=np.float32)

    # ------------------------------------------------------------
    # Precompute cell sums per anatomical region, per slice
    # using the registered/project atlas
    # ------------------------------------------------------------

    # If cells is binary/intensity/count volume:
    cell_values = np.where(count_atlas_mask, cells, 0).astype(np.float32, copy=False)

    # If cells contains labelled cell IDs, use this instead:
    # cell_values = ((cells > 0) & count_atlas_mask).astype(np.float32)

    max_label = max(
        int(count_atlas.max()),
        int(display_atlas.max()),
    )

    region_sums_by_count_slice = np.zeros((count_z, max_label + 1), dtype=np.float32)

    for z in tqdm(range(count_z), desc="Counting cells by region per slice"):
        region_sums_by_count_slice[z] = np.bincount(
            count_atlas[z].ravel(),
            weights=cell_values[z].ravel(),
            minlength=max_label + 1,
        )
    test = np.sum(region_sums_by_count_slice, axis = 1)
    plt.figure()
    plt.plot(test)
    plt.savefig("test.png")

    # Global color scaling, but values are still per slice.
    nonzero_region_sums = region_sums_by_count_slice[:, 1:]
    nonzero_region_sums = nonzero_region_sums[nonzero_region_sums > 0]

    if len(nonzero_region_sums) == 0:
        region_vmax = 1.0
    else:
        region_vmax = float(np.percentile(nonzero_region_sums, 99))

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

    previous_slice = np.zeros_like(display_atlas[0])
    # ------------------------------------------------------------
    # Render frames
    # ------------------------------------------------------------
    for added in tqdm(range(missing_frames)):
        display_idx = start + added

        # Slice shown in the output image: original BrainGlobe atlas.
        atlas_slice = display_atlas[added]

        if (atlas_slice == previous_slice).all():
            continue
        else:
            previous_slice = atlas_slice.copy()

        region_sums = region_sums_by_count_slice[display_idx]

        ax.clear()
        ax.set_axis_off()

        tissue_rgba.fill(0)

        gm_mask = gm_lut[atlas_slice]
        wm_mask = wm_lut[atlas_slice]

        tissue_rgba[gm_mask] = gm_colour
        tissue_rgba[wm_mask] = wm_colour

        # Show original atlas base.
        ax.imshow(tissue_rgba, interpolation="nearest")

        # Red overlay from cell counts computed using count_atlas.
        region_cell_overlay = make_region_cell_overlay(
            atlas_slice=atlas_slice,
            region_sums=region_sums,
            vmax=region_vmax,
            max_alpha=0.85,
            exclude_mask=wm_mask,
        )

        ax.imshow(
            region_cell_overlay,
            interpolation="nearest",
        )

        # Contours from original atlas.
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

        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)

        fig.savefig(
            output_dir / f"frame_{display_idx:04d}.png",
            transparent=True,
            pad_inches=0,
        )
