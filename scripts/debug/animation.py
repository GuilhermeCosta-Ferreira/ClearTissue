# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib import pyplot as plt

from tqdm import tqdm
from scipy.ndimage import binary_erosion, zoom
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import binary_opening, generate_binary_structure, binary_dilation

from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject



# ================================================================
# 1. Section: INPUTS
# ================================================================



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def get_density(
    points: np.ndarray,
    density_radius: float,
):
    tree = KDTree(points)
    densities = np.array(
        [len(tree.query_ball_point(p, r=density_radius)) - 1 for p in points]
    )

    return densities

def get_descendant_ids_from_root(
    atlas_original: BrainGlobeAtlas,
    root_id: int,
) -> set[int]:
    descendant_ids = set()

    for structure_id, structure_info in atlas_original.structures.items():
        structure_path = structure_info.get("structure_id_path", [])
        if root_id in structure_path:
            descendant_ids.add(int(structure_id))

    descendant_ids.add(root_id)

    return descendant_ids





# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    project = ClearTissueProject.load(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )

    atlas_name = project.source.atlas_name
    atlas_original = BrainGlobeAtlas(atlas_name)

    os.makedirs("frames/", exist_ok=True)

    final_batch = project.io.load_batch(pipeline_id=11, step=8)

    atlas = final_batch.atlas.data
    cells = final_batch.cells.data
    look_up = final_batch.atlas.look_up


    target_red = "#F9525B"  # replace with your specific red hex

    cell_cmap = LinearSegmentedColormap.from_list(
        "cell_red_alpha",
        [
            (1.0, 1.0, 1.0, 0.1),   # light red, fully transparent
            (0.976, 0.322, 0.357, 1.0),   # same red, fully opaque
        ],
    )

    outline_cmap = LinearSegmentedColormap.from_list(
        "outline_red_alpha",
        [
            (0, 0, 0),   # light red, fully transparent
            (0.0, 0.0, 0.0),   # same red, fully opaque
        ],
    )

    start = 10
    for i in range(atlas.shape[0]):
        if np.sum(atlas[i, :, :]) > 0:
            start = i
            break

    missing_frames = atlas.shape[0] - start
    labels = np.unique(atlas)
    colour_outline = "#000000"

    gm_root_id = 71
    wm_root_id = 130

    gm_ids = get_descendant_ids_from_root(atlas_original, gm_root_id)
    wm_ids = get_descendant_ids_from_root(atlas_original, wm_root_id)

    # Keep only cells inside the atlas region
    cells_inside_atlas = np.where(atlas > 0, cells, 0)

    # Binary cell mask
    cell_mask = cells_inside_atlas > 0

    # Keep original cell values only where the opened mask survives
    cells_clean = np.where(atlas > 0, cells, 0)

    cells_transparent = np.where(cells_clean > 0, cells_clean, np.nan)
    atlas_transparent = np.where(atlas > 0, atlas, np.nan)

    for added in tqdm(range(missing_frames)):
        idx = start + added

        # Current atlas slice
        atlas_slice = atlas_transparent[idx, :, :]
        offset = idx + int(final_batch.cells.shape[1]//4)

        cell_slice = np.nansum(cells_transparent[idx:offset, :, :], axis=0)
        cell_slice = gaussian_filter(cell_slice, 1.2)
        cell_slice = np.where(cell_slice > 0, cell_slice, np.nan)
        cell_slice = np.where(atlas_slice, cell_slice, np.nan)

        atlas_slice_filled = np.nan_to_num(atlas_slice, nan=0)
        grad_y, grad_x = np.gradient(atlas_slice_filled.astype(float))
        atlas_outline = np.sqrt(grad_x**2 + grad_y**2)
        atlas_outline = np.where(atlas_outline > 0, 1, np.nan)

        gm_mask = np.isin(atlas_slice, list(gm_ids))
        wm_mask = np.isin(atlas_slice, list(wm_ids))

        gm_colour = (0.5, 0.5, 0.5, 1.0)
        wm_colour = (0.8, 0.8, 0.8, 1.0)

        gm_rgba = np.zeros((*atlas_slice.shape, 4), dtype=float)
        wm_rgba = np.zeros((*atlas_slice.shape, 4), dtype=float)

        gm_rgba[gm_mask] = gm_colour
        wm_rgba[wm_mask] = wm_colour

        plt.figure()
        plt.imshow(gm_rgba, interpolation="nearest")
        plt.imshow(wm_rgba, interpolation="nearest")
        for label in labels:
            mask = atlas_slice == label

            if not np.any(mask):
                continue

            plt.contour(
                mask.astype(float),
                levels=[0.5],
                colors=[colour_outline],
                linewidths=0.5,
                antialiased=True,
            )
        plt.imshow(cell_slice, cmap=cell_cmap, interpolation="nearest")
        plt.axis("off")
        plt.savefig(
            f"frames/frame_{idx}.png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
            dpi=1000,
        )
        plt.close()
