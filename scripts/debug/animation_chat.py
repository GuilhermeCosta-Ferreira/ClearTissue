# ================================================================
# 0. Section: IMPORTS
# ================================================================
import json
from pathlib import Path

import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    gaussian_filter,
    generate_binary_structure,
)

from cleartissue.domain_model.data import TissueType
from cleartissue.service.ClearTissueProject import ClearTissueProject


# ================================================================
# 1. Section: INPUTS
# ================================================================
OUTPUT_DIR = Path("frames")

MOUSE = "32B"
TISSUE_TYPE = TissueType.SPINAL_CORD
PIPELINE_ID = 11
STEP = 8

GM_ROOT_ID = 71
WM_ROOT_ID = 130

CELL_RED = "#F9525B"

GM_COLOUR = "#707070"
WM_COLOUR = "#C8C8C8"

ATLAS_OUTLINE_COLOUR = "#FFD700"  # outer atlas outline, yellow/gold
GM_OUTLINE_COLOUR = "#202020"
WM_OUTLINE_COLOUR = "#202020"

CELL_SMOOTH_SIGMA = 1.2
OPENING_ITERATIONS = 1
DILATION_ITERATIONS = 1

DPI = 1000


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def get_first_non_empty_slice(volume: np.ndarray) -> int:
    non_empty_slices = np.where(np.any(volume > 0, axis=(1, 2)))[0]

    if len(non_empty_slices) == 0:
        return 0

    return int(non_empty_slices[0])


def get_descendant_ids_from_root(
    atlas_original: BrainGlobeAtlas,
    root_id: int,
) -> set[int]:
    descendant_ids: set[int] = set()

    for structure_id, structure_info in atlas_original.structures.items():
        structure_path = structure_info.get("structure_id_path", [])
        structure_path = [int(item) for item in structure_path]

        if root_id in structure_path:
            descendant_ids.add(int(structure_id))

    descendant_ids.add(root_id)

    return descendant_ids


def make_rgba_layer(
    mask: np.ndarray,
    colour: str | tuple[float, float, float, float],
) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask] = to_rgba(colour)

    return rgba


def make_outline_mask(
    mask: np.ndarray,
    iterations: int = 1,
) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    structure = np.ones((3, 3), dtype=bool)

    eroded = binary_erosion(
        mask,
        structure=structure,
        iterations=iterations,
        border_value=0,
    )

    return mask & ~eroded


def make_cell_cmap(cell_red: str) -> LinearSegmentedColormap:
    red_rgba = to_rgba(cell_red, alpha=1.0)

    cmap = LinearSegmentedColormap.from_list(
        "cell_red_alpha",
        [
            (1.0, 1.0, 1.0, 0.0),
            red_rgba,
        ],
    )

    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    return cmap


def clean_cells(
    atlas: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    cells_inside_atlas = np.where(atlas > 0, cells, 0)

    cell_mask = cells_inside_atlas > 0

    structure = generate_binary_structure(
        rank=3,
        connectivity=1,
    )

    cell_mask_opened = binary_opening(
        cell_mask,
        structure=structure,
        iterations=OPENING_ITERATIONS,
    )

    cell_mask_opened = binary_dilation(
        cell_mask_opened,
        structure=structure,
        iterations=DILATION_ITERATIONS,
    )

    return np.where(cell_mask_opened, cells_inside_atlas, 0)


def make_projected_cell_slice(
    cells_transparent: np.ndarray,
    atlas_slice_mask: np.ndarray,
    z_start: int,
    z_end: int,
) -> np.ndarray:
    cell_slice = np.nansum(
        cells_transparent[z_start:z_end, :, :],
        axis=0,
    )

    cell_slice = gaussian_filter(
        cell_slice,
        sigma=CELL_SMOOTH_SIGMA,
    )

    cell_slice = np.where(cell_slice > 0, cell_slice, np.nan)
    cell_slice = np.where(atlas_slice_mask, cell_slice, np.nan)

    return cell_slice


def save_frame(
    output_path: Path,
    gm_rgba: np.ndarray,
    wm_rgba: np.ndarray,
    cell_slice: np.ndarray,
    cell_cmap: LinearSegmentedColormap,
    atlas_outline_rgba: np.ndarray,
    gm_outline_rgba: np.ndarray,
    wm_outline_rgba: np.ndarray,
) -> None:
    fig, ax = plt.subplots()

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.imshow(gm_rgba, interpolation="nearest")
    ax.imshow(wm_rgba, interpolation="nearest")

    ax.imshow(
        cell_slice,
        cmap=cell_cmap,
        interpolation="nearest",
    )

    # Outlines are plotted last so their exact colours remain visible.
    ax.imshow(gm_outline_rgba, interpolation="nearest")
    ax.imshow(wm_outline_rgba, interpolation="nearest")
    ax.imshow(atlas_outline_rgba, interpolation="nearest")

    ax.axis("off")

    fig.savefig(
        output_path,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
        dpi=DPI,
    )

    #plt.close(fig)


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    project = ClearTissueProject.load(
        mouse=MOUSE,
        tissue_type=TISSUE_TYPE,
    )

    atlas_name = project.source.atlas_name
    atlas_original = BrainGlobeAtlas(atlas_name)

    final_batch = project.io.load_batch(
        pipeline_id=PIPELINE_ID,
        step=STEP,
    )

    atlas = final_batch.atlas.data
    cells = final_batch.cells.data
    look_up = final_batch.atlas.look_up

    print(look_up.head())

    gm_ids = get_descendant_ids_from_root(
        atlas_original=atlas_original,
        root_id=GM_ROOT_ID,
    )

    wm_ids = get_descendant_ids_from_root(
        atlas_original=atlas_original,
        root_id=WM_ROOT_ID,
    )

    cell_cmap = make_cell_cmap(CELL_RED)

    idx = get_first_non_empty_slice(atlas)

    cells_clean = clean_cells(
        atlas=atlas,
        cells=cells,
    )

    cells_transparent = np.where(
        cells_clean > 0,
        cells_clean,
        np.nan,
    )

    # Same idea as your original projection depth, but using the z-axis.
    z_projection_depth = max(1, cells.shape[0] // 4)

    frame_map = []
    frame_number = 0

    for z in range(idx, atlas.shape[0]):
        atlas_slice = atlas[z, :, :]

        if not np.any(atlas_slice > 0):
            continue

        z_end = min(
            z + z_projection_depth,
            cells.shape[0],
        )

        atlas_mask = atlas_slice > 0
        gm_mask = np.isin(atlas_slice, list(gm_ids))
        wm_mask = np.isin(atlas_slice, list(wm_ids))

        cell_slice = make_projected_cell_slice(
            cells_transparent=cells_transparent,
            atlas_slice_mask=atlas_mask,
            z_start=z,
            z_end=z_end,
        )

        gm_rgba = make_rgba_layer(
            mask=gm_mask,
            colour=GM_COLOUR,
        )

        wm_rgba = make_rgba_layer(
            mask=wm_mask,
            colour=WM_COLOUR,
        )

        atlas_outline_mask = make_outline_mask(
            atlas_slice,
            iterations=1,
        )

        gm_outline_mask = make_outline_mask(
            gm_mask,
            iterations=1,
        )

        wm_outline_mask = make_outline_mask(
            wm_mask,
            iterations=1,
        )

        atlas_outline_rgba = make_rgba_layer(
            mask=atlas_outline_mask,
            colour=ATLAS_OUTLINE_COLOUR,
        )

        gm_outline_rgba = make_rgba_layer(
            mask=gm_outline_mask,
            colour=GM_OUTLINE_COLOUR,
        )

        wm_outline_rgba = make_rgba_layer(
            mask=wm_outline_mask,
            colour=WM_OUTLINE_COLOUR,
        )

        output_path = OUTPUT_DIR / f"{frame_number:04d}.png"

        save_frame(
            output_path=output_path,
            gm_rgba=gm_rgba,
            wm_rgba=wm_rgba,
            cell_slice=cell_slice,
            cell_cmap=cell_cmap,
            atlas_outline_rgba=atlas_outline_rgba,
            gm_outline_rgba=gm_outline_rgba,
            wm_outline_rgba=wm_outline_rgba,
        )

        frame_map.append(
            {
                "frame": frame_number,
                "z_slice": int(z),
                "z_projection_start": int(z),
                "z_projection_end": int(z_end),
            }
        )

        print(f"Saved {output_path} from z={z}")

        frame_number += 1

    with (OUTPUT_DIR / "frame_map.json").open("w") as file:
        json.dump(frame_map, file, indent=2)

    print(f"Saved {frame_number} frames to {OUTPUT_DIR}")
