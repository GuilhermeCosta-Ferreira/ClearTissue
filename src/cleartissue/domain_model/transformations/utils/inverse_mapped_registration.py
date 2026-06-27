# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm
from typing import cast
from numpy.typing import NDArray

from ....registration import Registrator
from ...data import Atlas, ClearVolume, ClearData



# ================================================================
# 1. Section: Functions
# ================================================================
def register_sample_to_atlas(
    atlas: Atlas,
    tissue: ClearVolume,
    cells: ClearVolume,
    atlas_index_map: NDArray,
    affine_registrator: Registrator,
    warp_registrator: Registrator,
    max_retries: int
) -> tuple[ClearVolume, ClearVolume]:
    registered_tissue = np.zeros_like(atlas.data)
    registered_cells = np.zeros_like(atlas.data, dtype=np.float32)

    previous_affine_parameters = None
    previous_affine_fixed_parameters = None
    previous_warp_parameters = None
    previous_warp_fixed_parameters = None

    for i in tqdm(range(tissue.shape[0]), total=len(atlas_index_map)):
        # 6.1 Find the corresponding slices
        atlas_idx = atlas_index_map[i]
        if atlas_idx is None or np.isnan(atlas_idx):
            continue
        atlas_idx = int(atlas_idx)

        affine_registrator.config.optimizer.initial_parameters = previous_affine_parameters
        affine_registrator.config.optimizer.initial_fixed_parameters = previous_affine_fixed_parameters

        # 6.2. Extract those slices from the atlas and tissue
        atlas_slice = atlas.data[atlas_idx, :, :]
        cell_slice = cells.data[i, :, :]
        tissue_slice = tissue.data[i, :, :]

        # 6.4 Register the template slice to the sample slice
        affine_result = affine_registrator.register(atlas_slice, tissue_slice)
        affine_cells = affine_registrator.apply(
            atlas_slice,
            cell_slice,
            affine_result.transform,
            as_array=True
        )

        # 6.5. Clear the sample slice and affine template for warp registration
        clear_affine_tissue = np.where(affine_result.registered_image > 0, affine_result.registered_image, 0)
        clear_atlas_slice = np.where(atlas_slice > 0, atlas_slice, 0)

        best_metric = -float('inf')
        best_warp_tissue = np.zeros_like(affine_result.registered_image)
        best_warp_cells = np.zeros_like(affine_cells)

        warp_registrator.config.optimizer.initial_parameters = previous_warp_parameters
        warp_registrator.config.optimizer.initial_fixed_parameters = previous_warp_fixed_parameters

        for _ in range(max_retries + 1):
            # 6.6. Apply the warp transform to the template and hemisphere
            warp_result = warp_registrator.register(clear_atlas_slice, clear_affine_tissue)
            warp_transform = warp_result.transform
            warp_tissue = warp_registrator.apply(
                atlas_slice,
                affine_result.registered_image,
                warp_transform,
                as_array=True
            )
            warp_cells = warp_registrator.apply(
                atlas_slice,
                affine_cells,
                warp_transform,
                as_array=True
            )

            best_metric = max(best_metric, warp_result.final_metric)

            if warp_result.final_metric >= best_metric:
                best_warp_tissue = warp_tissue
                best_warp_cells = warp_cells

            # 6.7. Re-apply the warp if the final metric is too high
            if best_metric < -0.01:
                break

        # 6.8. Register the warp-transformed template and hemisphere to the atlas volume
        registered_tissue[atlas_idx, :, :] = best_warp_tissue
        registered_cells[atlas_idx, :, :] = best_warp_cells


    return tissue.copy_with(data=registered_tissue), cells.copy_with(data=registered_cells)



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_data_shape_array(data: ClearData) -> NDArray:
    data_size = data.data.shape[0]
    return np.arange(data_size) / data_size

def find_on_b(value: float, template_b_array: NDArray) -> int:
    idx = np.searchsorted(template_b_array, value)
    return cast(int, idx)
