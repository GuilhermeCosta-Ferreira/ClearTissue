# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from tqdm import tqdm
from typing import cast
from numpy.typing import NDArray

from ....registration import Registrator
from ...data import Atlas, ClearVolume, ClearData



# ================================================================
# 1. Section: Functions
# ================================================================
def register_atlas_to_sample(
    atlas: Atlas,
    tissue: ClearVolume,
    atlas_index_map: NDArray,
    affine_registrator: Registrator,
    warp_registrator: Registrator,
    max_retries: int
) -> Atlas:
    registered_atlas = np.zeros_like(tissue.data)
    registered_hemisphere = np.zeros_like(tissue.data)

    for i in tqdm(range(tissue.shape[0]), total=len(atlas_index_map)):
        # 6.1 Find the corresponding slices
        atlas_idx = atlas_index_map[i]
        if atlas_idx is None or np.isnan(atlas_idx):
            continue
        atlas_idx = int(atlas_idx)

        # 6.2. Extract those slices from the atlas and tissue
        template_slice = atlas.data[atlas_idx, :, :]
        hemisphere_slice = atlas.hemisphere[atlas_idx, :, :]
        tissue_slice = tissue.data[i, :, :]

        # 6.4 Register the template slice to the sample slice
        affine_result = affine_registrator.register(tissue_slice, template_slice)
        affine_hemisphere = affine_registrator.apply(
            tissue_slice,
            hemisphere_slice,
            affine_result.transform,
            as_array=True
        )

        # 6.5. Clear the sample slice and affine template for warp registration
        clear_affine_template = np.where(affine_result.registered_image > 0, 1, 0)
        clear_sample_slice = np.where(tissue_slice > 0, 1, 0)

        best_metric = -float('inf')
        best_warp_template = np.zeros_like(affine_result.registered_image)
        best_warp_hemisphere = np.zeros_like(affine_hemisphere)
        for _ in range(max_retries + 1):
            # 6.6. Apply the warp transform to the template and hemisphere
            warp_result = warp_registrator.register(clear_sample_slice, clear_affine_template)
            warp_transform = warp_result.transform
            warp_template = warp_registrator.apply(
                tissue_slice,
                affine_result.registered_image,
                warp_transform,
                as_array=True
            )
            warp_hemisphere = warp_registrator.apply(
                tissue_slice,
                affine_hemisphere,
                warp_transform,
                as_array=True
            )

            best_metric = max(best_metric, warp_result.final_metric)

            if warp_result.final_metric >= best_metric:
                best_warp_template = warp_template
                best_warp_hemisphere = warp_hemisphere

            # 6.7. Re-apply the warp if the final metric is too high
            if best_metric < -0.01:
                break

        # 6.8. Register the warp-transformed template and hemisphere to the atlas volume
        registered_atlas[i, :, :] = best_warp_template
        registered_hemisphere[i, :, :] = best_warp_hemisphere

    return atlas.copy_with(data=registered_atlas, hemisphere=registered_hemisphere)



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_data_shape_array(data: ClearData) -> NDArray:
    data_size = data.data.shape[0]
    return np.arange(data_size) / data_size

def find_on_b(value: float, template_b_array: NDArray) -> int:
    idx = np.searchsorted(template_b_array, value)
    return cast(int, idx)
