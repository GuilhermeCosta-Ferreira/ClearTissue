# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from tqdm import tqdm
from typing import cast
from pathlib import Path
from numpy.typing import NDArray
from brainglobe_atlasapi import BrainGlobeAtlas

from clearbrain.tissue import TissueType
from clearbrain.tissue.ClearVolume import ClearVolume
from clearbrain.registration import Registrator, RegistratorResampler
from clearbrain.data import TissueLoader, TissueSource, TissueDownloader

from clearbrain.registration.configs import (
    TEMPLATE_WARP_REGISTRATION,
    TEMPLATE_AFFINE_REGISTRATION,
)
from clearbrain.registration.strategies import (
    BSplineRegistration,
    AffineRegistration,
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD

MAX_RETRIES: int = 5



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def find_on_b(value: float, template_b_array: NDArray) -> int:
    idx = np.searchsorted(template_b_array, value)
    return cast(int, idx)



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Import the atlas and get important variables
    atlas = BrainGlobeAtlas("allen_cord_20um", check_latest=False)
    template = atlas.annotation
    hemisphere = atlas.hemispheres

    # 2. Load the sample data
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    tissue = loader.load_volume(suffix="_tissue_untwisted_cleaned", sample_factor=1)

    # 3. Compute the slice mapping between tissue and template
    tissue_size = tissue.volume.shape[1]
    tissue_a_array = np.arange(tissue_size) / tissue_size
    template_size = template.shape[0]
    template_b_array = np.arange(template_size) / template_size

    # 4. Build the registrators
    affine_registrator = Registrator(
        strategy=AffineRegistration(),
        resampler=RegistratorResampler(),
        config=TEMPLATE_AFFINE_REGISTRATION,
    )
    warp_registrator = Registrator(
        strategy=BSplineRegistration(),
        resampler=RegistratorResampler(),
        config=TEMPLATE_WARP_REGISTRATION,
    )

    # 5. Initialize the registered atlas and hemisphere
    registered_atlas = np.zeros_like(tissue.volume)
    registered_hemisphere = np.zeros_like(tissue.volume)
    registration_data = []

    # 6. Register each slice
    picked_template_slices = []
    for i, value in tqdm(enumerate(tissue_a_array), total=len(tissue_a_array)):
        # 6.1 Find the corresponding slices
        idx = find_on_b(value, template_b_array)
        picked_template_slices.append(idx)
        sample_slice = tissue.volume[:, i, :]
        template_slice = template[idx, :, :]
        hemisphere_slice = hemisphere[idx, :, :]

        # 6.2 Reduce the template size by half
        template_slice = template_slice[::2, ::2]
        hemisphere_slice = hemisphere_slice[::2, ::2]

        # 6.3 Rotate the sample slice 180 degrees
        sample_slice = np.rot90(sample_slice, k=2)
        sample_slice = np.where(sample_slice > 50, sample_slice, 0)

        # 6.4 Register the template slice to the sample slice
        affine_result = affine_registrator.register(sample_slice, template_slice)
        affine_hemisphere = affine_registrator.apply(sample_slice, hemisphere_slice, affine_result.transform, as_array=True)

        # 6.5. Clear the sample slice and affine template for warp registration
        clear_affine_template = np.where(affine_result.registered_image > 0, 1, 0)
        clear_sample_slice = np.where(sample_slice > 0, 1, 0)

        best_metric = -float('inf')
        best_warp_template = np.zeros_like(affine_result.registered_image)
        best_warp_hemisphere = np.zeros_like(affine_hemisphere)
        best_transform = None
        best_warp_result = None
        for _ in range(MAX_RETRIES + 1):
            # 6.6. Apply the warp transform to the template and hemisphere
            warp_result = warp_registrator.register(clear_sample_slice, clear_affine_template)
            warp_transform = warp_result.transform
            warp_template = warp_registrator.apply(sample_slice, affine_result.registered_image, warp_transform, as_array=True)
            warp_hemisphere = warp_registrator.apply(sample_slice, affine_hemisphere, warp_transform, as_array=True)

            best_metric = max(best_metric, warp_result.final_metric)

            if warp_result.final_metric >= best_metric:
                best_warp_template = warp_template
                best_warp_hemisphere = warp_hemisphere
                best_transform = warp_transform
                best_warp_result = warp_result

            # 6.7. Re-apply the warp if the final metric is too high
            if best_metric < -0.01:
                break

        # 6.8. Register the warp-transformed template and hemisphere to the atlas volume
        best_warp_template = np.rot90(np.asarray(best_warp_template), k=2)
        best_warp_hemisphere = np.rot90(np.asarray(best_warp_hemisphere), k=2)
        registered_atlas[:, i, :] = best_warp_template
        registered_hemisphere[:, i, :] = best_warp_hemisphere
        registration_data.append((affine_result, best_warp_result))

    registered_atlas_volume = ClearVolume(
        volume=registered_atlas,
        metadata=tissue.metadata,
        sample_factor=tissue.sample_factor,
    )

    registered_hemisphere_volume = ClearVolume(
        volume=registered_hemisphere,
        metadata=tissue.metadata,
        sample_factor=tissue.sample_factor,
    )

    downloader = TissueDownloader(source)
    p = downloader.download_volume(registered_atlas_volume, suffix="_registered_atlas", to_update=True)
    print(f"Downloaded registered atlas volume to {p}")

    p = downloader.download_volume(registered_hemisphere_volume, suffix="_registered_hemisphere", to_update=True)
    print(f"Downloaded registered hemisphere volume to {p}")

    p = downloader.download_twisting_data(registration_data, suffix="_atlas_registration_data", to_update=True)
    print(f"Downloaded registration data to {p}")
