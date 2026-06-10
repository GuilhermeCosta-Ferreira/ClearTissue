# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from typing import cast
from pathlib import Path
from brainglobe_atlasapi import BrainGlobeAtlas
from numpy.typing import NDArray
from tqdm import tqdm

from clearbrain.data.TissueDownloader import TissueDownloader
from clearbrain.tissue import TissueType
from clearbrain.data import TissueLoader, TissueSource
from clearbrain.registration import Registrator, RegistratorResampler
from clearbrain.registration.configs import TEMPLATE_WARP_REGISTRATION, TEMPLATE_AFFINE_REGISTRATION
from clearbrain.registration.strategies import BSplineRegistration, AffineRegistration
from clearbrain.tissue.ClearVolume import ClearVolume



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def find_on_b(value: float, template_b_array: NDArray) -> int:
    idx = np.searchsorted(template_b_array, value)
    return cast(int, idx)

def mask_to_distance_map(mask: sitk.Image) -> sitk.Image:
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)

    distance = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )

    return sitk.Cast(distance, sitk.sitkFloat32)



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    atlas = BrainGlobeAtlas("allen_cord_20um", check_latest=False)
    template = atlas.annotation
    hemispheres = atlas.hemispheres

    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    tissue = loader.load_volume(suffix="_tissue_untwisted_cleaned", sample_factor=1)

    tissue.volume = tissue.volume[:, 49:, :]

    tissue_size = tissue.volume.shape[1]
    tissue_a_array = np.arange(tissue_size) / tissue_size

    template_size = template.shape[0]
    template_b_array = np.arange(template_size) / template_size

    diff = template_size - tissue_size
    margin_per_slice = np.round(diff / tissue_size).astype(np.int16)

    # prints
    """ print(f"tissue_size: {tissue_size}")
    print(f"template_size: {template_size}")
    print(f"diff: {diff}")
    print(f"margin_per_slice: {margin_per_slice}")
    print(f"tissue_a_array: {tissue_a_array}")
    print(f"template_b_array: {template_b_array}") """

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

    registered_atlas = np.zeros_like(tissue.volume)
    registered_hemispheres = np.zeros_like(tissue.volume)
    for i, value in tqdm(enumerate(tissue_a_array), total=len(tissue_a_array)):
        if i != 576:
            i = 576
            value = tissue_a_array[i]
        idx = find_on_b(value, template_b_array)
        sample_slice = tissue.volume[:, i, :]
        template_slice = template[idx, :, :]
        hemispheres_slice = hemispheres[idx, :, :]

        # reduce the template size by half
        template_slice = template_slice[::2, ::2]
        hemispheres_slice = hemispheres_slice[::2, ::2]

        # Rotate the sample slice 180 degrees
        sample_slice = np.rot90(sample_slice, k=2)
        sample_slice = np.where(sample_slice > 50, sample_slice, 0)

        print(f"tissue_a_array[{i}] = {value} -> template_b_array[{idx}] = {template_b_array[idx]}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_slice, label="tissue_a_array")
        plt.subplot(1, 2, 2)
        plt.imshow(template_slice, label="template_b_array")
        plt.title(f"Before atlas registration, slice {i}")
        plt.show()

        affine_result = affine_registrator.register(sample_slice, template_slice)
        affine_hemisphere = affine_registrator.apply(sample_slice, hemispheres_slice, affine_result.transform, as_array=True)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_slice, label="tissue_a_array")
        plt.subplot(1, 2, 2)
        plt.imshow(sample_slice, label="sample_slice")
        plt.imshow(affine_result.registered_image, label="registered_image", alpha=0.5)
        plt.title(
            f"After atlas registration, slice {i}\n"
            f"registered_image shape: {affine_result.registered_image.shape}\n"
            f"stop_condition: {affine_result.stop_condition}\n"
            f"final_metric: {affine_result.final_metric}"
        )
        plt.show()

        clear_sample_slice = np.where(sample_slice > 0, 1, 0)
        clear_affine_template = np.where(affine_result.registered_image > 0, 1, 0)

        warp_result = warp_registrator.register(sample_slice, clear_affine_template)
        warp_transform = warp_result.transform

        warp_template = warp_registrator.apply(clear_sample_slice, affine_result.registered_image, warp_transform, as_array=True)
        warp_hemispheres = warp_registrator.apply(clear_sample_slice, affine_hemisphere, warp_transform, as_array=True)

        registered_atlas[:, i, :] = warp_template
        registered_hemispheres[:, i, :] = warp_hemispheres

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_slice, label="tissue_a_array")
        plt.subplot(1, 2, 2)
        plt.imshow(sample_slice, label="sample_slice")
        plt.imshow(warp_result.registered_image, label="registered_image", alpha=0.5)
        plt.title(
            f"After warp registration, slice {i}\n"
            f"registered_image shape: {warp_result.registered_image.shape}\n"
            f"stop_condition: {warp_result.stop_condition}\n"
            f"final_metric: {warp_result.final_metric}"
        )
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_slice, label="tissue_a_array")
        plt.subplot(1, 2, 2)
        plt.imshow(sample_slice, label="sample_slice")
        plt.imshow(warp_template, label="registered_image", alpha=0.5)
        plt.title(
            f"After warp registration, slice {i}\n"
            f"registered_image shape: {warp_template.shape}\n"
            f"stop_condition: {warp_result.stop_condition}\n"
            f"final_metric: {warp_result.final_metric}"
        )
        plt.show()
        print(f"Overlap: {np.sum(sample_slice * warp_template)}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_slice, label="tissue_a_array")
        plt.subplot(1, 2, 2)
        plt.imshow(sample_slice, label="sample_slice")
        plt.imshow(warp_hemispheres, label="registered_hemispheres", alpha=0.5)
        plt.title(
            f"After warp registration, slice {i}\n"
            f"registered_hemispheres shape: {warp_hemispheres.shape}\n"
        )
        plt.show()
        print(f"Overlap: {np.sum(sample_slice * warp_hemispheres)}")

        if np.abs(warp_result.final_metric) < 0.01:
            pass
