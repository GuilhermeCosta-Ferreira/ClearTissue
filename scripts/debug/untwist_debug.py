# ================================================================
# 0. Section: IMPORTS
# ================================================================
from copy import deepcopy

import zarr

import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from typing import cast

from clearbrain.tissue import ClearTissue, TissueType, ClearVolume
from clearbrain.data import TissueLoader, TissueDownloader, TissueSource

from clearbrain.processing import (
    get_centerline,
    scale_tissue,
    compress_to_volume,
    stretch_tissue,
    untwist_spinal_coord,
    clear_external_points
)
from clearbrain.tissue.view import (
    plot_spinal_direction,
    plot_volume_coronal,
    plot_volume_overview,
)
from clearbrain.registration import (
    RegistrationConfig,
    Registrator,
    RegistratorResampler,
    RigidRegistration,
)



# ================================================================
# 1. Section: INPUTS
# ================================================================
ZARR_PATH: Path = Path("/Volumes/GuiNR/Transfer/561_CFos_raw.zarr")

DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD

SCALING: tuple[float, float, float] = (2.22, 1.0, 1.0)
WINDOW_SIZE: int = 14

SMOOTH_WINDOW_SIZE: int = 25

TO_SAVE: bool = False



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def print_zarr_tree(group, prefix: str = "") -> None:
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}" if prefix else key

        if hasattr(item, "shape"):
            print(f"ARRAY: {path}")
            print(f"  shape:  {item.shape}")
            print(f"  dtype:  {item.dtype}")
            print(f"  chunks: {item.chunks}")
            print()

        else:
            print(f"GROUP: {path}")
            print_zarr_tree(item, path)

def build_volume(
    points: np.ndarray,
    volume_shape: tuple[int, int, int] | np.ndarray,
    window_size: int,
) -> np.ndarray:
    downsampled_points = (points // window_size).astype(int)

    volume_shape = np.asarray(volume_shape, dtype=int)

    valid_mask = np.all(
        (downsampled_points >= 0) & (downsampled_points < volume_shape),
        axis=1,
    )

    valid_points = downsampled_points[valid_mask]

    volume = np.zeros(volume_shape, dtype=int)

    for p in valid_points:
        volume[p[0], p[1], p[2]] += 1

    return volume


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    tissue = loader.load_points(suffix="_scaled_full")

    print("================================================================")
    print(f"Working with tissue from file: {str(source.source_filepath)}")
    print("================================================================")
    print(f"Range of X: {np.min(tissue.points[:,0])} - {np.max(tissue.points[:,0])}")
    print(f"Range of Y: {np.min(tissue.points[:,1])} - {np.max(tissue.points[:,1])}")
    print(f"Range of Z: {np.min(tissue.points[:,2])} - {np.max(tissue.points[:,2])}\n")

    stretch_tissue = loader.load_volume(suffix="_stretch_full", sample_factor=1)

    temp_vol = stretch_tissue.volume
    #temp_vol[:, :16, :] = 0
    stretch_tissue.volume = temp_vol

    cleaned_tissue = deepcopy(stretch_tissue)
    """ cleaned_tissue = clear_external_points(stretch_tissue)
    plot_volume_coronal(cleaned_tissue, 20)
    plt.show() """

    # 6. Applies the untwisting of the coord
    registrator = Registrator(
        strategy=RigidRegistration(),
        resampler=RegistratorResampler(),
        config=RegistrationConfig(),
    )

    registrator.config.metric.name = "CC"
    registrator.config.metric.sampling_percentage = 1
    registrator.config.metric.histogram_bins = 50

    registrator.config.optimizer.iterations = 500
    registrator.config.optimizer.convergence_minimum_value = 1e-8
    registrator.config.optimizer.convergence_window_size = 30

    #registrator.config.multiple_res.enabled = True
    #cleaned_tissue.volume = cleaned_tissue.volume[:, :100, :]

    untwisted_tissue, twisting_data = untwist_spinal_coord(cleaned_tissue, registrator, window_size=250, gap=0)
    plot_volume_coronal(untwisted_tissue, 10, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(untwisted_tissue, 3, is_save=TO_SAVE)
    density_before = np.sum(cleaned_tissue.volume)
    density_after = np.sum(untwisted_tissue.volume)
    percentage_change = (density_after - density_before) / density_before * 100
    print(
        "Tissue has been untwisted, with stable density:\n"
        f"  Sum Density before: {density_before}\n"
        f"  Sum Density after:  {density_after}\n"
        f"  Percentage change:  {percentage_change:.2f}%\n"
    )
    plt.show(block=True)

    angles = np.asarray([result.transform.GetParameters()[0] for result in twisting_data])
    timesteps = np.asarray([result.elapsed_time for result in twisting_data])
    images = np.asarray([result.registered_image for result in twisting_data])

    plt.figure()
    plt.plot(angles)
    #plt.plot(timesteps)
    plt.xlabel("Coronal Axis")
    plt.ylabel("Angle")
    plt.show(block=False)

    plt.figure()
    plt.plot(timesteps)
    plt.xlabel("Coronal Axis")
    plt.ylabel("Time")
    plt.show(block=False)

    # gradient
    plt.figure()
    plt.plot(angles[1:] - angles[:-1])
    plt.show(block=False)

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(images[13])
    plt.subplot(2, 3, 2)
    plt.imshow(images[14])
    plt.subplot(2, 3, 3)
    plt.imshow(images[15])
    plt.subplot(2, 3, 4)
    plt.imshow(images[16])
    plt.subplot(2, 3, 5)
    plt.imshow(images[17])
    plt.subplot(2, 3, 6)
    plt.imshow(images[18])
    plt.show(block=False)

    input("Press Enter to close...")
    plt.close('all')

    downloader = TissueDownloader(source)
    p = downloader.download_points(tissue, suffix="_scaled_full", to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_metadata(tissue.metadata, to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_volume(stretch_tissue, suffix="_stretch_full", to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_twisting_data(twisting_data, to_update=True)
    print(f"Downloaded at {p}")

    p = downloader.download_volume(
        untwisted_tissue, suffix="_untwisted_full", to_update=True
    )
    print(f"Downloaded at {p}")
