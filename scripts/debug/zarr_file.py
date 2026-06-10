# ================================================================
# 0. Section: IMPORTS
# ================================================================
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
MOUSE: str = "32B_old"
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
    print(f"Zarr version: {zarr.__version__}")

    root = zarr.open_group(ZARR_PATH, mode="r")

    print(root)
    print()
    print_zarr_tree(root)

    arr = cast(zarr.Array, root["level_03"])

    print(arr.shape)
    print(arr.dtype)
    print(arr.chunks)

    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    tissue = loader.load_points()

    print("================================================================")
    print(f"Working with tissue from file: {str(source.source_filepath)}")
    print("================================================================")
    print(f"Range of X: {np.min(tissue.points[:,0])} - {np.max(tissue.points[:,0])}")
    print(f"Range of Y: {np.min(tissue.points[:,1])} - {np.max(tissue.points[:,1])}")
    print(f"Range of Z: {np.min(tissue.points[:,2])} - {np.max(tissue.points[:,2])}\n")

    shape_01 = np.array(root["level_01"].shape)
    shape_03 = np.array(root["level_03"].shape)

    scaling_factor = tuple(shape_03 / shape_01)

    tissue = cast(ClearTissue, scale_tissue(tissue, scaling_factor))

    print(f"Scaling applied with scaling of: {scaling_factor}")
    print(f"Range of X: {np.min(tissue.points[:,0])} - {np.max(tissue.points[:,0])}")
    print(f"Range of Y: {np.min(tissue.points[:,1])} - {np.max(tissue.points[:,1])}")
    print(f"Range of Z: {np.min(tissue.points[:,2])} - {np.max(tissue.points[:,2])}\n")

    # 3.Compress into a volume (instead a list of points)
    vol_tissue_arr = build_volume(tissue.points, shape_03, 1)
    vol_tissue = ClearVolume(vol_tissue_arr, tissue.metadata, 1)
    print(
        f"Volume has now shape: {vol_tissue.volume.shape}, thanks to a window of size: {1}\n"
    )

    slice_img = arr[:, 500, :].astype(np.uint16)
    #slice_img = np.where(slice_img > 0, 1, 0)

    cell_img = np.where(vol_tissue.volume[:, 500, :] > 0, vol_tissue.volume[:, 500, :], np.nan)
    #cell_img = np.where(vol_tissue.volume[:, 500, :] > 0, 1, np.nan)

    plt.figure()
    plt.imshow(slice_img, cmap="gray")
    plt.show(block=False)



    plt.figure()
    plt.imshow(slice_img, cmap="gray")
    plt.imshow(cell_img, cmap="cool")
    plt.show()

    arr = np.where(arr[:, :, :] > 20, arr[:, :, :], 0).astype(np.uint16)
    #arr = np.where(arr[:, :, :] > 20, 1, 0).astype(np.uint8)
    template = ClearVolume(
        volume=arr[:, :, :],
        metadata=vol_tissue.metadata,
        sample_factor=vol_tissue.sample_factor,
    )
    plot_volume_overview(template, 3, is_save=TO_SAVE)

    centerline = get_centerline(template)
    plot_spinal_direction(template, centerline)
    plt.show(block=False)
    print("Centerline with spinal direction assessed\n")

    # 5. Applies the stretching of the coord
    stretch_tissue = stretch_tissue(
        template, centerline, smooth_window=SMOOTH_WINDOW_SIZE
    )
    plot_volume_coronal(stretch_tissue, 10, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(stretch_tissue, 3, is_save=TO_SAVE)
    plt.show(block=True)
    density_before = np.sum(template.volume)
    density_after = np.sum(stretch_tissue.volume)
    percentage_change = (density_after - density_before) / density_before * 100
    print(
        "Tissue has been stretched, with stable density:\n"
        f"  Sum Density before: {density_before}\n"
        f"  Sum Density after:  {density_after}\n"
        f"  Percentage change:  {percentage_change:.2f}%\n"
    )

    cleaned_tissue = clear_external_points(stretch_tissue)
    plot_volume_coronal(cleaned_tissue, 20)
    plt.show()

    # 6. Applies the untwisting of the coord
    registrator = Registrator(
        strategy=RigidRegistration(),
        resampler=RegistratorResampler(),
        config=RegistrationConfig(),
    )

    registrator.config.metric.name = "CC"
    registrator.config.metric.sampling_percentage = 1
    #registrator.config.multiple_res.enabled = True

    untwisted_tissue, twisting_data = untwist_spinal_coord(cleaned_tissue, registrator, window_size=1, gap=0)
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

    downloader = TissueDownloader(source)
    p = downloader.download_volume(vol_tissue, to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_points(tissue, suffix="_scaled_full", to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_metadata(tissue.metadata, to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_volume(stretch_tissue, suffix="_stretch_full", to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_twisting_data(twisting_data, to_update=True)
    print(f"Downloaded at {p}")

    p = downloader.download_volume(
        untwisted_tissue, suffix="_untwisted_full", to_update=False
    )
    print(f"Downloaded at {p}")
