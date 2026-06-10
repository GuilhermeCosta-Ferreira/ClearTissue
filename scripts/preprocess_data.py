# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from typing import cast

from clearbrain.tissue import ClearTissue, TissueType, ClearVolume
from clearbrain.data import TissueLoader, TissueDownloader, TissueSource

from clearbrain.processing import (
    get_centerline,
    scale_tissue,
    build_volume,
    stretch_tissue,
    untwist_spinal_coord,
    apply_know_untwisting,
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
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD

TO_SAVE: bool = False

SMOOTH_WINDOW_SIZE: int = 25



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def evaluate_loss(initial_volume: ClearVolume, followed_by: ClearVolume) -> None:
    density_before = float(np.sum(initial_volume.volume))
    density_after = float(np.sum(followed_by.volume))
    percentage_change = (density_after - density_before) / density_before * 100
    print(
        "Tissue has been stretched, with stable density:\n"
        f"  Sum Density before: {density_before}\n"
        f"  Sum Density after:  {density_after}\n"
        f"  Percentage change:  {percentage_change:.2f}%\n"
    )



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Loads the main data points
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    loader = TissueLoader(source)
    downloader = TissueDownloader(source)
    cell_detection = loader.load_points(suffix="_cells")
    metadata = loader.load_metadata()
    tissue = loader.load_volume()

    print("================================================================")
    print(f"Working with cell detection from file: {str(source.source_filepath)}")
    print("================================================================")
    print(f"Range of X: {np.min(cell_detection.points[:,0])} - {np.max(cell_detection.points[:,0])}")
    print(f"Range of Y: {np.min(cell_detection.points[:,1])} - {np.max(cell_detection.points[:,1])}")
    print(f"Range of Z: {np.min(cell_detection.points[:,2])} - {np.max(cell_detection.points[:,2])}\n")

    # 2. Scales the tissue points according to the metadata scale factor
    cell_detection = cast(ClearTissue, scale_tissue(cell_detection, metadata.scale_factor))
    print(f"Scaling applied with scaling of: {metadata.scale_factor}")
    print(f"Range of X: {np.min(cell_detection.points[:,0])} - {np.max(cell_detection.points[:,0])}")
    print(f"Range of Y: {np.min(cell_detection.points[:,1])} - {np.max(cell_detection.points[:,1])}")
    print(f"Range of Z: {np.min(cell_detection.points[:,2])} - {np.max(cell_detection.points[:,2])}")
    p = downloader.download_points(cell_detection, suffix="_cells_scaled", to_update=True)
    print(f"Cell detection points saved to: {p}\n")

    # 3. Builds the tissue volume from the scaled points
    cell_detection_arr = build_volume(cell_detection.points, tissue.volume.shape, 1)
    cell_volume = ClearVolume(cell_detection_arr, metadata, 1)
    print(f"Volume has now shape: {cell_volume.volume.shape}")
    p = downloader.download_volume(cell_volume, suffix="_cells_scaled", to_update=True)
    print(f"Cell volume saved to: {p}\n")

    # 4. Gets the centerline and plots the spinal direction
    centerline = get_centerline(tissue)
    plot_spinal_direction(tissue, centerline)
    plt.show(block=False)
    print("Centerline with spinal direction assessed")

    # 5. Stretch the tissue and apply the same transformation to the cell volume
    stretched_tissue = stretch_tissue(tissue, centerline, SMOOTH_WINDOW_SIZE)
    stretched_cell_volume = stretch_tissue(cell_volume, centerline, SMOOTH_WINDOW_SIZE)
    p = downloader.download_volume(stretched_tissue, suffix="_tissue_stretched", to_update=True)
    print(f"Stretched tissue saved to: {p}")
    p = downloader.download_volume(stretched_cell_volume, suffix="_cells_stretched", to_update=True)
    print(f"Stretched cell volume saved to: {p}\n")

    # 6. Plot the stretched tissue and cell volume
    plot_volume_coronal(stretched_tissue, 10, show_centers=True)
    plot_volume_overview(stretched_tissue, 3)
    plot_volume_coronal(stretched_cell_volume, 10, show_centers=True)
    plot_volume_overview(stretched_cell_volume, 3)
    evaluate_loss(tissue, stretched_tissue)
    plt.show(block=False)

    input("Press Enter to close...")
    plt.close('all')

    # 7. Builds the registrator for the untwisting of the coord
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

    # 8. Finds the untwisting of the coord
    untwisted_tissue, twisting_data = untwist_spinal_coord(stretched_tissue, registrator, window_size=250)
    plot_volume_coronal(untwisted_tissue, 10, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(untwisted_tissue, 3, is_save=TO_SAVE)
    evaluate_loss(tissue, untwisted_tissue)
    p = downloader.download_volume(untwisted_tissue, suffix="_tissue_untwisted", to_update=True)
    print(f"Downloaded untwisted tissue at {p}")
    p = downloader.download_twisting_data(twisting_data, suffix="_twisting_data", to_update=True)
    print(f"Downloaded twisting data at {p}")
    plt.show(block=False)

    # 9. Applies the known untwisting
    registrator.config.interpolator.resampling = "nearest"
    untwisted_cell = apply_know_untwisting(stretched_cell_volume, registrator, twisting_data)
    plot_volume_coronal(untwisted_cell, 10, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(untwisted_cell, 3, is_save=TO_SAVE)
    evaluate_loss(stretched_cell_volume, untwisted_cell)
    p = downloader.download_volume(untwisted_cell, suffix="_cells_untwisted", to_update=True)
    print(f"Downloaded untwisted cells at {p}")
    plt.show(block=False)

    input("Press Enter to close...")
    plt.close('all')
