# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pickle

from clearbrain.registration import RegistrationConfig, Registrator, RegistratorResampler, RigidRegistration
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

from pathlib import Path
from typing import cast

from clearbrain.data import TissueLoader, TissueDownloader, TissueSource
from clearbrain.processing import get_centerline, scale_tissue, compress_to_volume, stretch_tissue, untwist_spinal_coord
from clearbrain.tissue import ClearTissue, TissueType
from clearbrain.tissue.view import plot_spinal_direction, plot_volume_coronal, plot_volume_overview



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "74"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD

SCALING: tuple[float, float, float] = (2.22, 1.0, 1.0)
WINDOW_SIZE: int = 25

SMOOTH_WINDOW_SIZE: int = 25

TO_SAVE: bool = False



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Load the tissue
    source = TissueSource(
        mouse = MOUSE,
        tissue_type = TISSUE_TYPE,
        base_path=DATA_FOLDER
    )
    loader = TissueLoader(source)
    tissue = loader.load_points()

    print("================================================================")
    print(f"Working with tissue from file: {str(source.source_filepath)}")
    print("================================================================")
    print(f"Range of X: {np.min(tissue.points[:,0])} - {np.max(tissue.points[:,0])}")
    print(f"Range of Y: {np.min(tissue.points[:,1])} - {np.max(tissue.points[:,1])}")
    print(f"Range of Z: {np.min(tissue.points[:,2])} - {np.max(tissue.points[:,2])}\n")

    # 2. Scale it
    tissue = cast(ClearTissue, scale_tissue(tissue, SCALING))

    print(f"Scaling applied with scaling of: {SCALING}")
    print(f"Range of X: {np.min(tissue.points[:,0])} - {np.max(tissue.points[:,0])}")
    print(f"Range of Y: {np.min(tissue.points[:,1])} - {np.max(tissue.points[:,1])}")
    print(f"Range of Z: {np.min(tissue.points[:,2])} - {np.max(tissue.points[:,2])}\n")

    # 3.Compress into a volume (instead a list of points)
    vol_tissue = compress_to_volume(tissue, WINDOW_SIZE)
    print(f"Volume has now shape: {vol_tissue.volume.shape}, thanks to a window of size: {WINDOW_SIZE}\n")

    plot_volume_coronal(vol_tissue, 10, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(vol_tissue, 3, is_save=TO_SAVE)
    plt.show(block=False)

    # 4. Get the centerline direction
    centerline = get_centerline(vol_tissue)
    plot_spinal_direction(vol_tissue, centerline)
    plt.show(block=False)
    print("Centerline with spinal direction assessed\n")

    stretch_tissue = stretch_tissue(vol_tissue, centerline, smooth_window=SMOOTH_WINDOW_SIZE)
    plot_volume_coronal(stretch_tissue, 10, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(stretch_tissue, 3, is_save=TO_SAVE)
    plt.show(block=False)
    print(
        "Tissue has been stretched, with stable density: "
        f"  Sum Density before: {np.sum(vol_tissue.volume)}\n"
        f"  Sum Desnity after: {np.sum(stretch_tissue.volume)}\n"
    )

    registrator = Registrator(
        strategy=RigidRegistration(),
        resampler=RegistratorResampler(),
        config=RegistrationConfig(),
    )
    untwisted_tissue, twisting_data = untwist_spinal_coord(stretch_tissue, registrator)
    plot_volume_coronal(untwisted_tissue, 10, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(untwisted_tissue, 3, is_save=TO_SAVE)
    plt.show(block=False)
    print(
        "Tissue has been untwisted. the density stability is:\n"
        f"  Sum Density before: {np.sum(stretch_tissue.volume)}\n"
        f"  Sum Desnity after: {np.sum(untwisted_tissue.volume)}\n"
    )

    # 4. Saves the new files
    downloader = TissueDownloader(source)
    p = downloader.download_volume(vol_tissue, to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_points(tissue, suffix="_scaled", to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_metadata(tissue.metadata, to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_volume(stretch_tissue, suffix="_stretch", to_update=True)
    print(f"Downloaded at {p}")
    p = downloader.download_volume(untwisted_tissue, suffix="_untwisted", to_update=True)
    print(f"Downloaded at {p}")

    # saves the stretch
    volume = np.asarray(stretch_tissue.volume)  # your 3D numpy array
    print(np.max(volume))
    volume_to_save = volume.astype(np.uint8)
    print(np.max(volume_to_save))
    affine = np.eye(4)  # identity transform, voxel coordinates only
    nii = nib.Nifti1Image(volume_to_save, affine)
    nib.save(nii, f"{MOUSE}_volume.nii.gz")


    # saves the untwist
    volume = np.asarray(untwisted_tissue.volume)  # your 3D numpy array
    print(np.max(volume))
    volume_to_save = volume.astype(np.uint8)
    print(np.max(volume_to_save))
    affine = np.eye(4)  # identity transform, voxel coordinates only
    nii = nib.Nifti1Image(volume_to_save, affine)
    nib.save(nii, f"{MOUSE}_volume_untwist.nii.gz")

    with open(f"{MOUSE}_twisting_data.pkl", "wb") as f:
        pickle.dump(twisting_data, f, protocol=pickle.HIGHEST_PROTOCOL)
