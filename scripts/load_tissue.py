# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from typing import cast

from clearbrain.data import TissueLoader, TissueDownloader
from clearbrain.processing import scale_tissue, compress_to_volume
from clearbrain.tissue import ClearTissue, TissueType
from clearbrain.tissue.view import plot_volume_coronal, plot_volume_overview



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD

SCALING: tuple[float, float, float] = (2.22, 1.0, 1.0)
WINDOW_SIZE: int = 25

TO_SAVE: bool = False



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Load the tissue
    loader = TissueLoader(
        mouse = MOUSE,
        tissue_type = TISSUE_TYPE,
        base_path=DATA_FOLDER
    )
    tissue = loader.load_points()

    print("================================================================")
    print(f"Working with tissue from file: {str(loader._source_filepath)}")
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

    plot_volume_coronal(vol_tissue, 50, show_centers=True, is_save=TO_SAVE)
    plot_volume_overview(vol_tissue, 3, is_save=TO_SAVE)
    plt.show(block=False)

    downloader = TissueDownloader.from_loader(loader)

    p = downloader.download_volume(vol_tissue)
    print(f"Downloaded at {p}")
    p = downloader.download_points(tissue, suffix="_scaled")
    print(f"Downloaded at {p}")
