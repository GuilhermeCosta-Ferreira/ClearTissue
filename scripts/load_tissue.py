# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from typing import cast

from clearbrain.data import LoadTissue
from clearbrain.processing import scale_tissue, compress_to_volume
from clearbrain.tissue import ClearTissue
from clearbrain.tissue.view import plot_volume_overview, plot_volume_coronal



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
FILE_TARGET: str = "tissue_sc.json"

SCALING: tuple[float, float, float] = (2.22, 1.0, 1.0)
WINDOW_SIZE: int = 25



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Load the tissue
    filepath = DATA_FOLDER / MOUSE / FILE_TARGET
    tissue = cast(ClearTissue, LoadTissue(filepath).load_tissue())

    print("================================================================")
    print(f"Working with tissue from file: {str(filepath)}")
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

    plot_volume_coronal(vol_tissue, 50, show_centers=True)
    plt.show()
