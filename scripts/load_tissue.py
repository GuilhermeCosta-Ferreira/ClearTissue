# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path
from typing import cast

from clearbrain.data import LoadTissue
from clearbrain.processing import scale_tissue
from clearbrain.tissue import ClearTissue



# ================================================================
# 1. Section: INPUTS
# ================================================================
DATA_FOLDER: Path = Path("data")
MOUSE: str = "32B"
FILE_TARGET: str = "tissue_sc.json"

SCALING: tuple[float, float, float] = (2.22, 1.0, 1.0)



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
    print(f"Range of Z: {np.min(tissue.points[:,2])} - {np.max(tissue.points[:,2])}")

    # 2. Scale it
    scaled_tissue = cast(ClearTissue, scale_tissue(tissue, SCALING))
    print()
    print(f"Scaling applied with scaling of: {SCALING}")
    print(f"Range of X: {np.min(scaled_tissue.points[:,0])} - {np.max(scaled_tissue.points[:,0])}")
    print(f"Range of Y: {np.min(scaled_tissue.points[:,1])} - {np.max(scaled_tissue.points[:,1])}")
    print(f"Range of Z: {np.min(scaled_tissue.points[:,2])} - {np.max(scaled_tissue.points[:,2])}")

    # 3.Compress into a volume (instead a list of points)
    #compress_to_volume(tissue, )
