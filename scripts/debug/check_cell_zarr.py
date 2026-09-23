# ================================================================
# 0. Section: IMPORTS
# ================================================================
import zarr

import numpy as np

from pathlib import Path
from numpy.typing import NDArray

from cleartissue import ClearTissueProject
from cleartissue.domain_model import TissueType



# ================================================================
# 1. Section: INPUTS
# ================================================================
DRIVE_ROOT: Path = Path("/Volumes/GuiNR")
#ZARR_PATH: Path = DRIVE_ROOT / "Transfer/198B/561_CFos_raw.zarr"
ZARR_PATH: Path = DRIVE_ROOT / "Transfer/198B/561_CFos_cells.zarr"
#ZARR_PATH: Path = DRIVE_ROOT / "Transfer/01GT/488_Virus_raw.zarr"

MOUSE: str = "198B-Cells"
PIPELINE_ID: int = -1
STEP: int = 0



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def get_resolutions(root: zarr.Group) -> NDArray:
    resolutions = []
    for level in root.array_keys():
        resolutions.append(level)

    return np.sort(np.asarray(resolutions))

def print_resolution_metadata(root: zarr.Group) -> None:
    resolution_names = get_resolutions(root)

    first_level = root[resolution_names[0]]

    if isinstance(first_level, zarr.Group):
        raise TypeError("First level is a group, not an array")

    for idx, level in enumerate(resolution_names):
        level_data = root[level]
        if isinstance(level_data, zarr.Group):
            raise TypeError(f"Level {level} is a group, not an array")

        scale = np.round(np.asarray(level_data.shape) / np.asarray(first_level.shape), 5)
        res = root.metadata.attributes["multiscales"][0]["datasets"][idx]['coordinateTransformations'][0]["scale"]
        print(f"For {level} shape is: {level_data.shape}, resolution is {res}, with scale factor of {scale}")



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    project = ClearTissueProject.load(
        mouse=MOUSE,
        tissue_type=TissueType.SPINAL_CORD,
    )

    root = zarr.open_group(ZARR_PATH, mode="r")

    print_resolution_metadata(root)
