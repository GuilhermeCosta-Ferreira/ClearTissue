"""
This script is used to extract the data from the original zarr file.
This get's the low resolution and the scaling factor
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import zarr

import numpy as np

from typing import cast
from pathlib import Path
from numpy.typing import NDArray

from clearbrain.data.Metadata import Metadata
from clearbrain.tissue import TissueType, ClearVolume
from clearbrain.data import TissueDownloader, TissueSource



# ================================================================
# 1. Section: INPUTS
# ================================================================
DRIVE_ROOT: Path = Path("/Volumes/GuiNR")
ZARR_PATH: Path = DRIVE_ROOT / "Transfer/561_CFos_raw.zarr"

MOUSE: str = "32B"
STUDY_DESCRIPTION: str = ""
DATA_FOLDER: Path = Path("data")
TISSUE_TYPE: TissueType = TissueType.SPINAL_COORD



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

    for level in resolution_names:
        level_data = root[level]
        if isinstance(level_data, zarr.Group):
            raise TypeError(f"Level {level} is a group, not an array")

        scale = np.round(np.asarray(level_data.shape) / np.asarray(first_level.shape), 2)
        print(f"For {level} shape is: {level_data.shape} with scale factor of {scale}")

def get_scale_factor_to_high_resolution(root: zarr.Group, level: str) -> NDArray:
    resolution_names = get_resolutions(root)

    first_level = root[resolution_names[0]]

    if isinstance(first_level, zarr.Group):
        raise TypeError("First level is a group, not an array")

    level_data = root[level]
    if isinstance(level_data, zarr.Group):
        raise TypeError(f"Level {level} is a group, not an array")

    scale = np.round(np.asarray(level_data.shape) / np.asarray(first_level.shape), 2)

    return scale



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Load the ZARR file
    root = zarr.open_group(ZARR_PATH, mode="r")
    scale_factor = get_scale_factor_to_high_resolution(root, "level_03")

    print("================================================================")
    print(f"Working with tissue from file: {str(ZARR_PATH.name)}")
    print("================================================================")
    print(f"Levels of Resolution Available: {get_resolutions(root).shape[0]}")
    print_resolution_metadata(root)
    print()

    # 2. Build the metadata and the volume
    metadata = Metadata(
        mouse = MOUSE,
        tissue_type = TISSUE_TYPE,
        file_path = DATA_FOLDER / MOUSE / "tissue_sc",
        scale_factor = tuple(scale_factor),
        description = STUDY_DESCRIPTION
    )

    tissue_raw = ClearVolume(
        volume = np.asarray(cast(np.ndarray, root["level_03"])[:]),
        metadata = metadata,
        sample_factor = 1
    )

    # 3. Download the volume and metadata
    source = TissueSource(mouse=MOUSE, tissue_type=TISSUE_TYPE, base_path=DATA_FOLDER)
    downloader = TissueDownloader(source)
    p = downloader.download_volume(tissue_raw, to_update=True)
    print(f"Raw volume saved to: {str(p)}")
    p = downloader.download_metadata(metadata, to_update=True)
    print(f"Metadata saved to: {str(p)}")
