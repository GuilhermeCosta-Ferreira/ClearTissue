# ================================================================
# 0. Section: IMPORTS
# ================================================================
import zarr

import numpy as np

from typing import cast
from pathlib import Path
from numpy.typing import NDArray

from clearbrain.domain_model.data import TissueType, ClearVolume
from clearbrain.adapters.DataDownloader import DataDownloader
from clearbrain.adapters.Source import Source



# ================================================================
# 1. Section: INPUTS
# ================================================================
DRIVE_ROOT: Path = Path("/Volumes/GuiNR")
ZARR_PATH: Path = DRIVE_ROOT / "Transfer/561_CFos_raw.zarr"

MOUSE: str = "32B"
STUDY_DESCRIPTION: str = ""
DATA_FOLDER: Path = Path("data")
TISSUE_TYPE: TissueType = TissueType.SPINAL_CORD

TARGET_RESOLUTION: str = "level_03"
TARGET_RESOLUTION_POSITION: int = -1
UNIT: str = "um"
ORIENTATION: str = "sal"



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

        scale = np.round(np.asarray(level_data.shape) / np.asarray(first_level.shape), 5)
        print(f"For {level} shape is: {level_data.shape} with scale factor of {scale}")

def get_scale_factor_to_high_resolution(root: zarr.Group, level: str) -> NDArray:
    resolution_names = get_resolutions(root)

    first_level = root[resolution_names[0]]

    if isinstance(first_level, zarr.Group):
        raise TypeError("First level is a group, not an array")

    level_data = root[level]
    if isinstance(level_data, zarr.Group):
        raise TypeError(f"Level {level} is a group, not an array")

    scale = np.round(np.asarray(level_data.shape) / np.asarray(first_level.shape), 5)

    return scale



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Load the ZARR file
    root = zarr.open_group(ZARR_PATH, mode="r")

    resolution = root.metadata.attributes["multiscales"][0]["datasets"][TARGET_RESOLUTION_POSITION]['coordinateTransformations'][0]["scale"]

    arr = cast(np.ndarray, root["level_03"])
    tissue_volume = np.where(arr[:, :, :] > 20, arr[:, :, :], 0).astype(np.uint16)
    tissue_raw = ClearVolume(
        data = tissue_volume,
        resolution = resolution,
        tissue_type = TISSUE_TYPE,
        unit = tuple([UNIT]*3), # type: ignore
        orientation = ORIENTATION
    )

    source = Source(
        mouse = MOUSE,
        tissue_type = TISSUE_TYPE,
        base_path=Path("data"),
        atlas_name = "allen_cord_20um"
    )

    downloader = DataDownloader(source)
    source.step_path(-1, 0).mkdir(parents=True, exist_ok=True)
    downloader._download_tissue(
        tissue = tissue_raw,
        pipeline_id = -1,
        step = 0
    )
