# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

import numpy as np

from pathlib import Path

from numpy.typing import NDArray
from cleartissue import ClearTissueProject

from cleartissue.domain_model import TissueType
from cleartissue.adapters.HDF5_to_NII import HDF5_to_Nii
from cleartissue.domain_model.data.ClearVolume import ClearVolume
from cleartissue.domain_model.transformations.utils import get_points_as_volume, scale_points



# ================================================================
# 1. Section: INPUTS
# ================================================================
MOUSE = "32B"
OUT_PATH: Path = Path(f"data/debug/{MOUSE}")


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def estimate_shape(points_data: NDArray) -> tuple[int, int, int]:
    nr_axis = points_data.shape[1]

    estimated_shape = []
    for axis in range(nr_axis):
        estimated_shape.append(np.max(points_data[:,axis]))

    return tuple(estimated_shape)



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    os.makedirs(OUT_PATH, exist_ok=True)

    project = ClearTissueProject.load(
        mouse=MOUSE,
        tissue_type=TissueType.SPINAL_CORD,
    )

    batch = project.load_raw()
    print(batch.tissue.orientation)
    cells = batch.cells
    points = cells.data

    cells_shape = cells.estimate_shape
    tissue_shape = batch.tissue.shape

    print(cells_shape, tissue_shape)

    print(np.asarray(cells_shape) / np.asarray(tissue_shape))


    """

    scaling_factor = np.round(np.asarray(cells.resolution) / np.asarray(batch.tissue.resolution), 5)
    scaled_points = scale_points(points, scaling_factor)

    shape = estimate_shape(scaled_points)
    points_volume = get_points_as_volume(scaled_points, shape)

    print(np.prod(shape)*8/1e9, 'GB')

    cells_volume = ClearVolume(
        data=points_volume,
        resolution=cells.resolution,
        unit=cells.unit,
        orientation=cells.orientation,
        tissue_type=cells.tissue_type
    )
    converter = HDF5_to_Nii(project.source)

    nii_path = OUT_PATH / f"{project.source.cells_base_name}.nii.gz"
    path = converter._convert_volume(cells_volume, OUT_PATH, nii_path)
    print(path)

    """
