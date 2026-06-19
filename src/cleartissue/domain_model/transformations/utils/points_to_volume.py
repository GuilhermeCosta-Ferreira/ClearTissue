# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from numpy.typing import NDArray


# ================================================================
# 1. Section: Functions
# ================================================================
def get_points_as_volume(
    points: NDArray,
    volume_shape: tuple[int, int, int] | NDArray,
) -> NDArray:
    # 1. Colapses the points into voxel coordinates
    points_int = points.astype(int)
    volume_shape = np.asarray(volume_shape, dtype=int)

    # 2. Selects only the points inside the volume
    valid_mask = np.all(
        (points_int >= 0) & (points_int < volume_shape),
        axis=1,
    )
    valid_points = points_int[valid_mask]

    # 3. Builds the volume
    volume = np.zeros(volume_shape, dtype=int)
    for p in valid_points:
        volume[p[0], p[1], p[2]] += 1

    return volume
