# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ..tissue import ClearTissue, ClearVolume

MAX_SIZE = 400**3 # roughly the size we had on the CT or MRI


# ================================================================
# 1. Section: Functions
# ================================================================
def compress_to_volume(tissue: ClearTissue, window_size: int) -> ClearVolume:
    points = tissue.points
    data_range = points_range(points)

    volume_shape = get_volume_shape_from_window(data_range, window_size)

    if np.prod(volume_shape) > MAX_SIZE:
        raise OverflowError("The window needs to be bigger, we should not exceed "
            f"{MAX_SIZE} and with window size of {window_size} we get {np.prod(volume_shape)}")

    volume = build_volume(points, volume_shape, window_size)

    return ClearVolume(volume, tissue.metadata, window_size)



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def points_range(points: np.ndarray) -> np.ndarray:
    return np.asarray([np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])], dtype=int)

def get_volume_shape_from_window(data_range: np.ndarray, window_size: int) -> np.ndarray:
    # 1. Computes the amount of windows needed
    full_div = data_range // window_size
    remain = data_range % window_size

    # 2. Return the shape needed for those
    dim = data_range.shape[0]
    shape = np.zeros(dim, dtype=int)
    for i in range(dim):
        shape[i] = full_div[i] + (1 if remain[i] != 0 else 0)

    return shape

def build_volume(points: np.ndarray, volume_shape: np.ndarray, window_size: int) -> np.ndarray:
    downsampled_points = (points // window_size).astype(int)

    volume = np.zeros(volume_shape, dtype=int)
    for p in downsampled_points:
        volume[p[0], p[1], p[2]] += 1
    return volume
