# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ..tissue import ClearVolume



# ================================================================
# 1. Section: Functions
# ================================================================
def crop_excess(tissue: ClearVolume) -> ClearVolume:
    # 1. Load the data and crop it
    volume = tissue.volume
    cropped_volume = crop_3d_array(volume)

    return ClearVolume(cropped_volume, tissue.metadata, tissue.sample_factor)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def crop_3d_array(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != 0)

    # No data at all
    if coords.size == 0:
        return arr[0:0, 0:0, 0:0]

    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0) + 1

    cropped = arr[min_z:max_z, min_y:max_y, min_x:max_x]

    return cropped
