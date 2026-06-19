# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from numpy.typing import NDArray
from scipy.ndimage import zoom



# ================================================================
# 1. Section: Functions
# ================================================================
def resample_to_isotropic(
    volume: NDArray,
    old_resolution_um: tuple[float, float, float] | NDArray,
    target_resolution_um: float,
    is_label: bool = False,
) -> NDArray:
    # 1. Obtain the zoom factors for resampling
    old_resolution_um = np.asarray(old_resolution_um, dtype=float)
    zoom_factors = old_resolution_um / target_resolution_um

    # 2. Makes sure we conserve label consistency
    order = 0 if is_label else 1

    # 3. Perform the resampling
    resampled = zoom(volume, zoom_factors, order=order)
    resampled = np.asarray(resampled, dtype=volume.dtype)

    return resampled
