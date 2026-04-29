# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from ..tissue import ClearVolume, TissueType
from ..tissue import SpinalCenterline



# ================================================================
# 1. Section: Functions
# ================================================================
def get_centerline(tissue_volume: ClearVolume, momentum: float = 0.25) -> SpinalCenterline:
    if tissue_volume.metadata.tissue_type != TissueType.SPINAL_COORD:
        raise TypeError("This function is only for Spinal Coord, "
            f"not {tissue_volume.metadata.tissue_type}")

    volume = np.where(tissue_volume.volume > 0, 1, 0)

    centerline_volume = np.zeros_like(volume)
    nr_slices = volume.shape[1]
    centerline_centers = np.zeros((nr_slices, 2))

    for sl in range(nr_slices):
        coronal = volume[:, sl, :]
        mean_center = get_img_center(coronal)

        if sl == 0 or np.isnan(centerline_centers[sl-1]).any():
            center = np.round(mean_center)
        else:
            center = np.round(mean_center * (1 - momentum) + centerline_centers[sl-1] * momentum)

        if not np.isnan(center).any():
            center = center.astype(int)
            centerline_volume[center[0], sl, center[1]] = 1

        centerline_centers[sl] = np.array([center[0], center[1]])

    return SpinalCenterline(centerline_volume, centerline_centers, tissue_volume.metadata)



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_img_center(img: np.ndarray) -> np.ndarray:
    if not np.sum(img) == 0:
        coords = np.argwhere(img)
        center = np.mean(coords, axis=0)
    else:
        center = np.array([np.nan, np.nan])

    return center
