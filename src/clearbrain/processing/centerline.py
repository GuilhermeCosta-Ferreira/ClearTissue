# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ..tissue import ClearVolume, TissueType
from ..tissue import SpinalCenterline


# ================================================================
# 1. Section: Functions
# ================================================================
def get_centerline(
    tissue_volume: ClearVolume, momentum: float = 0.25
) -> SpinalCenterline:
    # A. Makes sure we only apply this to a spinal coord
    if tissue_volume.metadata.tissue_type != TissueType.SPINAL_COORD:
        raise TypeError(
            "This function is only for Spinal Coord, "
            f"not {tissue_volume.metadata.tissue_type}"
        )

    # 1. We don't care about the densities, only the mask
    volume = np.where(tissue_volume.volume > 0, 1, 0)

    # 2. Prepare the data to be filled
    centerline_volume = np.zeros_like(volume)
    nr_slices = volume.shape[1]
    centerline_centers = np.zeros((nr_slices, 2))

    # 3. Iterate over each slice
    for sl in range(nr_slices):
        # 3.1 Extract the naive center
        coronal = volume[:, sl, :]
        mean_center = get_img_center(coronal)

        # 3.2 Apply only when there is previous data the moment shift (make more smooth)
        if sl == 0 or np.isnan(centerline_centers[sl - 1]).any():
            center = np.round(mean_center)
        else:
            center = np.round(
                mean_center * (1 - momentum) + centerline_centers[sl - 1] * momentum
            )

        # 3.3 Save the data, avoiding storing nan on the volume
        if not np.isnan(center).any():
            center = center.astype(int)
            centerline_volume[center[0], sl, center[1]] = 1
        centerline_centers[sl] = np.array([center[0], center[1]])

    # 4. Adds to 3D the centerline
    centerline_centers = np.column_stack(
        (
            centerline_centers[:, 0],
            np.arange(nr_slices),
            centerline_centers[:, 1],
        )
    )

    # 5. Extend the coord where is nan
    centerline_centers = fill_nan_coords(centerline_centers)

    return SpinalCenterline(
        centerline_volume, centerline_centers, tissue_volume.metadata
    )


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


def fill_nan_coords(centerline_centers: np.ndarray) -> np.ndarray:
    filled = centerline_centers.copy()

    slice_ids = filled[:, 1]

    for col in (0, 2):  # x and z columns
        values = filled[:, col]
        valid = ~np.isnan(values)

        if valid.sum() == 0:
            raise ValueError(
                f"Column {col} is fully NaN; cannot interpolate centerline."
            )

        filled[:, col] = np.round(
            np.interp(
                slice_ids,
                slice_ids[valid],
                values[valid],
            )
        )

    return filled
