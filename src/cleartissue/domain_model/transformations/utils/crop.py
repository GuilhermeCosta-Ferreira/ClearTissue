# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

import numpy as np

from ...data import ClearVolume


# ================================================================
# 1. Section: DATA CLASSES
# ================================================================
@dataclass(frozen=True)
class CropParams:
    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    source_shape: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z_min, self.z_max),
            slice(self.y_min, self.y_max),
            slice(self.x_min, self.x_max),
        )

    @property
    def cropped_shape(self) -> tuple[int, int, int]:
        return (
            self.z_max - self.z_min,
            self.y_max - self.y_min,
            self.x_max - self.x_min,
        )


# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def crop_excess(tissue: ClearVolume) -> tuple[ClearVolume, CropParams]:
    volume = tissue.data
    cropped_volume, crop_params = crop_3d_array(volume)

    cropped_tissue = tissue.copy_with(data=cropped_volume)

    return cropped_tissue, crop_params


def apply_crop_excess(
    tissue: ClearVolume,
    crop_params: CropParams,
    strict_shape: bool = True,
) -> ClearVolume:
    cropped_volume = apply_crop_3d_array(
        tissue.data,
        crop_params,
        strict_shape=strict_shape,
    )

    return tissue.copy_with(data=cropped_volume)


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def crop_3d_array(arr: np.ndarray) -> tuple[np.ndarray, CropParams]:
    validate_3d_array(arr)

    coords = np.argwhere(arr != 0)

    # No data at all
    if coords.size == 0:
        crop_params = CropParams(
            z_min=0,
            z_max=0,
            y_min=0,
            y_max=0,
            x_min=0,
            x_max=0,
            source_shape=arr.shape,
        )

        return arr[0:0, 0:0, 0:0], crop_params

    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0) + 1

    crop_params = CropParams(
        z_min=int(min_z),
        z_max=int(max_z),
        y_min=int(min_y),
        y_max=int(max_y),
        x_min=int(min_x),
        x_max=int(max_x),
        source_shape=arr.shape,
    )

    cropped = apply_crop_3d_array(
        arr,
        crop_params,
        strict_shape=True,
    )

    return cropped, crop_params


def apply_crop_3d_array(
    arr: np.ndarray,
    crop_params: CropParams,
    strict_shape: bool = True,
) -> np.ndarray:
    validate_3d_array(arr)

    if strict_shape and arr.shape != crop_params.source_shape:
        raise ValueError(
            f"Array shape {arr.shape} does not match crop source shape "
            f"{crop_params.source_shape}."
        )

    validate_crop_params(arr, crop_params)

    return arr[crop_params.slices]


def validate_3d_array(arr: np.ndarray) -> None:
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr.shape}.")


def validate_crop_params(arr: np.ndarray, crop_params: CropParams) -> None:
    z_size, y_size, x_size = arr.shape

    valid = (
        0 <= crop_params.z_min <= crop_params.z_max <= z_size
        and 0 <= crop_params.y_min <= crop_params.y_max <= y_size
        and 0 <= crop_params.x_min <= crop_params.x_max <= x_size
    )

    if not valid:
        raise ValueError(
            "Invalid crop bounds for array shape "
            f"{arr.shape}: {crop_params}"
        )
