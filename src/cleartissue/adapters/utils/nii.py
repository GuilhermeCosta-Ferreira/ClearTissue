# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
import nibabel as nib

from pathlib import Path

from ...domain_model.data import ClearData


# ================================================================
# 1. Section: Functions
# ================================================================
def affine_from_attrs(data: ClearData) -> np.ndarray:
    resolution = data.resolution
    unit = data.unit
    orientation = data.orientation

    return make_affine(
        resolution=resolution,
        unit=unit,
        orientation=orientation,
    )

def save_nii_gz(
    data: np.ndarray,
    affine: np.ndarray,
    out_path: Path,
) -> None:

    if data.ndim != 3:
        raise ValueError(
            f"NIfTI export only supports 3D arrays. Got shape {data.shape}."
        )

    data = to_nifti_safe_dtype(data)

    image = nib.Nifti1Image(data, affine)
    image.header.set_data_dtype(data.dtype)

    nib.save(image, out_path)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def make_affine(
    resolution: tuple[float, float, float],
    unit: tuple[str, str, str],
    orientation: str,
) -> np.ndarray:
    resolution_mm = tuple(
        to_mm(value, unit_i)
        for value, unit_i in zip(resolution, unit)
    )

    axis_map = {
        "r": (0, +1),
        "l": (0, -1),
        "a": (1, +1),
        "p": (1, -1),
        "s": (2, +1),
        "i": (2, -1),
    }

    affine = np.eye(4)
    affine[:3, :3] = 0.0

    for data_axis, axis_code in enumerate(orientation.lower()):
        if axis_code not in axis_map:
            raise ValueError(
                f"Invalid orientation character '{axis_code}' "
                f"in orientation '{orientation}'."
            )

        world_axis, sign = axis_map[axis_code]
        affine[world_axis, data_axis] = sign * resolution_mm[data_axis]

    return affine

def to_mm(value: float, unit: str) -> float:
    unit = unit.lower()

    if unit in {"mm", "millimeter", "millimetre", "millimeters", "millimetres"}:
        return value

    if unit in {"um", "µm", "micrometer", "micrometre", "micrometers", "micrometres"}:
        return value / 1000.0

    raise ValueError(f"Unsupported spatial unit for NIfTI export: {unit}")

def to_nifti_safe_dtype(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.bool_:
        return data.astype(np.uint8)

    if np.issubdtype(data.dtype, np.floating):
        if data.dtype == np.float64:
            return data.astype(np.float32)
        return data

    if np.issubdtype(data.dtype, np.integer):
        min_value = int(data.min())
        max_value = int(data.max())

        if min_value >= 0:
            if max_value <= np.iinfo(np.uint8).max:
                return data.astype(np.uint8)

            if max_value <= np.iinfo(np.uint16).max:
                return data.astype(np.uint16)

            if max_value <= np.iinfo(np.uint32).max:
                return data.astype(np.uint32)

        else:
            if min_value >= np.iinfo(np.int16).min and max_value <= np.iinfo(np.int16).max:
                return data.astype(np.int16)

            if min_value >= np.iinfo(np.int32).min and max_value <= np.iinfo(np.int32).max:
                return data.astype(np.int32)

        raise ValueError(
            f"Cannot safely export integer data with range "
            f"[{min_value}, {max_value}] to NIfTI."
        )

    raise TypeError(f"Unsupported dtype for NIfTI export: {data.dtype}")
