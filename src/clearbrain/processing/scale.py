# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from functools import singledispatch

from ..tissue import ClearTissue


# ================================================================
# 1. Section: Functions
# ================================================================
@singledispatch
def scale_tissue(obj, scale: tuple[float, float, float]) -> np.ndarray | ClearTissue:
    raise TypeError(f"Unsupported type: {type(obj).__name__}")


@scale_tissue.register
def _(points: np.ndarray, scale: tuple[float, float, float]) -> np.ndarray:
    # A. Makes sure the input has the correct shape
    if points.shape[1] != len(scale):
        raise ValueError(
            "Points and Scale should have the same dim"
            f"Points ({points.shape} and Scale ({len(scale)}))"
        )

    # 1. Applies the scalling for each dimension
    original_shape = points.shape
    dims = original_shape[1]
    for dim in range(dims):
        points[:, dim] = np.round(points[:, dim] * scale[dim]).astype(int)

    # B. Check if output has shape and content valid
    if points.shape != original_shape:
        raise RuntimeError("Scaled points changed shape unexpectedly")
    if not np.isfinite(points).all():
        raise RuntimeError("Scaled points contain non-finite values")

    return points


@scale_tissue.register
def _(tissue: ClearTissue, scale: tuple[float, float, float]) -> ClearTissue:
    # 1. Applies the scalling on the tissue points
    points = tissue.points
    scaled_points = scale_tissue(points, scale)

    # 1.A Makes sure we are working with points as array
    if isinstance(scaled_points, ClearTissue):
        raise TypeError(
            "Some unexpected error, points hsould be returned as an "
            "array, not as an ClearTissue"
        )

    # 2. Creates a tissue copy
    scaled_tissue = tissue.copy()
    scaled_tissue.points = scaled_points

    return scaled_tissue
