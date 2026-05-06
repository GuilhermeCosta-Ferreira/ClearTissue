# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def scale_points(points: np.ndarray, scale: tuple) -> np.ndarray:
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
        points[:, dim] *= scale[dim]

    # B. Check if output has shape and content valid
    if points.shape != original_shape:
        raise RuntimeError("Scaled points changed shape unexpectedly")
    if not np.isfinite(points).all():
        raise RuntimeError("Scaled points contain non-finite values")

    return points
