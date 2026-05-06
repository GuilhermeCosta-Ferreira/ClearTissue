# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ..density import get_density



# ================================================================
# 1. Section: Functions
# ================================================================
def filter_low_density_points(
    points: np.ndarray,
    density_radius: float,
    min_density: int,
) -> np.ndarray:
    densities = get_density(points, density_radius)
    return points[densities > min_density]
