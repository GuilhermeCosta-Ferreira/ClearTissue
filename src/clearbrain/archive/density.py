# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scipy.spatial import KDTree



# ================================================================
# 1. Section: Functions
# ================================================================
def get_density(
    points: np.ndarray,
    density_radius: float,
):
    tree = KDTree(points)
    densities = np.array(
        [len(tree.query_ball_point(p, r=density_radius)) - 1 for p in points]
    )

    return densities
