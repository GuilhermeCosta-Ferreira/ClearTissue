# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scipy.interpolate import splev, splprep



# ================================================================
# 1. Section: Functions
# ================================================================
def get_centerline(
    points: np.ndarray,
    bin_width: float,
) -> np.ndarray:

    y_coords = points[:, 1]
    min_y = y_coords.min()
    max_y = y_coords.max()

    bin_centers = np.arange(min_y + 100, max_y - 50, bin_width)

    centerline = []
    for yc in bin_centers:
        mask = (y_coords >= yc - bin_width / 2) & (y_coords < yc + bin_width / 2)
        if np.sum(mask) >= 10:
            pts = points[mask]
            centerline.append([pts[:, 0].mean(), yc, pts[:, 2].mean()])

    return np.array(centerline)

def smooth_centerline(
    centerline: np.ndarray,
    spline_smoothing: float,
    n_points_on_line: int,
) -> np.ndarray:

    # A. Ignores if it is small
    if len(centerline) <= 5:
        return centerline

    tck, _ = splprep(centerline.T, s=spline_smoothing, k=3)
    u_new = np.linspace(0, 1, n_points_on_line)
    smooth_x, smooth_y, smooth_z = splev(u_new, tck)

    return np.column_stack((smooth_x, smooth_y, smooth_z))
