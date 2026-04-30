# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from tqdm import tqdm



# ================================================================
# 1. Section: Functions
# ================================================================
def get_symmetry_axis(img):
    center = (img.shape[0] // 2, img.shape[1]// 2)
    length = int(np.sqrt(img.shape[0]**2 + img.shape[1]**2))

    angle_list = np.linspace(0, 180, 90, endpoint=False)

    data = np.where(img > 0, 1, 0)

    separability = []
    for angle in tqdm(angle_list):
        first_mask = side_mask(img.shape, angle, center, eps=0.5)
        second_mask = ~first_mask

        first_data = np.sum(np.where(first_mask, data, 0))
        second_data = np.sum(np.where(second_mask, data, 0))

        symmetry_rate = first_data / second_data
        separability.append(symmetry_rate)

    diffs = np.abs(np.asarray(separability) - 0.5)
    best_symmetry_angle = angle_list[np.argmin(diffs)]

    angle=best_symmetry_angle
    return add_line(img, angle, center, length, value=-10)



# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def add_line(img, angle_deg, center, length, value=1):
    h, w = img.shape
    y0, x0 = center

    theta = np.deg2rad(angle_deg)

    # image coords: x right, y down
    dx = np.cos(theta)
    dy = np.sin(theta)

    t = np.arange(-length, length + 1)

    x = np.round(x0 + t * dx).astype(int)
    y = np.round(y0 + t * dy).astype(int)

    valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)

    line = np.zeros_like(img)
    #line = img
    line[y[valid], x[valid]] = value

    return line

def side_mask(shape, angle_deg, center, eps=0.5):
    h, w = shape
    y0, x0 = center

    theta = np.deg2rad(angle_deg)
    dx = np.cos(theta)
    dy = np.sin(theta)

    y, x = np.mgrid[0:h, 0:w]

    # signed side test
    signed = dx * (y - y0) - dy * (x - x0)

    return signed > eps
