# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from ..tissue import ClearVolume
from ..plots.interactive import plot_interactive_circle_on_image


# ================================================================
# 1. Section: Functions
# ================================================================
def clear_external_points(tissue: ClearVolume, margin: int = -1) -> tuple[ClearVolume, int]:
    # 1. Load the data
    volume = tissue.volume
    biggest_slice = get_biggest_slice(volume)

    # 2. Use interactive masking if not specified
    if margin == -1:
        margin = select_circle_margin_interactive(biggest_slice)
        print(f"Selected margin = {margin}")

    # 3. Build the mask
    circular_mask = get_circlular_mask(biggest_slice, margin)

    # 4. Apply the mask
    cleaned_volume = np.where(
        circular_mask[:, None, :],
        volume,
        0,
    )

    return ClearVolume(cleaned_volume, tissue.metadata, tissue.sample_factor), margin


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_biggest_slice(volume: np.ndarray) -> np.ndarray:
    nr_slices = volume.shape[1]

    biggest_slice = np.zeros_like(volume[:, 0, :])
    for sl in range(nr_slices):
        coronal = volume[:, sl, :]

        if np.sum(coronal) > np.sum(biggest_slice):
            biggest_slice = coronal.copy()

    return biggest_slice


def get_minimum_circle_params(coronal: np.ndarray) -> tuple[tuple, float]:
    best_mask = np.where(coronal > 0, 1, 0)

    ys, xs = np.where(best_mask)

    cy = float(np.median(ys))
    cx = float(np.median(xs))

    distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    data_radius = float(np.max(distances))

    return (cx, cy), data_radius


def get_circlular_mask(biggest_slice: np.ndarray, margin: int) -> np.ndarray:
    # 1. init the shapes
    height, width = biggest_slice.shape
    yy, xx = np.ogrid[:height, :width]

    # 2. Get the parameters
    (center_x, center_y), radius = get_minimum_circle_params(biggest_slice)
    radius += margin

    # 3. Build the mask
    circular_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2

    return circular_mask


def select_circle_margin_interactive(
    best_slice: np.ndarray,
) -> int:
    center, data_radius = get_minimum_circle_params(best_slice)

    selected_margin = plot_interactive_circle_on_image(best_slice, data_radius, center)

    return selected_margin
