# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from typing import Literal
from numpy.typing import NDArray

from ...data import ClearVolume, Atlas

PreferredDirection = Literal["horizontal", "vertical"]



# ================================================================
# 1. Section: Functions
# ================================================================
def build_size_matched_map(
    tissue: ClearVolume,
    atlas: Atlas,
    preferred_direction: PreferredDirection
) -> None:
    tissue_map = get_data_bi_size(tissue)
    atlas_map = get_data_bi_size(atlas)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(tissue_map[:, 0], color="blue", label="tissue")
    plt.plot(atlas_map[:, 0], color="red", label="atlas")
    plt.title("Horizontal Map")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(tissue_map[:, 1], color="blue", label="tissue")
    plt.plot(atlas_map[:, 1], color="red", label="atlas")
    plt.title("Vertical Map")
    plt.legend()
    plt.show(block=False)

    horizontal_shift, horizontal_correlation_map = find_best_shift_by_correlation(tissue_map[:, 0], atlas_map[:, 0])
    vertical_shift, vertical_correlation_map = find_best_shift_by_correlation(tissue_map[:, 1], atlas_map[:, 1])

    print(f"Horizontal shift: {horizontal_shift}")
    print(f"Vertical shift: {vertical_shift}")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(horizontal_correlation_map[:, 0], horizontal_correlation_map[:, 1])
    plt.axvline(horizontal_shift, linestyle="--")
    plt.xlabel("Shift")
    plt.ylabel("Correlation")
    plt.title("Horizontal Correlation")
    plt.subplot(1, 2, 2)
    plt.plot(vertical_correlation_map[:, 0], vertical_correlation_map[:, 1])
    plt.axvline(vertical_shift, linestyle="--")
    plt.xlabel("Shift")
    plt.ylabel("Correlation")
    plt.title("Vertical Correlation")
    plt.show(block=False)

    horizontal_shifted_atlas = apply_shift_for_plot(tissue_map[:, 0], atlas_map[:, 0], horizontal_shift)
    vertical_shifted_atlas = apply_shift_for_plot(tissue_map[:, 1], atlas_map[:, 1], vertical_shift)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(tissue_map[:, 0], color="blue", label="tissue")
    plt.plot(horizontal_shifted_atlas, color="red", label="shifted atlas")
    plt.title("Horizontal Map")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(tissue_map[:, 1], color="blue", label="tissue")
    plt.plot(vertical_shifted_atlas, color="red", label="shifted atlas")
    plt.title("Vertical Map")
    plt.legend()
    plt.show()





# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_data_bi_size(data: ClearVolume | Atlas) -> NDArray:
    nr_slices = data.shape[0]
    data_map = np.zeros((nr_slices, 2))

    for sl in range(nr_slices):
        data_slice = data.data[sl, :, :]
        mask = np.where(data_slice > 0, 1, 0)

        coords = np.argwhere(mask)
        center = np.round(np.mean(coords, axis=0)).astype(int)

        # Count the number of voxels along the axis 0 that go through the center
        horizontal_count = np.sum(mask[center[0], :])
        data_map[sl, 0] = horizontal_count

        # Count the number of voxels along the axis 1 that go through the center
        vertical_count = np.sum(mask[:, center[1]])
        data_map[sl, 1] = vertical_count

    return data_map

def find_best_shift_by_correlation(
    reference: NDArray,
    moving: NDArray,
    min_overlap: int = 500,
) -> tuple[int, NDArray]:
    reference = np.asarray(reference, dtype=float)
    moving = np.asarray(moving, dtype=float)

    if reference.ndim != 1 or moving.ndim != 1:
        raise ValueError("Both inputs must be 1D arrays.")

    min_shift = -(len(reference) - 1)
    max_shift = len(moving) - 1

    shifts = np.arange(min_shift, max_shift + 1)
    correlations = np.full(len(shifts), np.nan)

    for i, shift in enumerate(shifts):
        ref_segment, mov_segment = get_overlap(reference, moving, shift)  # type: ignore

        if len(ref_segment) < min_overlap:
            continue

        correlations[i] = pearson_correlation(ref_segment, mov_segment)

    if np.all(np.isnan(correlations)):
        raise ValueError("No valid correlation found. Try reducing min_overlap.")

    best_index = int(np.nanargmax(correlations))
    best_shift = int(shifts[best_index])

    correlation_map = np.column_stack([shifts, correlations])

    return best_shift, correlation_map

def get_overlap(
    reference: NDArray,
    moving: NDArray,
    shift: int,
) -> tuple[NDArray, NDArray]:
    if shift >= 0:
        ref_start = 0
        mov_start = shift
    else:
        ref_start = -shift
        mov_start = 0

    overlap = min(
        len(reference) - ref_start,
        len(moving) - mov_start,
    )

    if overlap <= 0:
        return reference[:0], moving[:0]

    ref_segment = reference[ref_start:ref_start + overlap]
    mov_segment = moving[mov_start:mov_start + overlap]

    return ref_segment, mov_segment


def pearson_correlation(x: NDArray, y: NDArray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)

    x_std = np.std(x)
    y_std = np.std(y)

    if x_std == 0 or y_std == 0:
        return np.nan

    return float(np.mean((x / x_std) * (y / y_std)))

def apply_shift_for_plot(reference: NDArray, moving: NDArray, shift: int) -> NDArray:
    shifted = np.full_like(reference, np.nan, dtype=float)

    for i in range(len(reference)):
        j = i + shift

        if 0 <= j < len(moving):
            shifted[i] = moving[j]

    return shifted
