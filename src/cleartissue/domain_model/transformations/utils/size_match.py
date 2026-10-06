# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from typing import Literal
from numpy.typing import NDArray
from scipy.signal import find_peaks

from ...data import ClearVolume, Atlas

PreferredDirection = Literal["horizontal", "vertical"]



# ================================================================
# 1. Section: Functions
# ================================================================
def build_size_matched_map(
    tissue: ClearVolume,
    atlas: Atlas,
    preferred_direction: PreferredDirection
) -> NDArray:
    # 1. Get tissue and atlas maps (elongation over horizontal and vertical axis)
    tissue_map = get_data_bi_size(tissue)
    atlas_map = get_data_bi_size(atlas)

    # DEBUG: elongation misalignment for both directions (comment out)
    _debug_plot_elongation(tissue_map, atlas_map)

    # 2. Find best shift by correlation and the candidate peaks
    directional_index = 0 if preferred_direction == "horizontal" else 1
    _, correlation_map = find_best_shift_by_correlation(
        tissue_map[:, directional_index], atlas_map[:, directional_index]
    )
    peak_shifts = find_correlation_peaks(correlation_map)

    # 3. Let the user pick a peak; the one shown on close is used
    shift = select_shift_interactive(
        tissue_map[:, directional_index],
        atlas_map[:, directional_index],
        peak_shifts,
        preferred_direction,
    )
    print(f"{preferred_direction.title()} shift: {shift}")

    # 4. Apply shift to atlas index
    atlas_index = np.arange(0, atlas_map.shape[0])
    shifted_atlas_index = apply_shift(tissue_map[:, directional_index], atlas_index, shift)

    return shifted_atlas_index


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Debug Plots (safe to comment out the calls above)
# ──────────────────────────────────────────────────────
def _debug_plot_elongation(tissue_map: NDArray, atlas_map: NDArray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, name in enumerate(("horizontal", "vertical")):
        axes[idx].plot(tissue_map[:, idx], label="tissue")
        axes[idx].plot(atlas_map[:, idx], label="atlas")
        axes[idx].set_title(f"{name} elongation (unshifted)")
        axes[idx].set_xlabel("slice")
        axes[idx].set_ylabel("elongation")
        axes[idx].legend()
    fig.tight_layout()
    plt.show(block=False)


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
    require_full_coverage: bool = True,
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

        # Tissue is fully contained by the atlas: only score shifts whose
        # overlap spans the whole tissue profile, never a partial window.
        if require_full_coverage:
            if len(ref_segment) < len(reference):
                continue
        elif len(ref_segment) < min_overlap:
            continue

        correlations[i] = pearson_correlation(ref_segment, mov_segment)

    if np.all(np.isnan(correlations)):
        raise ValueError(
            "No valid correlation found. The tissue may be larger than the "
            "atlas; otherwise retry with require_full_coverage=False or a "
            "smaller min_overlap."
        )

    best_index = int(np.nanargmax(correlations))
    best_shift = int(shifts[best_index])

    correlation_map = np.column_stack([shifts, correlations])

    return best_shift, correlation_map

def find_correlation_peaks(correlation_map: NDArray) -> NDArray:
    shifts = correlation_map[:, 0]
    correlations = correlation_map[:, 1]

    series = np.where(np.isfinite(correlations), correlations, -np.inf)
    peak_indices, _ = find_peaks(series)

    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.nanargmax(series))])

    # Strongest correlation first
    peak_indices = peak_indices[np.argsort(series[peak_indices])[::-1]]

    return shifts[peak_indices].astype(int)

def select_shift_interactive(
    tissue_dir: NDArray,
    atlas_dir: NDArray,
    peak_shifts: NDArray,
    preferred_direction: PreferredDirection,
) -> int:
    if len(peak_shifts) == 1:
        return int(peak_shifts[0])

    selected_index = 0

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.25)

    def draw(idx: int) -> None:
        shift = int(peak_shifts[idx])
        shifted_atlas = apply_shift(tissue_dir, atlas_dir, shift)
        ax.clear()
        ax.plot(tissue_dir, label="tissue")
        ax.plot(shifted_atlas, label=f"atlas (shift={shift})")
        ax.set_title(
            f"{preferred_direction} elongation | "
            f"peak {idx + 1}/{len(peak_shifts)} | shift={shift}"
        )
        ax.set_xlabel("slice")
        ax.set_ylabel("elongation")
        ax.legend()

    draw(selected_index)

    slider_ax = plt.axes((0.2, 0.08, 0.6, 0.04))
    peak_slider = Slider(
        ax=slider_ax,
        label="peak",
        valmin=0,
        valmax=len(peak_shifts) - 1,
        valinit=selected_index,
        valstep=1,
    )

    def update(value: float) -> None:
        nonlocal selected_index

        selected_index = int(value)
        draw(selected_index)
        fig.canvas.draw_idle()

    peak_slider.on_changed(update)

    plt.show(block=True)

    return int(peak_shifts[selected_index])

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

def apply_shift(reference: NDArray, moving: NDArray, shift: int) -> NDArray:
    shifted = np.full_like(reference, np.nan, dtype=float)

    for i in range(len(reference)):
        j = i + shift

        if 0 <= j < len(moving):
            shifted[i] = moving[j]

    return shifted
