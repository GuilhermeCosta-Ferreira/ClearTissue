# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.widgets import Slider

from ..tissue import ClearVolume


# ================================================================
# 1. Section: Functions
# ================================================================
def select_spinal_cord_limits(
    tissue: ClearVolume,
    start_slice: int = -1,
    end_slice: int = -1,
) -> tuple[int, int]:
    volume = tissue.volume

    if start_slice == -1 or end_slice == -1:
        initial_start, initial_end = get_nonzero_frame_limits(volume)

        if start_slice != -1:
            initial_start = start_slice

        if end_slice != -1:
            initial_end = end_slice

        start_slice, end_slice = plot_interactive_spinal_cord_limits(
            volume=volume,
            initial_start=initial_start,
            initial_end=initial_end,
        )

    validate_spinal_cord_limits(volume, start_slice, end_slice)

    print(f"Selected spinal cord start slice = {start_slice}")
    print(f"Selected spinal cord end slice = {end_slice}")

    return start_slice, end_slice


def crop_spinal_cord(
    tissue: ClearVolume,
    start_slice: int = -1,
    end_slice: int = -1,
) -> tuple[ClearVolume, int, int]:
    volume = tissue.volume

    start_slice, end_slice = select_spinal_cord_limits(
        tissue=tissue,
        start_slice=start_slice,
        end_slice=end_slice,
    )

    cropped_volume = volume[:, start_slice : end_slice + 1, :]

    cropped_tissue = ClearVolume(
        cropped_volume,
        tissue.metadata,
        tissue.sample_factor,
    )

    return cropped_tissue, start_slice, end_slice


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_nonzero_frame_limits(volume: np.ndarray) -> tuple[int, int]:
    """
    Finds the first and last y-frame with non-zero voxels.

    This follows the convention used in rotate_spinal_cord(), where
    preview frames are volume[:, y_index, :].
    """

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    non_zero_per_frame = np.count_nonzero(volume, axis=(0, 2))
    non_zero_indices = np.flatnonzero(non_zero_per_frame)

    if len(non_zero_indices) == 0:
        raise ValueError("Cannot find spinal cord limits. Volume is empty.")

    start_slice = int(non_zero_indices[0])
    end_slice = int(non_zero_indices[-1])

    return start_slice, end_slice


def validate_spinal_cord_limits(
    volume: np.ndarray,
    start_slice: int,
    end_slice: int,
) -> None:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    max_slice = volume.shape[1] - 1

    if not 0 <= start_slice <= max_slice:
        raise ValueError(
            f"start_slice must be between 0 and {max_slice}, got {start_slice}."
        )

    if not 0 <= end_slice <= max_slice:
        raise ValueError(
            f"end_slice must be between 0 and {max_slice}, got {end_slice}."
        )

    if start_slice > end_slice:
        raise ValueError(
            f"start_slice must be <= end_slice, got "
            f"{start_slice} > {end_slice}."
        )


def plot_interactive_spinal_cord_limits(
    volume: np.ndarray,
    initial_start: int = 0,
    initial_end: int | None = None,
) -> tuple[int, int]:
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    max_frame = volume.shape[1] - 1

    if initial_end is None:
        initial_end = max_frame

    selected_start = initial_start
    selected_end = initial_end

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25, wspace=0.05)

    start_image = volume[:, selected_start, :]
    end_image = volume[:, selected_end, :]

    start_im = axes[0].imshow(start_image, cmap="gray", origin="lower")
    end_im = axes[1].imshow(end_image, cmap="gray", origin="lower")

    axes[0].set_title(f"Start slice = {selected_start}")
    axes[1].set_title(f"End slice = {selected_end}")

    for ax in axes:
        ax.axis("off")

    start_slider_ax = plt.axes((0.12, 0.10, 0.32, 0.04))
    end_slider_ax = plt.axes((0.57, 0.10, 0.32, 0.04))

    start_slider = Slider(
        ax=start_slider_ax,
        label="Start",
        valmin=0,
        valmax=max_frame,
        valinit=initial_start,
        valstep=1,
    )

    end_slider = Slider(
        ax=end_slider_ax,
        label="End",
        valmin=0,
        valmax=max_frame,
        valinit=initial_end,
        valstep=1,
    )

    def update(_value):
        nonlocal selected_start
        nonlocal selected_end

        selected_start = int(start_slider.val)
        selected_end = int(end_slider.val)

        start_image = volume[:, selected_start, :]
        end_image = volume[:, selected_end, :]

        start_im.set_data(start_image)
        end_im.set_data(end_image)

        axes[0].set_title(f"Start slice = {selected_start}")
        axes[1].set_title(f"End slice = {selected_end}")

        if selected_start > selected_end:
            fig.suptitle(
                "Invalid selection: start slice is after end slice",
                color="red",
            )
        else:
            fig.suptitle("")

        fig.canvas.draw_idle()

    start_slider.on_changed(update)
    end_slider.on_changed(update)

    plt.show(block=True)

    selected_start = int(start_slider.val)
    selected_end = int(end_slider.val)

    validate_spinal_cord_limits(volume, selected_start, selected_end)

    return selected_start, selected_end
