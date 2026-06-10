# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.widgets import Slider
from scipy.ndimage import rotate

from ..tissue import ClearVolume


# ================================================================
# 1. Section: Functions
# ================================================================
def rotate_spinal_cord(
    tissue: ClearVolume,
    angle: int = -1,
) -> tuple[ClearVolume, int]:
    volume = tissue.volume

    if angle == -1:
        angle, selected_frame = select_angle_interactive(volume)
        print(f"Selected angle = {angle}")
        print(f"Selected frame = {selected_frame}")

    rotated_volume = rotate_volume_xy(volume, angle)

    return ClearVolume(rotated_volume, tissue.metadata, tissue.sample_factor), angle


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def select_angle_interactive(
    volume: np.ndarray,
) -> tuple[int, int]:
    initial_frame = get_biggest_frame_index(volume)

    selected_angle, selected_frame = plot_interactive_rotate_volume(
        volume=volume,
        initial_frame=initial_frame,
    )

    return selected_angle, selected_frame


def get_biggest_frame_index(volume: np.ndarray) -> int:
    """
    Finds the y-frame with the most non-zero voxels.

    This matches rotate_volume_xy(), which rotates over axes=(0, 2).
    Therefore, the preview frame is volume[:, y_index, :].
    """

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    non_zero_per_frame = np.count_nonzero(volume, axis=(0, 2))

    return int(np.argmax(non_zero_per_frame))


def plot_interactive_rotate_volume(
    volume: np.ndarray,
    initial_angle: int = 0,
    initial_frame: int = 0,
    min_angle: int = -180,
    max_angle: int = 180,
) -> tuple[int, int]:
    selected_angle = initial_angle
    selected_frame = initial_frame

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.30)

    image = volume[:, selected_frame, :]

    rotated_image = rotate_image(image, selected_angle)

    im = ax.imshow(rotated_image, cmap="gray", origin="lower")
    ax.set_title(
        f"Frame = {selected_frame} | Rotation angle = {selected_angle}°"
    )
    ax.axis("off")

    angle_slider_ax = plt.axes((0.2, 0.13, 0.6, 0.04))
    frame_slider_ax = plt.axes((0.2, 0.06, 0.6, 0.04))

    angle_slider = Slider(
        ax=angle_slider_ax,
        label="Angle",
        valmin=min_angle,
        valmax=max_angle,
        valinit=initial_angle,
        valstep=1,
    )

    frame_slider = Slider(
        ax=frame_slider_ax,
        label="Frame",
        valmin=0,
        valmax=volume.shape[1] - 1,
        valinit=initial_frame,
        valstep=1,
    )

    def update(_value):
        nonlocal selected_angle
        nonlocal selected_frame

        selected_angle = int(angle_slider.val)
        selected_frame = int(frame_slider.val)

        image = volume[:, selected_frame, :]
        rotated_image = rotate_image(image, selected_angle)

        im.set_data(rotated_image)
        ax.set_title(
            f"Frame = {selected_frame} | Rotation angle = {selected_angle}°"
        )

        fig.canvas.draw_idle()

    angle_slider.on_changed(update)
    frame_slider.on_changed(update)

    plt.show(block=True)

    return selected_angle, selected_frame


def rotate_image(
    image: np.ndarray,
    angle: int,
) -> np.ndarray:
    rotated_image = rotate(
        image,
        angle=angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )

    return rotated_image


def rotate_volume_xy(
    volume: np.ndarray,
    angle: int,
) -> np.ndarray:
    rotated_volume = rotate(
        volume,
        angle=angle,
        axes=(0, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )

    return rotated_volume
