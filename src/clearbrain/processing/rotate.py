# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.widgets import Slider
from scipy.ndimage import rotate

from ..tissue import ClearVolume
from .noise import get_biggest_slice


# ================================================================
# 1. Section: Functions
# ================================================================
def rotate_spinal_cord(tissue: ClearVolume, angle: int = -1) -> ClearVolume:
    volume = tissue.volume
    biggest_slice = get_biggest_slice(volume)

    if angle == -1:
        angle = select_angle_interactive(biggest_slice)
        print(f"Selected angle = {angle}")

    rotated_volume = rotate_volume_xy(volume, angle)

    return ClearVolume(rotated_volume, tissue.metadata, tissue.sample_factor)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def select_angle_interactive(
    best_slice: np.ndarray,
) -> int:
    selected_angle = plot_interactive_rotate_image(best_slice)

    return selected_angle


def plot_interactive_rotate_image(
    image: np.ndarray,
    initial_angle: int = 0,
    min_angle: int = -180,
    max_angle: int = 180,
) -> int:
    selected_angle = initial_angle

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    rotated_image = rotate(
        image,
        angle=selected_angle,
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )

    im = ax.imshow(rotated_image, cmap="gray", origin="lower")
    ax.set_title(f"Rotation angle = {selected_angle}°")
    ax.axis("off")

    slider_ax = plt.axes((0.2, 0.08, 0.6, 0.04))

    angle_slider = Slider(
        ax=slider_ax,
        label="Angle",
        valmin=min_angle,
        valmax=max_angle,
        valinit=initial_angle,
        valstep=1,
    )

    def update(value):
        nonlocal selected_angle

        selected_angle = int(value)

        rotated_image = rotate(
            image,
            angle=selected_angle,
            reshape=False,
            order=1,
            mode="constant",
            cval=0,
        )

        im.set_data(rotated_image)
        ax.set_title(f"Rotation angle = {selected_angle}°")

        fig.canvas.draw_idle()

    angle_slider.on_changed(update)

    plt.show(block=True)

    return selected_angle


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
