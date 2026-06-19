# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.widgets import Slider
from scipy.ndimage import rotate

from ...data import ClearVolume, Atlas



# ================================================================
# 1. Section: Functions
# ================================================================
def rotate_spinal_cord(
    tissue: ClearVolume,
    atlas: Atlas | None = None,
    angle: int = -1,
) -> tuple[ClearVolume, int]:
    volume = tissue.data

    if angle == -1:
        if atlas is None:
            raise ValueError("Atlas is required when angle is not provided")

        atlas_volume = atlas.data
        angle, selected_frame = select_angle_interactive(
            volume=volume,
            atlas_volume=atlas_volume,
        )
        print(f"Selected angle = {angle}")

    rotated_volume = rotate_volume_xy(volume, angle)

    return tissue.copy_with(data=rotated_volume), angle


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def select_angle_interactive(
    volume: np.ndarray,
    atlas_volume: np.ndarray,
) -> tuple[int, int]:
    initial_frame = get_biggest_frame_index(volume)
    atlas_frame = get_random_non_empty_frame_index(atlas_volume)

    selected_angle, selected_frame = plot_interactive_rotate_volume(
        volume=volume,
        atlas_volume=atlas_volume,
        initial_frame=initial_frame,
        atlas_frame=atlas_frame,
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

    non_zero_per_frame = np.count_nonzero(volume, axis=(1, 2))

    return int(np.argmax(non_zero_per_frame))


def plot_interactive_rotate_volume(
    volume: np.ndarray,
    atlas_volume: np.ndarray,
    initial_angle: int = 0,
    initial_frame: int = 0,
    atlas_frame: int = 0,
    min_angle: int = -180,
    max_angle: int = 180,
) -> tuple[int, int]:
    selected_angle = initial_angle
    selected_frame = initial_frame

    fig, (atlas_ax, tissue_ax) = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )
    plt.subplots_adjust(bottom=0.30, wspace=0.05)

    atlas_image = atlas_volume[atlas_frame, :, :]
    tissue_image = volume[selected_frame, :, :]

    rotated_tissue_image = rotate_image(tissue_image, selected_angle)

    atlas_ax.imshow(atlas_image, cmap="gray", origin="lower")
    atlas_ax.set_title(f"Atlas reference | Frame = {atlas_frame}")
    atlas_ax.axis("off")

    tissue_im = tissue_ax.imshow(
        rotated_tissue_image,
        cmap="gray",
        origin="lower",
    )
    tissue_ax.set_title(
        f"Tissue frame = {selected_frame} | Rotation angle = {selected_angle}°"
    )
    tissue_ax.axis("off")

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
        label="Tissue frame",
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

        tissue_image = volume[selected_frame, :, :]
        rotated_tissue_image = rotate_image(tissue_image, selected_angle)

        tissue_im.set_data(rotated_tissue_image)
        tissue_ax.set_title(
            f"Tissue frame = {selected_frame} | Rotation angle = {selected_angle}°"
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
        axes=(1, 2),
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )

    return rotated_volume

def get_random_non_empty_frame_index(volume: np.ndarray) -> int:
    """
    Selects a random y-frame with at least one non-zero voxel.

    This matches the tissue preview convention:
        volume[:, y_index, :]
    """

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    non_zero_per_frame = np.count_nonzero(volume, axis=(1, 2))
    valid_frames = np.where(non_zero_per_frame > 0)[0]

    if len(valid_frames) == 0:
        return int(np.random.randint(0, volume.shape[0]))

    return int(np.random.choice(valid_frames))
