# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.widgets import Slider
from matplotlib.patches import Circle

from ..tissue import ClearVolume



# ================================================================
# 1. Section: Functions
# ================================================================
def clear_external_points (tissue: ClearVolume, margin: int = -1) -> ClearVolume:
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

    return ClearVolume(cleaned_volume, tissue.metadata, tissue.sample_factor)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_biggest_slice(volume: np.ndarray) -> np.ndarray:
    nr_slices = volume.shape[1]

    biggest_slice = np.zeros_like(volume[:,0,:])
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
    circular_mask = (
        (xx - center_x) ** 2
        + (yy - center_y) ** 2
        <= radius ** 2
    )

    return circular_mask

def select_circle_margin_interactive(
    best_slice: np.ndarray,
    initial_margin: int = 0,
    min_margin: int = 0,
    max_margin: int = 100,
) -> int:
    (cx, cy), data_radius = get_minimum_circle_params(best_slice)
    selected_margin = int(initial_margin)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(best_slice, cmap="gray")
    ax.set_title(
        f"Best slice | Data radius={data_radius:.1f}px"
    )
    ax.axis("off")

    circle = Circle(
        (cx, cy),
        data_radius + selected_margin,
        fill=False,
        linewidth=2,
        color="yellow"
    )
    ax.add_patch(circle)

    slider_ax = plt.axes((0.2, 0.08, 0.6, 0.04))

    margin_slider = Slider(
        ax=slider_ax,
        label="Margin",
        valmin=min_margin,
        valmax=max_margin,
        valinit=initial_margin,
        valstep=1,
    )

    def update(value):
        nonlocal selected_margin

        selected_margin = int(value)
        circle.set_radius(data_radius + selected_margin)

        ax.set_title(
            f"Data radius={data_radius:.1f}px | "
            f"Margin={selected_margin}px | "
            f"Final radius={data_radius + selected_margin:.1f}px"
        )

        fig.canvas.draw_idle()

    margin_slider.on_changed(update)

    plt.show(block=True)

    return selected_margin
