# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.widgets import Slider
from matplotlib.patches import Circle


# ================================================================
# 1. Section: Functions
# ================================================================
def plot_interactive_circle_on_image(
    image: np.ndarray,
    min_radius: float,
    center: tuple,
    initial_margin: int = 0,
    min_margin: int = 0,
) -> int:
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(image, cmap="gray")
    ax.set_title(f"Data radius={min_radius:.1f}px")
    ax.axis("off")

    circle = Circle(
        center, min_radius + initial_margin, fill=False, linewidth=2, color="yellow"
    )
    ax.add_patch(circle)

    slider_ax = plt.axes((0.2, 0.08, 0.6, 0.04))

    max_margin = np.max(image.shape)
    margin_slider = Slider(
        ax=slider_ax,
        label="Margin",
        valmin=min_margin,
        valmax=max_margin,
        valinit=initial_margin,
        valstep=1,
    )

    def update(value):
        nonlocal initial_margin

        initial_margin = int(value)
        circle.set_radius(min_radius + initial_margin)

        ax.set_title(
            f"Data radius={min_radius:.1f}px | "
            f"Margin={initial_margin}px | "
            f"Final radius={min_radius + initial_margin:.1f}px"
        )

        fig.canvas.draw_idle()

    margin_slider.on_changed(update)

    plt.show(block=True)
    return initial_margin
