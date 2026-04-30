# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from ..plots import PlotSettings



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_imshow_grid(
    imgs: np.ndarray | list,
    titles: np.ndarray,
    plt_cfg: PlotSettings = PlotSettings()
):
    # 1. Start the grid
    fig, axes = plt.subplots(
        plt_cfg.nr_rows,
        plt_cfg.nr_cols,
        figsize=(3 * plt_cfg.nr_cols, 3 * plt_cfg.nr_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    # 2. Plot every cell of the grid
    for i, img in enumerate(imgs):
        # 2.1 PLot the img as a scatter plot (0,0 at lower left)
        ax = axes[i]
        ax.imshow(img, cmap=plt_cfg.imshow_cmap)

        # 2.2 Plot the center of the data
        if plt_cfg.show_imshow_data_center:
            center = get_img_center(img)
            ax.scatter(center[1], center[0], marker='x', color='white')

        # 2.3 Formating
        ax.set_title(titles[i])
        ax.axis("off")

    # 3. Clear subplots not used
    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    return fig, axes


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_img_center(img: np.ndarray) -> np.ndarray:
    coords = np.argwhere(img)
    center = np.mean(coords, axis=0)

    return center
