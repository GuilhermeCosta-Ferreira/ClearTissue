# ================================================================
# 0. Section: IMPORTS
# ================================================================
from matplotlib import pyplot as plt

from .pca import get_image_pca_components
from .symmetry_2 import get_symmetry_axis


# ================================================================
# 1. Section: Functions
# ================================================================
def plot_section_2d(prism_imgs):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()
    colors =['white', 'yellow', '#4E4EB1']

    for idx, img in enumerate(prism_imgs):
        ax = axes[idx]

        pcs = get_image_pca_components(img)
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        img = get_symmetry_axis(img)
        ax.imshow(img, origin='lower', cmap='hot', aspect='equal')

        for i in range(2):
            ax.arrow(center_x, center_y,
                        pcs[0, i] * 20,
                        pcs[1, i] * 20,
                        head_width=5, head_length=5, fc=colors[i], ec=colors[i])

        ax.set_xlabel("Local U")
        ax.set_ylabel("Local V")

    plt.tight_layout()

    return fig, axes
