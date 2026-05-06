# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.axes import Axes



# ================================================================
# 1. Section: Functions
# ================================================================
def plot_spinal_sections(
    spinal_sections: np.ndarray,
) -> tuple:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    all_x = []
    all_y = []
    all_z = []

    for prism in spinal_sections:
        ax = build_prims(ax, prism)
        all_x.extend(prism[:, 0])
        all_y.extend(prism[:, 1])
        all_z.extend(prism[:, 2])

    x = np.array(all_x)
    y = np.array(all_y)
    z = np.array(all_z)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    return fig, ax

def add_spinal_sections(
    ax: Axes,
    spinal_sections: np.ndarray,
) -> Axes:
    for prism in spinal_sections:
        ax = build_prims(ax, prism)

    return ax


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def build_prims(ax, prism):
    faces = [
        [prism[0], prism[1], prism[3], prism[2]],
        [prism[4], prism[5], prism[7], prism[6]],
        [prism[0], prism[1], prism[5], prism[4]],
        [prism[1], prism[3], prism[7], prism[5]],
        [prism[3], prism[2], prism[6], prism[7]],
        [prism[2], prism[0], prism[4], prism[6]],
    ]

    poly = Poly3DCollection(
        faces,
        alpha=0.2,
        edgecolor="black",
        linewidths=0.6,
    )
    ax.add_collection3d(poly)

    return ax
