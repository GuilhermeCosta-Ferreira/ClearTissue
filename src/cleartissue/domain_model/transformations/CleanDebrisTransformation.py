# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt

from scipy.ndimage import label
from dataclasses import dataclass
from matplotlib.widgets import Slider

from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class CleanDebrisTransformation(AbstractTransformation):
    def apply(self, batch: SampleBatch) -> SampleBatch:
        mask = build_mask(batch.tissue)

        # DEBUG: scroll through the connected components, each a color (comment out)
        _debug_plot_components(mask.data, axis=0)

        # 1. Get only the biggest connected component
        largest_component = get_largest_connected_component(mask)

        # 2. Restore original intensities inside the kept component
        cleaned_tissue = restore_original_intensities(batch.tissue, largest_component)

        return batch.copy_with(tissue=cleaned_tissue)


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Debug Plots
# ──────────────────────────────────────────────────────
def _debug_plot_components(mask: np.ndarray, axis: int = 0) -> None:
    connectivity = np.ones((3, 3, 3), dtype=int)
    labels, nr_components = label(mask, structure=connectivity)

    # Random colors per label, with background (0) forced to black
    rng = np.random.default_rng(0)
    colors = rng.random((nr_components + 1, 3))
    colors[0] = 0.0

    nr_slices = labels.shape[axis]

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)

    initial = nr_slices // 2
    image = ax.imshow(colors[np.take(labels, initial, axis=axis)], origin="lower")
    ax.set_title(f"axis {axis} | slice {initial}/{nr_slices - 1} | {nr_components} groups")
    ax.axis("off")

    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.04))
    slider = Slider(slider_ax, "slice", 0, nr_slices - 1, valinit=initial, valstep=1)

    def update(value: float) -> None:
        sl = int(value)
        image.set_data(colors[np.take(labels, sl, axis=axis)])
        ax.set_title(f"axis {axis} | slice {sl}/{nr_slices - 1} | {nr_components} groups")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def build_mask(volume: ClearVolume) -> ClearVolume:
    masked_data = np.where(volume.data > 0, 1, 0)

    return volume.copy_with(data=masked_data)

def get_largest_connected_component(volume: ClearVolume) -> ClearVolume:
    connectivity = np.ones((3, 3, 3), dtype=int)
    labels, nr_components = label(volume.data, structure=connectivity)

    if nr_components == 0:
        return volume

    # Component 0 is the background, so ignore it when counting voxels
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_label = int(np.argmax(component_sizes))

    largest_component = np.where(labels == largest_label, 1, 0)

    return volume.copy_with(data=largest_component)

def restore_original_intensities(
        tissue: ClearVolume,
        largest_component: ClearVolume,
    ) -> ClearVolume:
    cleaned_data = np.where(largest_component.data > 0, tissue.data, 0)
    return tissue.copy_with(data=cleaned_data)
