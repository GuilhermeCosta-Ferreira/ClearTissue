# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist
from dataclasses import dataclass

from ..data import SampleBatch, ClearVolume
from .AbstractTransformations import AbstractTransformation



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ContrastEnhanceSample(AbstractTransformation):
    background_sigma: float = 50.0
    pm_iterations: int = 10
    pm_kappa: float = 0.1
    pm_gamma: float = 0.1
    pm_option: int = 1
    clahe_clip_limit: float = 0.01
    clahe_kernel_size: int | tuple[int, ...] | None = None

    def apply(self, batch: SampleBatch) -> SampleBatch:
        background_tissue = background_subtraction(batch.tissue, self.background_sigma)

        perona_tissue = perona_malik(
            background_tissue,
            self.pm_iterations,
            self.pm_kappa,
            self.pm_gamma,
            self.pm_option,
        )

        enhanced_tissue_mask = clahe(
            perona_tissue,
            self.clahe_clip_limit,
            self.clahe_kernel_size,
        )

        # DEBUG: scroll through the enhanced slices (comment out)
        _debug_plot_slices(enhanced_tissue_mask.data, axis=0, title="enhanced")

        enhanced_tissue = enhance_white_matter(batch.tissue, enhanced_tissue_mask)
        _debug_plot_slices(enhanced_tissue.data, axis=0, title="summed")

        return batch.copy_with(tissue=enhanced_tissue)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Pipeline Stages
# ──────────────────────────────────────────────────────
def background_subtraction(volume: ClearVolume, sigma: float) -> ClearVolume:
    data = _normalize_unit_range(volume.data)
    background = gaussian_filter(data, sigma=sigma)
    corrected = np.clip(data - background, 0.0, None)

    return volume.copy_with(data=corrected)


def perona_malik(
    volume: ClearVolume,
    iterations: int,
    kappa: float,
    gamma: float,
    option: int,
) -> ClearVolume:
    data = volume.data
    diffused = _anisotropic_diffusion(data, iterations, kappa, gamma, option)

    return volume.copy_with(data=diffused)


def clahe(
    volume: ClearVolume,
    clip_limit: float,
    kernel_size: int | tuple[int, ...] | None,
) -> ClearVolume:
    data = _normalize_unit_range(volume.data)
    equalized = equalize_adapthist(
        data,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    )

    return volume.copy_with(data=equalized)

def enhance_white_matter(
    volume: ClearVolume,
    mask: ClearVolume,
) -> ClearVolume:
    mask_data = mask.data.copy()
    volume_data = volume.data.copy()

    # 1. Normalize mask data
    mask_data = _normalize_unit_range(mask_data)
    print(np.max(mask_data))
    mask_data = mask_data * np.max(volume_data)

    # 2. Add the mask data to the volume data
    enhanced = volume_data + mask_data * 0.1
    enhanced = np.where(enhanced > 10, enhanced, 0)
    enhanced = np.where(np.isnan(enhanced), 0, enhanced)

    return volume.copy_with(data=enhanced.astype(np.uint16))



# ──────────────────────────────────────────────────────
# 1.2 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _normalize_unit_range(data: np.ndarray) -> np.ndarray:
    minimum = float(data.min())
    maximum = float(data.max())
    span = maximum - minimum
    if span == 0.0:
        return np.zeros_like(data)

    return (data - minimum) / span


def _anisotropic_diffusion(
    data: np.ndarray,
    iterations: int,
    kappa: float,
    gamma: float,
    option: int,
) -> np.ndarray:
    if option not in (1, 2):
        raise ValueError(f"Perona-Malik option must be 1 or 2, got {option}")

    diffused = data.copy()
    for _ in range(iterations):
        update = np.zeros_like(diffused)
        for axis in range(diffused.ndim):
            for shift in (-1, 1):
                delta = np.roll(diffused, shift, axis=axis) - diffused
                update += _conduction(delta, kappa, option) * delta
        diffused += gamma * update

    return diffused


def _conduction(delta: np.ndarray, kappa: float, option: int) -> np.ndarray:
    if option == 1:
        return np.exp(-((delta / kappa) ** 2))

    return 1.0 / (1.0 + (delta / kappa) ** 2)


# ──────────────────────────────────────────────────────
# 1.3 Subsection: Debug Plots
# ──────────────────────────────────────────────────────
def _debug_plot_slices(
    volume_data: np.ndarray,
    axis: int = 0,
    title: str = "",
) -> None:
    nr_slices = volume_data.shape[axis]
    initial = nr_slices // 2

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.15)

    image = ax.imshow(
        np.take(volume_data, initial, axis=axis),
        cmap="gray",
        origin="lower",
    )
    ax.set_title(f"{title} | axis {axis} | slice {initial}/{nr_slices - 1}")
    ax.axis("off")

    slider_ax = plt.axes((0.2, 0.05, 0.6, 0.04))
    slider = Slider(slider_ax, "slice", 0, nr_slices - 1, valinit=initial, valstep=1)

    def _update(value: float) -> None:
        index = int(value)
        image.set_data(np.take(volume_data, index, axis=axis))
        ax.set_title(f"{title} | axis {axis} | slice {index}/{nr_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    plt.show()
