# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from .ClearPoints import ClearPoints



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SpinalCenterline(ClearPoints):
    def smooth_derivative(self, smoothing_factor: int = 1) -> np.ndarray:
        return get_smooth_derivative(self.data, smoothing_factor)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_smooth_derivative(coords: np.ndarray, smooth_window_size: int) -> np.ndarray:
    if smooth_window_size % 2 == 0:
        raise ValueError(f"Make the kernle size odd, you made {smooth_window_size}")

    kernel = np.ones(smooth_window_size) / smooth_window_size
    smoothed = np.empty_like(coords, dtype=float)
    pad = smooth_window_size // 2

    for dim in range(coords.shape[1]):
        padded = np.pad(coords[:, dim], pad_width=pad, mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")

    derivatives = get_coord_derivative(smoothed)

    return derivatives

def get_coord_derivative(coords: np.ndarray) -> np.ndarray:
    derivative = np.zeros_like(coords)

    derivative[1:] = coords[:-1] - coords[1:]
    derivative[0] = derivative[1]

    return derivative
