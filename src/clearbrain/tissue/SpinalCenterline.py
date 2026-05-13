# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from dataclasses import dataclass

from ..data.Metadata import Metadata


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass(frozen=True)
class SpinalCenterline:
    volume: np.ndarray
    points: np.ndarray
    metadata: Metadata

    @property
    def derivative(self) -> np.ndarray:
        return get_coord_derivative(self.points)

    def smooth_derivative(self, window_size) -> np.ndarray:
        return get_smooth_derivative(self.points, window_size)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_coord_derivative(coords: np.ndarray) -> np.ndarray:
    derivative = np.zeros_like(coords)

    derivative[1:] = coords[:-1] - coords[1:]
    derivative[0] = derivative[1]

    return derivative


def get_smooth_derivative(coords: np.ndarray, window_size) -> np.ndarray:
    if window_size % 2 == 0:
        raise ValueError(f"Make the kernle size odd, you made {window_size}")

    kernel = np.ones(window_size) / window_size
    smoothed = np.empty_like(coords, dtype=float)
    pad = window_size // 2

    for dim in range(coords.shape[1]):
        padded = np.pad(coords[:, dim], pad_width=pad, mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")

    derivatives = get_coord_derivative(smoothed)

    return derivatives
