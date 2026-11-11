"""Axis projections and per-slice PCA of a 3D volume along any axis.

Pure numpy helpers behind the cord view figures. The "top" of an axis is its
high-index end: a viewer looks down from there towards index 0.
"""

# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from numpy.typing import NDArray

from .cord_profile import PcaProfiles, pca_elongation_profiles


# ================================================================
# 1. Section: Functions
# ================================================================
def foreground_extent(
    volume: NDArray, axis: int, threshold: float = 0
) -> tuple[int, int]:
    """First and last index along axis holding a voxel above threshold (inclusive)."""
    other = tuple(a for a in range(volume.ndim) if a != axis)
    idx = np.flatnonzero((volume > threshold).any(axis=other))
    if idx.size == 0:
        raise ValueError("volume has no voxel above the threshold")
    return int(idx[0]), int(idx[-1])


def top_half(volume: NDArray, axis: int, threshold: float = 0) -> NDArray:
    """Slab from the cord's centre to its far (high-index) edge along axis.

    The centre is the middle of the foreground extent, not of the array.
    """
    lo, hi = foreground_extent(volume, axis, threshold)
    centre = (lo + hi) // 2
    return np.take(volume, np.arange(centre, hi + 1), axis=axis)


def mean_projection(volume: NDArray, axis: int, threshold: float = 0) -> NDArray:
    """Mean intensity along axis over the top half only.

    Halving stops symmetric structure from cancelling out of the projection.
    """
    return top_half(volume, axis, threshold).mean(axis=axis)


def view_projection(volume: NDArray, axis: int, threshold: float = 0) -> NDArray:
    """Intensity of the first voxel above threshold met looking from the top.

    Rays are cast from the high-index end towards index 0; rays that hit
    nothing are 0.
    """
    from_top = np.flip(np.moveaxis(volume, axis, 0), axis=0)
    hit = from_top > threshold
    first = hit.argmax(axis=0)
    values = np.take_along_axis(from_top, first[None], axis=0)[0]
    return np.where(hit.any(axis=0), values, 0)


def cumsum_projection(
    volume: NDArray, axis: int, threshold: float = 0, level: float = 0.9
) -> NDArray:
    """Depth at which each ray's intensity cumsum saturates, over the top half.

    Per ray, intensities (background clipped to 0) are summed from the top edge
    towards the centre and normalised by the ray's total. The pixel is the
    depth, as a fraction of the half's thickness (0 = top edge, 1 = centre),
    where the normalised cumsum first reaches level. Rays with no signal are 0.
    """
    half = np.flip(np.moveaxis(top_half(volume, axis, threshold), axis, 0), axis=0)
    signal = np.where(half > threshold, half, 0).astype(float)
    csum = np.cumsum(signal, axis=0)
    total = csum[-1]
    reached = csum >= level * total[None]
    depth = reached.argmax(axis=0) / max(len(half) - 1, 1)
    return np.where(total > 0, depth, 0.0)


def pca_profiles_along_axis(volume: NDArray, axis: int) -> PcaProfiles:
    """Per-slice PCA of the cross-sections perpendicular to axis.

    The angle is measured from the last remaining in-plane axis, in the order
    the axes appear in the volume.
    """
    return pca_elongation_profiles(np.moveaxis(volume, axis, 0))
