# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scipy.ndimage import map_coordinates

from ...data import ClearVolume, SpinalCenterline



# ================================================================
# 1. Section: Functions
# ================================================================
def stretch_tissue(
    clear_volume: ClearVolume, centerline: SpinalCenterline, smooth_window: int
) -> ClearVolume:
    volume = clear_volume.data
    spinal_centers = centerline.data
    spinal_direction = centerline.smooth_derivative(smooth_window)
    nr_slices, _,  _ = volume.shape

    stretch_volume = np.zeros_like(volume)
    for sl in range(nr_slices):
        center = spinal_centers[sl]
        direction = spinal_direction[sl]

        img = extract_perpendicular_slice(
            volume=volume,
            center=center,
            direction=direction,
            order=0,
        )

        stretch_volume[sl, :, :] = img

    return clear_volume.copy_with(data=stretch_volume)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _unit_vector(v: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)

    if not np.isfinite(norm) or norm < eps:
        return None

    return v / norm

def _perpendicular_plane_basis(direction: np.ndarray):
    normal = _unit_vector(direction)

    if normal is None:
        return None, None

    # Prefer x-axis as one in-plane axis
    ref = np.array([0.0, 1.0, 0.0])

    # Project ref onto the plane perpendicular to normal
    u = ref - np.dot(ref, normal) * normal

    # If direction is too aligned with x, use z instead
    if np.linalg.norm(u) < 1e-8:
        ref = np.array([0.0, 0.0, 1.0])
        u = ref - np.dot(ref, normal) * normal

    u = _unit_vector(u)
    v = _unit_vector(np.cross(normal, u))  # type: ignore

    return u, v


def extract_perpendicular_slice(
    volume: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    direction = np.asarray(direction, dtype=float)

    _, height, width = volume.shape

    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(direction)):
        return np.full((height, width), np.nan, dtype=float)

    u, v = _perpendicular_plane_basis(direction)

    if u is None or v is None:
        return np.full((height, width), np.nan, dtype=float)

    # Coordinates centered around zero
    uu = np.arange(width) - (width - 1) / 2
    vv = np.arange(height) - (height - 1) / 2

    U, V = np.meshgrid(uu, vv, indexing="xy")

    # 3D coordinates of every pixel in the oblique plane
    coords = (
        center[:, None, None]
        + u[:, None, None] * U[None, :, :]
        + v[:, None, None] * V[None, :, :]
    )

    img = map_coordinates(
        volume,
        coords,
        order=order,
        mode="constant",
        cval=np.nan,
    )

    return img
