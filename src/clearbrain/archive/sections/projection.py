# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np
from typing import cast

from .sections import get_basis_vector, get_tangent_vector


# ================================================================
# 1. Section: Functions
# ================================================================
def get_2d_sections(section_points, section_centers, centerline):
    projs = []
    for i, prism_points in enumerate(section_points):
        centerline_pos = section_centers[i]
        local_u, local_v = project_prims_2d(centerline_pos, centerline, prism_points)
        img, extent = prism_to_image(local_u, local_v)

        projs.append(img)

    return projs


def prism_to_image(local_u: np.ndarray, local_v: np.ndarray, pixel_size: float = 10):
    u_min, u_max = local_u.min(), local_u.max()
    v_min, v_max = local_v.min(), local_v.max()

    u_edges = np.arange(u_min, u_max + pixel_size, pixel_size)
    v_edges = np.arange(v_min, v_max + pixel_size, pixel_size)

    # H has shape (len(u_edges)-1, len(v_edges)-1)
    H, _, _ = np.histogram2d(local_u, local_v, bins=[u_edges, v_edges])

    # transpose so axis 0 is vertical for imshow
    img = H.T

    extent = [u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]]
    return img, extent


def project_prims_2d(centerline_pos, centerline, prism_points):
    idx = cast(int, np.argmin(np.linalg.norm(centerline - centerline_pos, axis=1)))

    # 2.2. Get the tangent vector
    tagent_vector = get_tangent_vector(centerline, idx, centerline_pos)

    # 2.3 Get the perpendicular basis vectors
    u, v = get_basis_vector(tagent_vector)

    # Project to 2D local coordinates (U-V plane)
    local_u = np.dot(prism_points - centerline_pos, u)
    local_v = np.dot(prism_points - centerline_pos, v)

    return local_u, local_v


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
