# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np



# ================================================================
# 1. Section: Functions
# ================================================================
def get_spinal_sections(
    points: np.ndarray,
    centerline: np.ndarray,
    nr_cuts: int,
    half_width: float,
    half_thickness: float
) -> tuple:
    # 1. Initializes the sections indexes
    start_idx = int(0.05 * len(centerline))
    end_idx   = int(0.95 * len(centerline))
    indices = np.linspace(start_idx, end_idx, nr_cuts, dtype=int)

    # 2. Get the vectors for each section
    all_prisms = []
    section_points = []
    centers = []
    for idx in indices:
        # 2.1. Get the centerline center position as a center ref
        centerline_pos = centerline[idx]
        centers.append(centerline_pos)

        # 2.2. Get the tangent vector
        tagent_vector = get_tangent_vector(centerline, idx, centerline_pos)

        # 2.3 Get the perpendicular basis vectors
        u, v = get_basis_vector(tagent_vector)

        # 2.4 Get the vertices of the prism
        vert = get_vertices(centerline_pos, tagent_vector, (u, v), half_width, half_thickness)

        # 2.5. Get the contained points
        d = points - centerline_pos
        inside = (
            (np.abs(d @ tagent_vector) <= half_thickness) &
            (np.abs(d @ u) <= half_width) &
            (np.abs(d @ v) <= half_width)
        )
        section_points.append(points[inside])

        # 2.6 Store the prism
        prism_corners = [corner.tolist() for corner in vert]
        all_prisms.append(prism_corners)

    return np.asarray(all_prisms), section_points, np.asarray(centers)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Vector Functions
# ──────────────────────────────────────────────────────
def get_tangent_vector(centerline: np.ndarray, idx: int, centerline_pos: np.ndarray) -> np.ndarray:
    if idx < len(centerline) - 1:
        tagent_vect = centerline[idx + 1] - centerline_pos
    else:
        tagent_vect = centerline_pos - centerline[idx - 1]
    tagent_vect /= np.linalg.norm(tagent_vect)

    return tagent_vect

def get_basis_vector(tagent_vect: np.ndarray) -> tuple:
    arbitrary = np.array([1., 0., 0.])
    if abs(np.dot(tagent_vect, arbitrary)) > 0.99:
        arbitrary = np.array([0., 1., 0.])
    u = np.cross(tagent_vect, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(tagent_vect, u)

    return u, v


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Helper Vertices Functions
# ──────────────────────────────────────────────────────
def get_vertices(
    centerline_pos: np.ndarray,
    tagent_vect: np.ndarray,
    uv: tuple,
    half_width: float,
    half_thickness: float
) -> list:
    direction = (centerline_pos - half_thickness * tagent_vect, centerline_pos + half_thickness * tagent_vect)

    return [
        direction[0] + half_width*uv[0] + half_width*uv[1],
        direction[0] + half_width*uv[0] - half_width*uv[1],
        direction[0] - half_width*uv[0] + half_width*uv[1],
        direction[0] - half_width*uv[0] - half_width*uv[1],
        direction[1]  + half_width*uv[0] + half_width*uv[1],
        direction[1]  + half_width*uv[0] - half_width*uv[1],
        direction[1]  - half_width*uv[0] + half_width*uv[1],
        direction[1]  - half_width*uv[0] - half_width*uv[1],
    ]
