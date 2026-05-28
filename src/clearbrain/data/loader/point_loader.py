# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from pathlib import Path

from .file_loader import load_json
from .metadata_loader import load_metadata
from ...tissue.ClearTissue import ClearTissue


# ================================================================
# 1. Section: Points
# ================================================================
def load_points(path: Path, suffix: str):
    data = load_points_data(path, suffix)
    metadata = load_metadata(path)

    return ClearTissue(data, metadata)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Load the individual parts
# ──────────────────────────────────────────────────────
def load_points_data(path: Path, suffix: str) -> np.ndarray:
    file_path = path.parent / f"{path.stem}{suffix}.json"
    payload = load_json(file_path)

    if not isinstance(payload, list):
        raise TypeError(
            f"The loaded file did not contain a list of points ({type(payload)})"
        )

    points = np.asarray(payload, dtype=int)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"The loaded points need to be of shape (N, 3) ({points.shape})"
        )

    return points
