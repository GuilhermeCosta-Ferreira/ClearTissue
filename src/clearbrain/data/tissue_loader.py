# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import numpy as np

from pathlib import Path

from .loader import load_json
from ..meta import Metadata
from ..tissue import ClearTissue


# ================================================================
# 1. Section: Points
# ================================================================
def load_points(path: Path):
    data = load_points_data(path)
    metadata = load_points_metadata(path)

    return ClearTissue(data, metadata)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Load the individual parts
# ──────────────────────────────────────────────────────
def load_points_data(path: Path) -> np.ndarray:
    payload = load_json(path)

    if not isinstance(payload, list):
        raise TypeError(f"The loaded file did not contain a list of points ({type(payload)})")

    points = np.asarray(payload, dtype=int)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"The loaded points need to be of shape (N, 3) ({points.shape})")

    return points

def load_points_metadata(path: Path):
    metadata_filepath = path.parent / f"{path.stem}_metadata{path.suffix}"

    if os.path.exists(metadata_filepath):
        return Metadata.from_path(metadata_filepath)
    else:
        print("No metadata found, creating a new one")
        return Metadata.from_filename(metadata_filepath)



# ================================================================
# 2. Section: Volume
# ================================================================
def load_volume(path: Path):
    data = load_volume_data(path)
    metadata = load_volume_metadata(path)


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Load the individual parts
# ──────────────────────────────────────────────────────
def load_volume_data(path: Path):
    pass

def load_volume_metadata(path: Path):
    pass
