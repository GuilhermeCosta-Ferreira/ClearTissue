# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import json

import numpy as np

from pathlib import Path


# ================================================================
# 1. Section: Functions
# ================================================================
def load_points(filepath: Path) -> np.ndarray:
    # A. Makes sure there is a file
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # 1. Extracts the data from the JSON
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        points = np.array(data, dtype=float)
    except (ValueError, TypeError):
        points = data

    # B. Makes sure the file is not empty and the shape is ok
    if len(points) == 0:
        raise ValueError(f"Points file is empty: {filepath}")

    return points
