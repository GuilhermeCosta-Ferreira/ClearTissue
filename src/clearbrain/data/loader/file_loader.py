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
def load_json(path: Path) -> dict | list:
    # A. Makes sure there is a file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(str(path), "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload

def load_npy(path: Path) -> np.ndarray:
    # A. Makes sure there is a file
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return np.load(path)
